"""Patient-level scoring for fixed-fold B-mode probability files."""
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value.strip()


def _label(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError('patient_label must be 0 or 1') from error
    if not np.isfinite(number) or number not in (0., 1.):
        raise ValueError('patient_label must be 0 or 1')
    return int(number)


def validate_manifest(records):
    """Require explicit patient labels; never infer image-level pathology."""
    if not records:
        raise ValueError('The manifest is empty')
    normalized = []
    samples = set()
    patient_labels = {}
    for record in records:
        try:
            sample = _identifier(record['sample_id'], 'sample_id')
            patient = _identifier(record['patient_id'], 'patient_id')
            label = _label(record['patient_label'])
        except KeyError as error:
            raise ValueError(f'Missing manifest field: {error.args[0]}') from error
        if sample in samples:
            raise ValueError(f'Duplicate sample_id: {sample}')
        if patient in patient_labels and patient_labels[patient] != label:
            raise ValueError(f'Conflicting patient_label for patient_id: {patient}')
        samples.add(sample)
        patient_labels[patient] = label
        normalized.append({'sample_id': sample, 'patient_id': patient, 'patient_label': label})
    return normalized


def align_fold_predictions(records, rows):
    """Join a complete probability file to the manifest by sample ID."""
    manifest = validate_manifest(records)
    expected = {row['sample_id']: row for row in manifest}
    values = {}
    for row in rows:
        try:
            sample = _identifier(row['sample_id'], 'sample_id')
            probability = float(row['probability'])
        except KeyError as error:
            raise ValueError(f'Missing prediction field: {error.args[0]}') from error
        except (TypeError, ValueError) as error:
            raise ValueError('Predictions require a sample_id and numeric probability') from error
        if sample in values:
            raise ValueError(f'Duplicate prediction sample_id: {sample}')
        if sample not in expected:
            raise ValueError(f'Unexpected prediction sample_id: {sample}')
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f'Probability outside [0,1] or nonfinite: {sample}')
        for key in ('patient_id', 'patient_label'):
            if key in row:
                actual = _label(row[key]) if key == 'patient_label' else _identifier(row[key], key)
                if actual != expected[sample][key]:
                    raise ValueError(f'Prediction {key} disagrees with manifest: {sample}')
        values[sample] = probability
    missing = set(expected) - set(values)
    if missing:
        raise ValueError(f'Missing predictions for {len(missing)} manifest images')
    return np.asarray([values[row['sample_id']] for row in manifest], dtype=float)


def aggregate_series(records, fold_probabilities, *, expected_folds=5):
    """Average folds for each image, then take the maximum within each patient."""
    manifest = validate_manifest(records)
    if not isinstance(expected_folds, int) or isinstance(expected_folds, bool) or expected_folds < 1:
        raise ValueError('expected_folds must be a positive integer')
    probabilities = np.asarray(fold_probabilities, dtype=float)
    if probabilities.shape != (expected_folds, len(manifest)):
        raise ValueError(f'Expected probabilities with shape ({expected_folds}, {len(manifest)})')
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Every fold must supply finite probabilities in [0,1] for every image')
    image_probabilities = probabilities.mean(axis=0)
    groups = {}
    for record, probability in zip(manifest, image_probabilities):
        group = groups.setdefault(record['patient_id'], {
            'patient_id': record['patient_id'], 'patient_label': record['patient_label'],
            'views': 0, 'probability': 0.,
        })
        group['views'] += 1
        group['probability'] = max(group['probability'], float(probability))
    patients = [groups[key] for key in sorted(groups)]
    return patients


def _binary_metrics(labels, probabilities):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    positive = y == 1
    predicted = p >= .5
    tp = int((positive & predicted).sum())
    fn = int((positive & ~predicted).sum())
    tn = int((~positive & ~predicted).sum())
    fp = int((~positive & predicted).sum())
    return {'auc': float(roc_auc_score(y, p)), 'f1': 2 * tp / (2 * tp + fp + fn),
            'accuracy': (tp + tn) / len(y), 'precision': tp / (tp + fp) if tp + fp else 0.,
            'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp),
            'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp}


def _malignant_metrics(probabilities):
    p = np.asarray(probabilities, dtype=float)
    tp = int((p >= .5).sum())
    interval = binomtest(tp, len(p)).proportion_ci(confidence_level=.95, method='exact')
    return {'tp': tp, 'fn': len(p) - tp, 'sensitivity': tp / len(p),
            'sensitivity_ci_low': float(interval.low), 'sensitivity_ci_high': float(interval.high)}


def score_series(records, fold_probabilities, *, cohort_kind, expected_folds=5,
                 bootstrap_repetitions=2000, seed=42, reference_fold_probabilities=None):
    """Return metrics and patient rows without changing any threshold or membership.

    ``malignant-only`` uses exact sensitivity intervals and does not calculate
    AUC, specificity, precision, F1 or overall accuracy; a supplied reference
    adds an exact McNemar comparison. ``binary`` requires both patient classes
    and optionally returns class-stratified, shared-draw bootstrap intervals.
    A bootstrap count of zero explicitly disables binary intervals.
    """
    if cohort_kind not in ('binary', 'malignant-only'):
        raise ValueError('cohort_kind must be binary or malignant-only')
    if not isinstance(bootstrap_repetitions, int) or isinstance(bootstrap_repetitions, bool) or bootstrap_repetitions < 0:
        raise ValueError('bootstrap_repetitions must be a nonnegative integer')
    patients = aggregate_series(records, fold_probabilities, expected_folds=expected_folds)
    labels = np.asarray([row['patient_label'] for row in patients], dtype=int)
    probability = np.asarray([row['probability'] for row in patients], dtype=float)
    if cohort_kind == 'malignant-only' and set(labels) != {1}:
        raise ValueError('malignant-only evaluation requires every patient_label to equal 1')
    if cohort_kind == 'binary' and set(labels) != {0, 1}:
        raise ValueError('binary evaluation requires both patient classes')
    reference = None
    if reference_fold_probabilities is not None:
        other = aggregate_series(records, reference_fold_probabilities, expected_folds=expected_folds)
        if [(r['patient_id'], r['patient_label'], r['views']) for r in other] != [(r['patient_id'], r['patient_label'], r['views']) for r in patients]:
            raise ValueError('Reference patient membership differs')
        reference = np.asarray([row['probability'] for row in other], dtype=float)
    metric = _malignant_metrics if cohort_kind == 'malignant-only' else lambda p: _binary_metrics(labels, p)
    report = {'cohort_kind': cohort_kind, 'images': len(records), 'patients': len(patients),
              'folds': expected_folds, 'fold_aggregation': 'mean_probability_per_image',
              'patient_aggregation': 'maximum_image_probability', 'threshold': .5,
              'threshold_ties': 'positive', 'metrics': metric(probability)}
    if reference is not None:
        report['reference_metrics'] = metric(reference)
        if cohort_kind == 'malignant-only':
            missed = probability < .5
            reference_missed = reference < .5
            primary_only = int(np.sum(missed & ~reference_missed))
            reference_only = int(np.sum(~missed & reference_missed))
            discordant = primary_only + reference_only
            report['paired_comparison'] = {
                'primary_only_missed': primary_only,
                'reference_only_missed': reference_only,
                'both_missed': int(np.sum(missed & reference_missed)),
                'neither_missed': int(np.sum(~missed & ~reference_missed)),
                'exact_mcnemar_p': (float(binomtest(primary_only, discordant, p=.5).pvalue)
                                    if discordant else 1.),
            }
    if cohort_kind == 'binary':
        uncertainty = {'seed': int(seed), 'requested_resamples': bootstrap_repetitions,
                       'sampling': 'class_stratified_patient_bootstrap',
                       'interval_method': 'percentile', 'confidence_level': .95,
                       'valid_resamples': 0, 'intervals': None}
        if bootstrap_repetitions:
            rng = np.random.default_rng(seed)
            values, differences = [], []
            class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
            for _ in range(bootstrap_repetitions):
                index = np.concatenate([rng.choice(group, size=len(group), replace=True)
                                        for group in class_indices])
                actual = _binary_metrics(labels[index], probability[index])
                values.append([actual['auc'], actual['f1']])
                if reference is not None:
                    other = _binary_metrics(labels[index], reference[index])
                    differences.append([actual['auc'] - other['auc'], actual['f1'] - other['f1']])
            uncertainty['valid_resamples'] = len(values)
            bounds = np.quantile(values, [.025, .975], axis=0)
            uncertainty['intervals'] = {key: bounds[:, j].tolist() for j, key in enumerate(('auc', 'f1'))}
            if differences:
                bounds = np.quantile(differences, [.025, .975], axis=0)
                uncertainty['paired_difference_intervals'] = {key: bounds[:, j].tolist() for j, key in enumerate(('auc', 'f1'))}
        report['uncertainty'] = uncertainty
    return report, patients
