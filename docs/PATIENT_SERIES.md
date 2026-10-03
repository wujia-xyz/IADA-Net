# Patient-series evaluation

This utility implements a fixed aggregation order: average model-fold probabilities for each image, then take the largest image probability for each patient. The decision rule is `probability >= 0.5`. The default model count is five, and the mixed-class bootstrap seed is 42. No threshold is fitted from the evaluation cohort.

## Inputs

The manifest describes patient membership and patient-level ground truth:

```csv
sample_id,patient_id,patient_label
image_a1,patient_a,1
image_a2,patient_a,1
image_b1,patient_b,1
```

Each fold prediction file contains:

```csv
sample_id,probability
image_a1,0.41
image_a2,0.62
image_b1,0.55
```

These are synthetic examples. Supply one file from each selected fold model in fold order. The command defaults to five files; an explicitly different protocol can specify `--expected-folds`. Each path must be distinct. A sample may occur once per file, and every file must cover all manifest samples. Row order may differ: alignment uses sample IDs. Duplicate, missing, unexpected or nonfinite predictions cause an error rather than silently removing cases. If a prediction file supplies patient IDs or patient labels, they must agree with the manifest.

Patient labels must be 0 or 1 and consistent within a patient. The tool requires an explicit `patient_label` column; it does not substitute a per-image `label` column or infer that every view depicts a malignant lesion. No image file is required by this scoring command.

## Malignant-only series

```bash
python -m iada.score_patient_series --manifest patient_manifest.csv --fold-predictions fold1.csv fold2.csv fold3.csv fold4.csv fold5.csv --cohort-kind malignant-only --output outputs/series_metrics.json --patient-output outputs/series_patients.csv
```

Every patient label must equal 1. The summary contains patient/image counts, TP, FN, sensitivity and a two-sided exact 95% Clopper–Pearson interval. With `--reference-fold-predictions`, it also reports patients missed only by the primary model, only by the reference, by both, or by neither, plus the two-sided exact McNemar p-value. Zero discordant patients give p=1. It does not report AUC, specificity, precision, F1 or overall accuracy for this single-class cohort. Patient rows are written only when `--patient-output` is supplied; they are not printed to the console. Keep private clinical manifests, predictions and patient output out of public releases.

## Mixed-class cohorts and paired differences

Use `--cohort-kind binary` only when both patient classes are present. The summary includes AUC, F1, accuracy, precision, sensitivity, specificity and confusion counts. The default 2,000 class-stratified patient resamples use seed 42: benign and malignant patients are sampled separately with replacement, preserving each class count in every draw. The 95% intervals use the 2.5th and 97.5th percentiles. Patient IDs are sorted lexically before resampling. Set `--bootstrap 0` when requesting point metrics without intervals; the output records that intervals were not calculated.

For a paired comparison, supply the comparator's five files through `--reference-fold-predictions`. The same patient draws are used for the primary and reference methods. Reported difference intervals are **primary minus reference**. Identical inputs therefore give zero paired differences. Both methods use the identical complete manifest and aggregation rule.

```bash
python -m iada.score_patient_series --manifest public_patients.csv --fold-predictions query1.csv query2.csv query3.csv query4.csv query5.csv --reference-fold-predictions base1.csv base2.csv base3.csv base4.csv base5.csv --cohort-kind binary --output outputs/paired_metrics.json
```

The JSON records the aggregation order, positive tie convention, model count and uncertainty settings. The scorer does not calculate file hashes or create audit receipts. Checkpoint choice and preprocessing belong to the prediction-producing experiment.

## Historical results

The earlier `iada.evaluate` and `iada.evaluate_heads` APIs retain their recorded patient-mean and threshold-tie conventions. Use them to reproduce their archived result series. Changing the aggregation rule creates a different evaluation; do not combine metrics from the two protocols or choose a rule after inspecting which gives the preferred result.
