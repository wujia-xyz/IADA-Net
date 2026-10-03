"""Score saved fold probabilities using the current patient-series protocol."""
import argparse,csv,json
from pathlib import Path
from .patient_series import validate_manifest,align_fold_predictions,score_series


def _read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--fold-predictions',nargs='+',required=True)
    parser.add_argument('--reference-fold-predictions',nargs='+')
    parser.add_argument('--cohort-kind',choices=['binary','malignant-only'],required=True)
    parser.add_argument('--expected-folds',type=int,default=5)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',required=True)
    parser.add_argument('--patient-output')
    args=parser.parse_args(argv)
    records=validate_manifest(_read(args.manifest))
    groups=[args.fold_predictions]+([args.reference_fold_predictions] if args.reference_fold_predictions else [])
    for paths in groups:
        if len(paths)!=args.expected_folds:parser.error('Prediction-file count must equal --expected-folds')
        if len({Path(p).resolve() for p in paths})!=len(paths):parser.error('Each fold must use a distinct prediction file')
    source_paths=[Path(args.manifest),*(Path(p) for group in groups for p in group)]
    outputs=[Path(args.output)]+([Path(args.patient_output)] if args.patient_output else [])
    resolved_sources={p.resolve() for p in source_paths}
    if len({p.resolve() for p in outputs})!=len(outputs):parser.error('Summary and patient outputs must be different files')
    if any(p.resolve() in resolved_sources for p in outputs):parser.error('An output must not overwrite a source file')
    predictions=[align_fold_predictions(records,_read(p)) for p in args.fold_predictions]
    reference=None if not args.reference_fold_predictions else [align_fold_predictions(records,_read(p)) for p in args.reference_fold_predictions]
    report,patients=score_series(records,predictions,cohort_kind=args.cohort_kind,expected_folds=args.expected_folds,
        bootstrap_repetitions=args.bootstrap,seed=args.seed,reference_fold_probabilities=reference)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    if args.patient_output:
        path=Path(args.patient_output);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['patient_id','patient_label','views','probability']);writer.writeheader();writer.writerows(patients)
    print(json.dumps(report,allow_nan=False))


if __name__=='__main__':main()
