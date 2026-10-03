import csv,json
import numpy as np
import pytest
from scipy.stats import beta
from iada.patient_series import aggregate_series,align_fold_predictions,score_series,validate_manifest
from iada.score_patient_series import main


def manifest():
    return [{'sample_id':'a1','patient_id':'a','patient_label':1},
            {'sample_id':'a2','patient_id':'a','patient_label':1},
            {'sample_id':'b1','patient_id':'b','patient_label':1}]


def test_fold_mean_precedes_patient_max_and_tie_is_positive():
    probabilities=np.asarray([[1,0,.5],[1,0,.5],[0,1,.5],[0,1,.5],[0,0,.5]])
    report,patients=score_series(manifest(),probabilities,cohort_kind='malignant-only')
    assert [p['probability'] for p in patients]==[.4,.5]
    assert [p['views'] for p in patients]==[2,1]
    assert report['metrics']['tp']==1 and report['metrics']['fn']==1
    assert report['patients']==2 and report['images']==3
    assert report['metrics']['sensitivity']==.5
    assert report['metrics']['sensitivity_ci_low']==pytest.approx(beta.ppf(.025,1,2),abs=1e-12)
    assert report['metrics']['sensitivity_ci_high']==pytest.approx(beta.ppf(.975,2,1),abs=1e-12)
    assert not {'auc','f1','accuracy','precision','specificity'}&set(report['metrics'])
    assert 'uncertainty' not in report


@pytest.mark.parametrize('value',[0.,1.])
def test_exact_interval_endpoints(value):
    records=[{'sample_id':str(i),'patient_id':str(i),'patient_label':1} for i in range(4)]
    report,_=score_series(records,np.full((5,4),value),cohort_kind='malignant-only')
    m=report['metrics']
    assert m['tp']==int(4*value) and m['fn']==4-int(4*value)
    if value==0:
        assert m['sensitivity_ci_low']==0
        assert m['sensitivity_ci_high']==pytest.approx(1-.025**.25)
    else:
        assert m['sensitivity_ci_high']==1
        assert m['sensitivity_ci_low']==pytest.approx(.025**.25)


def test_id_join_reorders_predictions_and_checks_coverage():
    records=manifest()
    rows=[{'sample_id':'b1','probability':.7},{'sample_id':'a2','probability':.6},{'sample_id':'a1','probability':.2}]
    assert align_fold_predictions(records,rows).tolist()==[.2,.6,.7]
    with pytest.raises(ValueError,match='Missing predictions'):align_fold_predictions(records,rows[:-1])
    with pytest.raises(ValueError,match='Duplicate prediction'):align_fold_predictions(records,rows+[rows[0]])
    with pytest.raises(ValueError,match='Unexpected prediction'):align_fold_predictions(records,rows+[{'sample_id':'x','probability':.5}])
    rows[0]['patient_id']='wrong'
    with pytest.raises(ValueError,match='disagrees'):align_fold_predictions(records,rows)


def test_manifest_never_treats_image_label_as_patient_label():
    record={'sample_id':'a','patient_id':'p','label':1}
    with pytest.raises(ValueError,match='patient_label'):validate_manifest([record])
    record['patient_label']=0
    assert validate_manifest([record])[0]['patient_label']==0
    record['patient_label']=.5
    with pytest.raises(ValueError,match='patient_label'):validate_manifest([record])
    record['patient_label']=0
    with pytest.raises(ValueError,match='Duplicate sample_id'):validate_manifest([record,record])
    other={**record,'sample_id':'b','patient_label':1}
    with pytest.raises(ValueError,match='Conflicting'):validate_manifest([record,other])


@pytest.mark.parametrize('invalid',[np.nan,np.inf,-.01,1.01])
def test_bad_probabilities_fail_without_dropping_images(invalid):
    values=np.full((5,3),.5);values[2,1]=invalid
    with pytest.raises(ValueError,match='finite probabilities'):aggregate_series(manifest(),values)


def test_fold_count_and_cohort_kind_are_explicit():
    with pytest.raises(ValueError,match='shape'):aggregate_series(manifest(),np.ones((1,3)))
    with pytest.raises(ValueError,match='both patient classes'):score_series(manifest(),np.ones((5,3)),cohort_kind='binary')
    records=manifest();records[2]['patient_label']=0
    with pytest.raises(ValueError,match='every patient_label'):score_series(records,np.ones((5,3)),cohort_kind='malignant-only')


def test_binary_pairing_uses_identical_draws_and_seed42():
    records=[{'sample_id':str(i),'patient_id':str(i),'patient_label':int(i>=3)} for i in range(6)]
    values=np.tile([.1,.4,.6,.3,.7,.9],(5,1))
    first,_=score_series(records,values,cohort_kind='binary',bootstrap_repetitions=30,reference_fold_probabilities=values)
    second,_=score_series(records,values,cohort_kind='binary',bootstrap_repetitions=30,reference_fold_probabilities=values)
    assert first==second
    assert first['uncertainty']['seed']==42 and first['uncertainty']['valid_resamples']==30
    assert first['uncertainty']['paired_difference_intervals']=={'auc':[0.,0.],'f1':[0.,0.]}


def test_binary_bootstrap_preserves_imbalanced_patient_class_counts(monkeypatch):
    import iada.patient_series as scoring
    records=[{'sample_id':str(i),'patient_id':str(i),'patient_label':int(i==4)} for i in range(5)]
    values=np.tile([.1,.3,.6,.8,.7],(5,1))
    counts=[]
    original=scoring._binary_metrics
    def capture(labels,probability):
        counts.append(np.bincount(labels,minlength=2).tolist())
        return original(labels,probability)
    monkeypatch.setattr(scoring,'_binary_metrics',capture)
    report,_=score_series(records,values,cohort_kind='binary',bootstrap_repetitions=25)
    assert len(counts)==26 and all(count==[4,1] for count in counts)
    assert report['uncertainty']['sampling']=='class_stratified_patient_bootstrap'
    assert report['uncertainty']['valid_resamples']==25


def test_clinical_pairing_reports_exact_mcnemar_and_zero_discordance():
    records=[{'sample_id':str(i),'patient_id':str(i),'patient_label':1} for i in range(8)]
    primary=np.tile([.2]*6+[.8]*2,(5,1))
    reference=np.full((5,8),.8)
    report,_=score_series(records,primary,cohort_kind='malignant-only',reference_fold_probabilities=reference)
    assert report['paired_comparison']=={
        'primary_only_missed':6,'reference_only_missed':0,'both_missed':0,'neither_missed':2,
        'exact_mcnemar_p':pytest.approx(.03125),
    }
    identical,_=score_series(records,primary,cohort_kind='malignant-only',reference_fold_probabilities=primary)
    assert identical['paired_comparison']['both_missed']==6
    assert identical['paired_comparison']['exact_mcnemar_p']==1.


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def test_cli_emits_only_requested_outputs_and_never_overwrites_sources(tmp_path,capsys):
    source=tmp_path/'manifest.csv';write_csv(source,manifest())
    paths=[]
    for fold in range(5):
        path=tmp_path/f'fold{fold+1}.csv';write_csv(path,[{'sample_id':r['sample_id'],'probability':.5} for r in manifest()]);paths.append(str(path))
    output=tmp_path/'metrics.json';patients=tmp_path/'patients.csv'
    args=['--manifest',str(source),'--fold-predictions',*paths,'--cohort-kind','malignant-only','--output',str(output),'--patient-output',str(patients)]
    main(args)
    report=json.loads(output.read_text());assert report['metrics']['tp']==2 and report['folds']==5
    assert 'input_sha256' not in report
    assert patients.exists() and 'patient_id' not in capsys.readouterr().out
    original=source.read_bytes()
    with pytest.raises(SystemExit):main(['--manifest',str(source),'--fold-predictions',*paths,'--cohort-kind','malignant-only','--output',str(source)])
    assert source.read_bytes()==original
