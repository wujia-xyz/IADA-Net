import numpy as np
import pytest
from iada.metrics import aggregate_patients,metrics,bootstrap

def test_threshold_tie_is_benign():
    m=metrics([0,1],[.5,.8]);assert m['accuracy']==1 and m['auc']==1

def test_patient_mean_and_conflicting_labels():
    rows=[{'case_id':'a','label':0},{'case_id':'a','label':0},{'case_id':'b','label':1}]
    ids,y,p=aggregate_patients(rows,[.1,.3,.9]);np.testing.assert_allclose(p,[.2,.9]);assert ids==['a','b']
    rows[1]['label']=1
    with pytest.raises(ValueError):aggregate_patients(rows,[.1,.3,.9])

def test_identical_paired_predictions_have_zero_difference():
    result=bootstrap([0,0,1,1],[.1,.4,.7,.9],reference=[.1,.4,.7,.9],repetitions=50)
    assert result['intervals']=={'auc':[0.,0.],'f1':[0.,0.]}

def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError):metrics([0,1],[.1,np.nan])
