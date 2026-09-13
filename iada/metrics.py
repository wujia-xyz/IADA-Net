"""Image/patient metrics and paired bootstrap intervals."""
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,precision_score,recall_score,confusion_matrix

def metrics(labels,probabilities):
    y=np.asarray(labels,dtype=int);p=np.asarray(probabilities,dtype=float)
    if not len(y) or len(y)!=len(p) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Expected equally sized labels and finite probabilities in [0,1]')
    if set(np.unique(y))!={0,1}:raise ValueError('Both classes are required for AUC')
    pred=p>.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'accuracy':float(accuracy_score(y,pred)),'auc':float(roc_auc_score(y,p)),
            'recall':float(recall_score(y,pred,zero_division=0)),'specificity':float(tn/(tn+fp)),
            'precision':float(precision_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def aggregate_patients(records,probabilities):
    if len(records)!=len(probabilities):raise ValueError('Prediction count differs from manifest')
    groups={}
    for i,(r,p) in enumerate(zip(records,probabilities)):
        key=str(r.get('case_id') or r.get('patient_id') or r.get('sample_id') or i)
        g=groups.setdefault(key,{'label':r['label'],'p':[]})
        if g['label']!=r['label']:raise ValueError(f'Conflicting labels within patient {key}')
        g['p'].append(float(p))
    ids=sorted(groups,key=lambda x:(0,int(x)) if x.isdecimal() else (1,x))
    return ids,np.array([groups[k]['label'] for k in ids]),np.array([np.mean(groups[k]['p']) for k in ids])

def bootstrap(labels,probabilities,reference=None,repetitions=2000,seed=20260909):
    y=np.asarray(labels);p=np.asarray(probabilities);ref=None if reference is None else np.asarray(reference)
    metrics(y,p)
    if ref is not None:metrics(y,ref)
    rng=np.random.default_rng(seed);samples={'auc':[],'f1':[]}
    draws=rng.multinomial(len(y),np.full(len(y),1/len(y)),size=repetitions)
    for weights in draws:
        idx=np.repeat(np.arange(len(y)),weights)
        if len(np.unique(y[idx]))<2:continue
        a=metrics(y[idx],p[idx]);b={} if ref is None else metrics(y[idx],ref[idx])
        for k in samples:samples[k].append(a[k]-b.get(k,0.))
    if not samples['auc']:raise ValueError('No two-class bootstrap resamples')
    return {'intervals':{k:np.quantile(v,[.025,.975]).tolist() for k,v in samples.items()},
            'valid_resamples':len(samples['auc']),'requested_resamples':repetitions,'seed':seed,'paired_difference':ref is not None}
