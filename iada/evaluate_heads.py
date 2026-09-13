"""Evaluate matched heads on clean or reordered cached external features."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from .heads import build_head
from .data import read_records
from .metrics import aggregate_patients,metrics,bootstrap
from .probes import permutation_indices

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--checkpoints',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--condition',choices=['clean','lateral_shuffle','depth_shuffle','depth_reverse','transpose'],default='clean')
    p.add_argument('--device',default='cpu');p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args(argv)
    torch.set_num_threads(4);records=read_records(a.manifest);features=np.load(a.features,mmap_mode='r')
    metadata=json.loads(Path(a.features).with_suffix('.json').read_text())
    if metadata['sample_ids']!=[r['sample_id'] for r in records]:p.error('Cache and manifest order differ')
    base=np.array(features[:,0] if features.ndim==4 else features,copy=True)
    if a.condition!='clean':
        for i,r in enumerate(records):
            idx=permutation_indices(a.dataset,r.get('case_id') or r.get('patient_id') or r['sample_id'])[a.condition]
            base[i,1:]=base[i,1:][idx]
    ensemble=[];kinds=[]
    for checkpoint in a.checkpoints:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True);kinds.append(state['kind'])
        if len(set(kinds))!=1:p.error('Each ensemble must contain the same head kind')
        model=build_head(state['kind']).to(a.device).eval();model.load_state_dict(state['head_state_dict'],strict=True);pred=[]
        with torch.inference_mode():
            for start in range(0,len(base),32):pred.extend(model(torch.from_numpy(base[start:start+32]).to(a.device)).softmax(-1)[:,1].cpu().tolist())
        ensemble.append(pred)
    ids,y,probs=aggregate_patients(records,np.mean(ensemble,axis=0));report={'condition':a.condition,'kind':kinds[0],'patients':len(ids),
        'metrics':metrics(y,probs),'uncertainty':bootstrap(y,probs,repetitions=a.bootstrap,seed=int.from_bytes(hashlib.sha256(('eaai-matched-case-bootstrap-v1:'+a.dataset).encode()).digest()[:8],'big'))}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
