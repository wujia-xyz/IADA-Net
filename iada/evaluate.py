"""Average fold probabilities, aggregate patients, and estimate confidence intervals."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .checkpoints import load_model
from .data import Images,read_records
from .metrics import aggregate_patients,metrics,bootstrap

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--adapters',nargs='+');p.add_argument('--manifest',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--dataset',required=True);p.add_argument('--bootstrap-seed',type=int);a=p.parse_args(argv)
    if a.adapters and len(a.adapters)!=len(a.checkpoints):p.error('Supply one adapter per base checkpoint')
    torch.set_num_threads(4);records=read_records(a.manifest);loader=DataLoader(Images(records,a.data_root),batch_size=16,num_workers=0)
    ensemble=[]
    for i,path in enumerate(a.checkpoints):
        model,_=load_model(path,a.device,None if a.adapters is None else a.adapters[i]);pred=[]
        with torch.inference_mode():
            for x,_,_ in loader:pred.extend(model(x.to(a.device)).float().softmax(-1)[:,1].cpu().tolist())
        ensemble.append(pred);del model
    ids,y,probs=aggregate_patients(records,np.mean(ensemble,axis=0))
    seed=a.bootstrap_seed if a.bootstrap_seed is not None else int.from_bytes(hashlib.sha256(f'external-v1-case-bootstrap:{a.dataset}:all:{len(ids)}'.encode()).digest()[:8],'big')
    report={'metrics':metrics(y,probs),'uncertainty':bootstrap(y,probs,repetitions=a.bootstrap,seed=seed),
            'patients':len(ids),'images':len(records),'ensemble_size':len(ensemble)}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
