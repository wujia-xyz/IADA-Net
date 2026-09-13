"""Train a capacity-matched head on an existing common feature cache."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from .heads import build_head,KINDS
from .data import read_records
from .metrics import metrics
from .cache_features import stable_seed

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--kind',choices=KINDS,required=True);p.add_argument('--fold',type=int,choices=range(1,6),required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=100);a=p.parse_args(argv)
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):p.error('Use a new empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    rows=read_records(a.manifest);features=np.load(a.features,mmap_mode='r')
    metadata=json.loads(Path(a.features).with_suffix('.json').read_text())
    if metadata['sample_ids']!=[r['sample_id'] for r in rows]:p.error('Cache and manifest order differ')
    if features.shape!=(len(rows),8,257,768):p.error('Expected an N x 8 x 257 x 768 feature cache')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(42+a.fold);random.seed(42+a.fold);np.random.seed(42+a.fold)
    model=build_head(a.kind).to(a.device).float();device=torch.device(a.device)
    tr=np.array([i for i,r in enumerate(rows) if r['fold']!=a.fold]);va=np.array([i for i,r in enumerate(rows) if r['fold']==a.fold]);y=np.array([r['label'] for r in rows])
    counts=np.bincount(y[tr],minlength=2)
    if not len(tr) or not len(va) or np.any(counts==0):p.error('Invalid training/validation partition')
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(len(tr)/(2*counts),device=device,dtype=torch.float32),label_smoothing=.1)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01,fused=device.type=='cuda');best=(-1.,-1.)
    for epoch in range(a.epochs+1):
        if epoch:
            model.train();torch.manual_seed(42+a.fold+epoch*10007)
            rng=np.random.default_rng(stable_seed(f'matched-heads-v1:{a.dataset}:{a.fold}:{epoch}'))
            order=rng.permutation(tr);views=rng.integers(0,8,len(order))
            for start in range(0,len(order),32):
                idx=order[start:start+32];view=views[start:start+32]
                x=torch.from_numpy(np.array(features[idx,view],copy=True)).to(device);target=torch.tensor(y[idx],device=device)
                opt.zero_grad(set_to_none=True);loss=criterion(model(x),target)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();opt.step()
        model.eval();probs=[]
        with torch.inference_mode():
            for start in range(0,len(va),32):
                x=torch.from_numpy(np.array(features[va[start:start+32],0],copy=True)).to(device)
                probs.extend(model(x).softmax(-1)[:,1].cpu().tolist())
        values=metrics(y[va],probs);key=values['f1'],values['auc']
        if key[0]>best[0]+1e-12 or abs(key[0]-best[0])<=1e-12 and key[1]>best[1]+1e-12:
            best=key;torch.save({'format':'EAAI_MATCHED_FROZEN_HEAD_V1','kind':a.kind,'fold':a.fold,'dataset':a.dataset,
                'head_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':epoch,'metrics':values},out/'best.pt')
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch,'validation':values})+'\n')
        print(json.dumps({'epoch':epoch,'validation':values}),flush=True)
if __name__=='__main__':main()
