"""Train the base classifier or a second-stage adapter from an explicit manifest."""
import argparse,json,math,random
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import IADANet
from .data import Images,read_records,worker_seed
from .checkpoints import load_payload,state_dict
from .metrics import metrics

def ranking_loss(logits,labels):
    scores=logits.float()[:,1]-logits.float()[:,0]
    pos,neg=scores[labels==1],scores[labels==0]
    return torch.nn.functional.softplus(neg[None,:]-pos[:,None]).mean() if len(pos) and len(neg) else scores.sum()*0

@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval();ys=[];ps=[]
    for images,y,_ in loader:
        ps.extend(model(images.to(device)).float().softmax(-1)[:,1].cpu().tolist());ys.extend(y.tolist())
    return metrics(ys,ps),ps

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--manifest',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--fold',type=int,choices=range(1,6),required=True);p.add_argument('--output',required=True)
    p.add_argument('--backbone-weights');p.add_argument('--base-checkpoint');p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--epochs',type=int)
    a=p.parse_args(argv);c=json.loads(Path(a.config).read_text());stage=c['stage'];device=torch.device(a.device)
    if stage=='base' and not a.backbone_weights:p.error('Base training requires --backbone-weights')
    if stage=='adapt' and not a.base_checkpoint:p.error('Adaptation requires --base-checkpoint from this same fold')
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):p.error('Use a new empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    seed=int(c.get('seed',42))+(a.fold if stage=='adapt' else 0)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    records=read_records(a.manifest);train=[r for r in records if r['fold']!=a.fold];val=[r for r in records if r['fold']==a.fold]
    if not train or not val:p.error('Manifest has an empty training or validation partition')
    counts=np.bincount([r['label'] for r in train],minlength=2)
    if np.any(counts==0):p.error('Training partition requires both classes')
    model=IADANet(c['variant'],backbone_weights=a.backbone_weights)
    if stage=='adapt':
        payload=load_payload(a.base_checkpoint)
        if payload.get('fold') is not None and payload['fold']!=a.fold:p.error('Base checkpoint fold differs')
        model.load_base_state(state_dict(payload));model.enable_adaptation()
    model=model.to(device)
    loader=DataLoader(Images(train,a.data_root,stage,seed),batch_size=c['batch_size'],shuffle=True,num_workers=a.workers,
                      worker_init_fn=worker_seed,generator=torch.Generator().manual_seed(seed),pin_memory=device.type=='cuda')
    valid=DataLoader(Images(val,a.data_root),batch_size=c['batch_size']*2,num_workers=0)
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(len(train)/(2*counts),device=device,dtype=torch.float32),label_smoothing=c['label_smoothing'])
    parameters=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(parameters,lr=c['lr'],weight_decay=c['weight_decay'],fused=device.type=='cuda')
    epochs=a.epochs if a.epochs is not None else c['epochs'];best=(-1.,-1.)
    info={'config':c,'fold':a.fold,'seed':seed,'train_samples':len(train),'validation_samples':len(val),
          'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in parameters)}
    (out/'run.json').write_text(json.dumps(info,indent=2))
    for epoch in range(0 if stage=='adapt' else 1,epochs+1):
        if epoch:
            warm=c.get('warmup_epochs',0)
            if epoch<=warm:lr=c['lr']*epoch/max(warm,1)
            else:
                step=epoch-1 if stage=='adapt' else epoch-warm-1
                denominator=max(1,epochs-1 if stage=='adapt' else epochs-warm-1)
                lr=c['min_lr']+.5*(c['lr']-c['min_lr'])*(1+math.cos(math.pi*step/denominator))
            for group in opt.param_groups:group['lr']=lr
            model.train()
            for images,y,_ in loader:
                images,y=images.to(device),y.to(device);opt.zero_grad(set_to_none=True)
                amp=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' and c.get('amp')=='bf16' else nullcontext()
                with amp:
                    logits=model(images);loss=criterion(logits.float(),y)+c.get('ranking_weight',0)*ranking_loss(logits,y)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);opt.step()
        values,probs=evaluate(model,valid,device)
        key=(values['f1'],values['auc'] if stage=='adapt' else 0.)
        if key>best:
            best=key
            torch.save({'format':'IADA_NET_V1','variant':c['variant'],'fold':a.fold,'config':c,'epoch':epoch,
                        'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'val_metrics':values},out/'best.pt')
            (out/'selected.json').write_text(json.dumps({'epoch':epoch,'metrics':values,'probabilities':probs},indent=2))
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch,'validation':values})+'\n')
        print(json.dumps({'epoch':epoch,'f1':values['f1'],'auc':values['auc']}),flush=True)

if __name__=='__main__':main()
