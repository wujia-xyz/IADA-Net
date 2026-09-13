"""Cache shared DINOv2 CLS/patch features, with the recorded fixed-view seeds."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from .backbone import DINOv2Backbone
from .data import Images,read_records

def stable_seed(value):return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4],'big')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--weights',required=True);p.add_argument('--output',required=True)
    p.add_argument('--views',type=int,choices=[1,8],default=8);p.add_argument('--device',default='cuda');a=p.parse_args(argv)
    target=Path(a.output)
    if target.exists():p.error('Output already exists; choose a new cache path')
    target.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    records=read_records(a.manifest);clean=Images(records,a.data_root);aug=Images(records,a.data_root,'adapt')
    model=DINOv2Backbone(a.weights).to(a.device).float().eval()
    cache=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=(len(records),a.views,257,768))
    with torch.inference_mode():
        for view in range(a.views):
            for start in range(0,len(records),32):
                batch=[]
                for i in range(start,min(start+32,len(records))):
                    ds=clean if view==0 else aug
                    if view:ds.transform.set_random_seed(stable_seed(f'eaai-fixed-view-v1:{a.dataset}:{records[i]["sample_id"]}:{view}'))
                    batch.append(ds[i][0])
                z=model.forward_features(torch.stack(batch).to(a.device))
                cache[start:start+len(batch),view]=torch.cat([z['x_norm_clstoken'][:,None],z['x_norm_patchtokens']],1).cpu().numpy()
            cache.flush();print(f'Completed view {view+1}/{a.views}',flush=True)
    target.with_suffix('.json').write_text(json.dumps({'dataset':a.dataset,'shape':list(cache.shape),'sample_ids':[r['sample_id'] for r in records]},indent=2))
if __name__=='__main__':main()
