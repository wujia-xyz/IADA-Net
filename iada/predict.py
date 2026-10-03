"""Whole-image prediction with a full checkpoint or a bound legacy adapter."""
import argparse,json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from .checkpoints import load_model
from .data import image_tensor

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--adapter')
    p.add_argument('--image',required=True);p.add_argument('--device',default='cpu');a=p.parse_args(argv)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_=load_model(a.checkpoint,a.device,a.adapter)
    with torch.inference_mode():
        logits=model(image_tensor(a.image)[None].to(a.device)).float()
        prob=float(torch.sigmoid(logits[0,1].double()-logits[0,0].double()))
    print(json.dumps({'malignancy_probability':prob,'prediction':int(prob>=.5),'label':'malignant' if prob>=.5 else 'benign'}))
if __name__=='__main__':main()
