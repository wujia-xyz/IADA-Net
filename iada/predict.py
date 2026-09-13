"""Whole-image prediction with a full checkpoint or a bound legacy adapter."""
import argparse,json
import torch
from .checkpoints import load_model
from .data import image_tensor

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--adapter')
    p.add_argument('--image',required=True);p.add_argument('--device',default='cpu');a=p.parse_args(argv)
    torch.set_num_threads(4);model,_=load_model(a.checkpoint,a.device,a.adapter)
    with torch.inference_mode():prob=float(model(image_tensor(a.image)[None].to(a.device)).softmax(-1)[0,1])
    print(json.dumps({'malignancy_probability':prob,'prediction':int(prob>.5),'label':'malignant' if prob>.5 else 'benign'}))
if __name__=='__main__':main()
