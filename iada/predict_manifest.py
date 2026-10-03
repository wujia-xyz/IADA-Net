"""Save complete per-image probabilities for the patient-series scorer."""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from scipy.special import expit

from .checkpoints import load_model
from .data import image_tensor
from .source_protocol import rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,default=Path("."))
    parser.add_argument("--roots",type=Path,help="Named roots for a prepared source manifest")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--batch-size",type=int,default=32)
    args = parser.parse_args(argv)
    if args.batch_size<1: parser.error("Batch size must be positive")
    if args.output.resolve() in {args.manifest.resolve(), args.checkpoint.resolve()}:
        parser.error("Predictions must not overwrite the manifest or checkpoint")
    records = rows(args.manifest)
    if not records or any(not record.get("sample_id") for record in records): parser.error("Every image needs a sample_id")
    if len({record["sample_id"] for record in records})!=len(records): parser.error("sample_id must be unique")
    roots = json.loads(args.roots.read_text(encoding="utf-8-sig")) if args.roots else None
    paths=[]
    for record in records:
        if "image_path" in record:
            path=Path(record["image_path"])
            if not path.is_absolute(): path=args.data_root/path
            path=path.resolve()
        elif roots and "root_key" in record and "relative_path" in record:
            root=Path(roots[record["root_key"]]).expanduser().resolve()
            path=(root/record["relative_path"]).resolve()
            if not path.is_relative_to(root): parser.error("An image path leaves its named dataset root")
        else: parser.error("Provide image_path, or root_key/relative_path with --roots")
        if not path.is_file(): parser.error(f"Image is missing: {record['sample_id']}")
        paths.append(path)
    torch.set_num_threads(4);torch.set_float32_matmul_precision("highest");torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    model,_=load_model(args.checkpoint,args.device)
    scores=[]
    with torch.inference_mode():
        for start in range(0,len(paths),args.batch_size):
            images=torch.stack([image_tensor(path) for path in paths[start:start+args.batch_size]]).to(args.device)
            logits=model(images).float()
            if tuple(logits.shape)!=(len(images),2) or not bool(torch.isfinite(logits).all()):
                raise ValueError("The checkpoint did not produce finite two-class logits")
            scores.extend((logits[:,1].double()-logits[:,0].double()).cpu().tolist())
    probabilities=expit(np.asarray(scores,np.float64))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["sample_id","score","probability"],lineterminator="\n")
        writer.writeheader()
        for record,score,probability in zip(records,scores,probabilities):
            writer.writerow(dict(sample_id=record["sample_id"],score=format(score,".17g"),probability=format(probability,".17g")))
    summary=dict(status="complete",images=len(records),
                 probability="scipy expit of the float64 difference between FP32 class logits",
                 manifest_order_preserved=True,diagnosis_labels_used=False,patient_aggregation_performed=False,
                 device=args.device)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
