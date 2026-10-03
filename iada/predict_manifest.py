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
from .source_protocol import file_hash, rows


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
    receipt_path = args.output.with_suffix(args.output.suffix+".json")
    if args.output.exists() or receipt_path.exists(): parser.error("Use new prediction and receipt paths")
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
        if record.get("image_sha256") and file_hash(path)!=record["image_sha256"]:
            parser.error(f"Image hash differs: {record['sample_id']}")
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
    temporary=args.output.with_suffix(args.output.suffix+".tmp")
    with temporary.open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["sample_id","score","probability"],lineterminator="\n")
        writer.writeheader()
        for record,score,probability in zip(records,scores,probabilities):
            writer.writerow(dict(sample_id=record["sample_id"],score=format(score,".17g"),probability=format(probability,".17g")))
    temporary.replace(args.output)
    receipt=dict(status="complete",images=len(records),checkpoint_sha256=file_hash(args.checkpoint),
                 manifest_sha256=file_hash(args.manifest),prediction_sha256=file_hash(args.output),
                 probability="scipy expit of the float64 difference between FP32 class logits",
                 manifest_order_preserved=True,diagnosis_labels_used=False,patient_aggregation_performed=False,
                 device=args.device,roots_sha256=file_hash(args.roots) if args.roots else None)
    receipt_path.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":main()
