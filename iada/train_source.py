"""IADA-only training for the fixed pooled-source and reader-grade protocol."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from datetime import datetime, timezone

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score

from .data import read_rgb, image_tensor, transform
from .model import IADANet
from .reader_grades import attach_reader_head, reader_grade_loss, diagnosis_state
from .source_position import install
from .source_protocol import SEED, SOURCES, file_hash, rows, write_rows
from .source_augmentation import without_vertical_application


ARMS = ("binary", "true_grade", "shuffled_grade", "no_vertical")
METRICS = ("auc", "f1", "accuracy", "precision", "sensitivity", "specificity")


def uses_reader_grades(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown source arm: {arm}")
    return arm in ("true_grade", "shuffled_grade")


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:cpu(v) for k,v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    return value


def save(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(cpu(payload), temporary)
    temporary.replace(path)


def metric_values(labels, scores):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    if set(labels) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Selection requires finite scores and both classes")
    prediction, positive = scores >= 0, labels == 1
    return dict(auc=float(roc_auc_score(labels, scores)), f1=float(f1_score(labels, prediction, zero_division=0)),
                accuracy=float(accuracy_score(labels, prediction)), precision=float(precision_score(labels, prediction, zero_division=0)),
                sensitivity=float(prediction[positive].mean()), specificity=float((~prediction[~positive]).mean()),
                tp=int((positive & prediction).sum()), fn=int((positive & ~prediction).sum()),
                tn=int((~positive & ~prediction).sum()), fp=int((~positive & prediction).sum()))


def selection_metrics(records, scores):
    scores = np.asarray(scores, float)
    if len(records) != len(scores):
        raise ValueError("Selection-score length mismatch")
    parts = {}
    for dataset in SOURCES:
        indices = [i for i, r in enumerate(records) if r["dataset"] == dataset]
        parts[dataset] = metric_values([records[i]["label"] for i in indices], scores[indices])
    return dict(macro={key:float(np.mean([value[key] for value in parts.values()])) for key in METRICS}, by_dataset=parts)


def improves(value, best):
    if best is None:
        return True
    a, b = value["macro"], best["macro"]
    return a["f1"] > b["f1"] + 1e-12 or (abs(a["f1"] - b["f1"]) <= 1e-12 and a["auc"] > b["auc"] + 1e-12)


def ranking_loss(logits, labels):
    scores = logits.float()[:, 1] - logits.float()[:, 0]
    positive, negative = scores[labels == 1], scores[labels == 0]
    return F.softplus(negative[None, :] - positive[:, None]).mean() if len(positive) and len(negative) else scores.sum() * 0


def classification_loss(logits, labels, weight, phase, query_smoothing):
    if phase == "base":
        return F.cross_entropy(logits.float(), labels, weight=weight, label_smoothing=.1)
    return F.cross_entropy(logits.float(), labels, weight=weight, label_smoothing=query_smoothing, reduction="none").mean() + .1 * ranking_loss(logits, labels)


def learning_rate(phase, epoch):
    if phase == "query":
        return 1e-5 + .5 * 9e-5 * (1 + math.cos(math.pi * (epoch - 1) / 39))
    return 1e-5 * epoch / 5 if epoch <= 5 else 1e-6 + .5 * 9e-6 * (1 + math.cos(math.pi * (epoch - 6) / 94))


def make_optimizer(model, phase, graded, device):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if phase == "base" and graded:
        head = model.reader_ordinal
        core = [p for name,p in model.named_parameters() if p.requires_grad and not name.startswith("reader_ordinal.")]
        groups = [dict(params=core, lr=1e-5, weight_decay=.01, role="diagnosis"),
                  dict(params=list(head.severity.parameters()), lr=1e-3, weight_decay=.01, role="ordinal_severity"),
                  dict(params=[head.first_cutpoint,head.raw_gaps], lr=1e-3, weight_decay=0., role="ordinal_cutpoints")]
        optimizer = torch.optim.AdamW(groups, fused=device.type == "cuda")
    else:
        optimizer = torch.optim.AdamW(parameters, lr=1e-4 if phase == "query" else 1e-5, weight_decay=.01, fused=device.type == "cuda")
    return parameters, optimizer


def set_lr(optimizer, phase, epoch, graded):
    value = learning_rate(phase, epoch)
    for i, group in enumerate(optimizer.param_groups):
        group["lr"] = value * (100 if graded and phase == "base" and i > 0 else 1)
    return value


def build_model(phase, arm, device, backbone_weights=None, base_checkpoint=None, fold=None):
    graded = uses_reader_grades(arm)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if phase == "base":
        model = IADANet("fixed", backbone_weights=str(backbone_weights))
        install(model.dinov2)
        model = model.to(device)
        if graded:
            attach_reader_head(model)
    else:
        model = IADANet("query")
        checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("phase") != "base" or (fold is not None and checkpoint.get("fold") != fold):
            raise ValueError("Query adaptation requires a base checkpoint from the same fold")
        state = checkpoint["model_state_dict"]
        has_head = any(key.startswith("reader_ordinal.") for key in state)
        if has_head != graded or checkpoint.get("arm", arm) != arm:
            raise ValueError("Base checkpoint and requested supervision arm differ")
        if arm == "no_vertical" and checkpoint.get("arm") != arm:
            raise ValueError("The no-vertical query requires an explicitly bound no-vertical base checkpoint")
        model.load_base_state(diagnosis_state(state) if has_head else state)
        model.enable_adaptation()
        install(model.dinov2)
        model = model.to(device)
    return model


class SourceData:
    def __init__(self, prepared, roots, fold, arm):
        graded = uses_reader_grades(arm)
        self.arm, self.fold = arm, fold
        self.train = rows(prepared / f"fold{fold}/train.csv")
        self.selection = rows(prepared / f"fold{fold}/selection.csv")
        self.aux = rows(prepared / "auxiliary.csv")
        self.n, self.stepsize = len(self.train), 16
        self.labels = np.asarray([int(r["label"]) for r in self.train], dtype=np.int64)
        self.input_hashes = {}

        def resolve(record):
            root = Path(roots[record["root_key"]]).expanduser().resolve()
            path = (root / record["relative_path"]).resolve()
            if not path.is_relative_to(root) or file_hash(path) != record["image_sha256"]:
                raise ValueError(f"Source image does not match its manifest: {record.get('sample_id', record.get('image_id'))}")
            self.input_hashes[str(path)] = record["image_sha256"]
            return str(path)

        self.paths = [resolve(record) for record in self.train + self.aux]
        valid_paths = [resolve(record) for record in self.selection]
        self.schedules = {phase:dict(np.load(prepared/f"fold{fold}/{phase}_schedule.npz", allow_pickle=False)) for phase in ("base","query")}
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        self.aug = {"base":transform("base",SEED), "query":transform("adapt",SEED)}
        if arm == "no_vertical":
            self.aug["base"] = without_vertical_application(self.aug["base"])
        self.cache = [cv2.resize(read_rgb(path),(224,224),interpolation=cv2.INTER_LINEAR) for path in self.paths]
        self.valid = torch.stack([image_tensor(path) for path in valid_paths])
        self.grades, self.available = None, None
        if graded:
            records = rows(prepared / "grade_targets.csv")
            if [r["image_id"] for r in records] != [r["image_id"] for r in self.aux]:
                raise ValueError("Reader grades must match auxiliary image order")
            prefix = "true" if arm == "true_grade" else "shuffled"
            targets = np.asarray([[int(r[f"{prefix}_grade{k}"]) for k in (1,2)] for r in records], dtype=np.int64)
            available = np.asarray([[int(r[f"reader{k}_available"]) for k in (1,2)] for r in records], dtype=np.uint8)
            self.grades = np.concatenate([np.full((self.n,2),-1,np.int64),targets])
            self.available = np.concatenate([np.zeros((self.n,2),np.uint8),available])

    def batch(self, phase, epoch, start):
        schedule = self.schedules[phase]
        anchor = schedule["anchor"][epoch-1,start:start+self.stepsize]
        auxiliary = schedule["auxiliary"][epoch-1,start:start+self.stepsize]
        index = np.where(auxiliary >= 0, self.n + auxiliary, anchor)
        seeds, values = [], []
        for slot,i in enumerate(index):
            seed = SEED + (0 if phase == "base" else 100000000) + epoch*1000003 + start + slot
            seeds.append(seed); self.aug[phase].set_random_seed(seed)
            values.append(self.aug[phase](image=self.cache[int(i)])["image"])
        return torch.stack(values), torch.tensor(self.labels[anchor]), index, np.asarray(seeds,np.int64)


@torch.no_grad()
def predict(model, images, device):
    model.eval(); scores = []
    for start in range(0,len(images),32):
        logits = model(images[start:start+32].to(device)).float()
        scores.extend((logits[:,1].double()-logits[:,0].double()).cpu().tolist())
    return np.asarray(scores)


def signature(model):
    digest = hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_stage(args, data, binding, resume):
    out, device = args.output, torch.device(args.device)
    phase, graded = args.stage, uses_reader_grades(args.arm)
    model = build_model(phase,args.arm,device,args.backbone_weights,args.base_checkpoint,args.fold)
    initial = signature(model)
    parameters, optimizer = make_optimizer(model,phase,graded,device)
    rounds = 100 if phase == "base" else 40
    counts = np.bincount(data.labels,minlength=2)
    weight = torch.tensor(data.n/(2*counts),device=device,dtype=torch.float32)
    history, scores, best_value, best_state, best_epoch = [], [], None, None, -1
    exposure = np.zeros(len(data.paths),np.int64)
    start_epoch = 1

    def payload(state, epoch, selection=None):
        return dict(model_state_dict=state,epoch=epoch,phase=phase,fold=args.fold,arm=args.arm,
                    selection=selection,protocol_binding=binding["binding_sha256"])

    def snapshot(epoch):
        save(out/"resume.pt",dict(**payload(model.state_dict(),epoch),optimizer=optimizer.state_dict(),
             best_model_state=best_state,best_epoch=best_epoch,best_value=best_value,history=history,
             selection_scores=[torch.as_tensor(v) for v in scores],exposure=torch.as_tensor(exposure)))
        write_json(out/"RESUME.json",dict(epoch=epoch,sha256=file_hash(out/"resume.pt"),binding_sha256=binding["binding_sha256"]))

    if resume:
        receipt = json.loads((out/"RESUME.json").read_text())
        if receipt["binding_sha256"] != binding["binding_sha256"] or receipt["sha256"] != file_hash(out/"resume.pt"):
            raise ValueError("Resume state differs from its bound checkpoint")
        checkpoint = torch.load(out/"resume.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"],strict=True); optimizer.load_state_dict(checkpoint["optimizer"])
        best_state,best_epoch,best_value = checkpoint["best_model_state"],checkpoint["best_epoch"],checkpoint["best_value"]
        history = checkpoint["history"]; scores = [value.numpy() for value in checkpoint["selection_scores"]]
        exposure = checkpoint["exposure"].numpy().copy(); start_epoch = checkpoint["epoch"]+1
        if (out/"history.json").exists() and len(json.loads((out/"history.json").read_text())) > len(history):
            archive = out/f"interrupted_after_round{checkpoint['epoch']:03d}_{int(time.time())}"; archive.mkdir()
            for name in ("history.json","best.pt"):
                if (out/name).exists(): shutil.copy2(out/name,archive/name)
        if best_state is not None: save(out/"best.pt",payload(best_state,best_epoch,best_value))
    else:
        if phase == "query":
            value = predict(model,data.valid,device); best_value = selection_metrics(data.selection,value)
            best_epoch,best_state = 0,cpu(model.state_dict()); scores.append(value)
            history.append(dict(epoch=0,selection=best_value,loss=None))
            save(out/"best.pt",payload(best_state,0,best_value))
        write_json(out/"START.json",dict(utc=utc(),seed=42,phase=phase,arm=args.arm,fold=args.fold,
                   rounds=rounds,batch_size=16,initial_signature=initial,trainable_parameters=sum(p.numel() for p in parameters),
                   binding_sha256=binding["binding_sha256"]))
    for epoch in range(start_epoch,rounds+1):
        if (out/"STOP_AFTER_ROUND").exists():
            snapshot(epoch-1)
            write_json(out/"STATUS.json",dict(status="stopped_on_request",epoch=epoch-1,utc=utc(),owner=binding["owner"]))
            return
        for path,expected in binding["code_hashes"].items():
            if file_hash(path) != expected: raise RuntimeError("Training code changed during the run")
        tick = time.time(); lr = set_lr(optimizer,phase,epoch,graded); model.train()
        total,seen = 0.,0
        hashes = {key:hashlib.sha256() for key in ("index","labels","augmentation_seed","dropout_rng","normalized_pixels")}
        grade_hash = hashlib.sha256(); grade_observations = np.zeros(2,np.int64); grade_batches = 0; grade_total = 0.
        for start in range(0,2*data.n,16):
            images,labels,index,seeds = data.batch(phase,epoch,start); np.add.at(exposure,index,1)
            for key,buffer in (("index",index.astype(np.int32)),("labels",labels.numpy()),("augmentation_seed",seeds),("normalized_pixels",images.numpy())):
                hashes[key].update(buffer.tobytes())
            torch.manual_seed(10000000+SEED+(0 if phase=="base" else 200000000)+epoch*10000+start)
            rng = torch.cuda.get_rng_state(device) if device.type=="cuda" else torch.get_rng_state()
            hashes["dropout_rng"].update(rng.numpy().tobytes())
            images,labels = images.to(device),labels.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda" and phase=="query"):
                logits = model(images); loss = classification_loss(logits,labels,weight,phase,args.query_label_smoothing)
                if phase=="base" and graded:
                    grades,available = data.grades[index],data.available[index]
                    grade_hash.update(grades.tobytes()); grade_hash.update(available.tobytes())
                    grade_observations += available.sum(axis=0).astype(np.int64)
                    grade_batches += int(available.any())
                    auxiliary_loss = reader_grade_loss(model,torch.as_tensor(grades,device=device),torch.as_tensor(available,device=device,dtype=torch.bool))
                    grade_total += float(auxiliary_loss.detach()); loss = loss + .1*auxiliary_loss
            if not bool(torch.isfinite(loss)): raise RuntimeError("Nonfinite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True); optimizer.step()
            total += float(loss.detach())*len(labels); seen += len(labels)
        if seen != 2*data.n: raise RuntimeError("Source observation count changed")
        value = predict(model,data.valid,device); metric = selection_metrics(data.selection,value); scores.append(value)
        if improves(metric,best_value):
            best_value,best_epoch,best_state = metric,epoch,cpu(model.state_dict())
            save(out/"best.pt",payload(best_state,best_epoch,best_value))
        history.append(dict(epoch=epoch,selection=metric,loss=total/seen,lr=lr,observations=seen,updates=math.ceil(seen/16),seconds=time.time()-tick,
                            **{key+"_sha256":digest.hexdigest() for key,digest in hashes.items()},grade_target_mask_sha256=grade_hash.hexdigest(),
                            grade_reader_observations=grade_observations.tolist(),grade_active_batches=grade_batches,mean_auxiliary_loss=grade_total/math.ceil(seen/16)))
        write_json(out/"history.json",history)
        if epoch%10==0 or epoch==rounds: snapshot(epoch)
        write_json(out/"STATUS.json",dict(status="training",phase=phase,epoch=epoch,selected_epoch=best_epoch,utc=utc(),owner=binding["owner"]))
        if epoch==1 or epoch%10==0: print(json.dumps(dict(epoch=epoch,selected_epoch=best_epoch,selection=metric["macro"])),flush=True)
    save(out/"final.pt",payload(model.state_dict(),rounds))
    fields = ("sample_id","dataset","image_family_id","label")
    selected_index = best_epoch if phase=="query" else best_epoch-1
    selected = [{**{k:r[k] for k in fields},"score":format(scores[selected_index][i],".17g")} for i,r in enumerate(data.selection)]
    write_rows(out/"selected_predictions.csv",selected)
    first = 0 if phase=="query" else 1
    all_scores = [{**{k:r[k] for k in fields},**{f"epoch{epoch:03d}":format(value[i],".17g") for epoch,value in enumerate(scores,first)}} for i,r in enumerate(data.selection)]
    write_rows(out/"selection_all_epochs.csv",all_scores)
    if not np.all(exposure[:data.n]==rounds) or int(exposure[data.n:].sum())!=data.n*rounds or not bool((exposure[data.n:]>0).all()):
        raise RuntimeError("Final source exposures differ from the fixed protocol")
    write_rows(out/"exposures.csv",[dict(path=path,exposures=int(count),role="original" if i<data.n else "auxiliary") for i,(path,count) in enumerate(zip(data.paths,exposure))])
    write_json(out/"RESULT.json",dict(status="complete",utc=utc(),seed=42,fold=args.fold,phase=phase,arm=args.arm,
               rounds=rounds,selected_round=best_epoch,selected=best_value,final=history[-1]["selection"],
               updates=rounds*math.ceil(2*data.n/16),observations=int(exposure.sum()),binding_sha256=binding["binding_sha256"],
               artifacts={p.name:file_hash(p) for p in out.iterdir() if p.is_file() and p.name not in ("STATUS.json","RESULT.json")}))
    write_json(out/"STATUS.json",dict(status="complete",utc=utc(),owner=binding["owner"]))


def main(argv=None):
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",type=Path,required=True); parser.add_argument("--roots",type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(1,6),required=True); parser.add_argument("--stage",choices=("base","query"),required=True)
    parser.add_argument("--arm",choices=ARMS,default="binary"); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--backbone-weights",type=Path); parser.add_argument("--base-checkpoint",type=Path)
    parser.add_argument("--query-label-smoothing",type=float,choices=(0.,.1),default=0.)
    parser.add_argument("--device",default="cuda"); parser.add_argument("--resume",action="store_true")
    args = parser.parse_args(argv)
    if args.stage=="base" and (not args.backbone_weights or args.base_checkpoint): parser.error("Base stage needs --backbone-weights only")
    if args.stage=="query" and (not args.base_checkpoint or args.backbone_weights): parser.error("Query stage needs --base-checkpoint only")
    if args.arm!="binary" and args.query_label_smoothing!=0.: parser.error("Reader-grade and no-vertical arms fix hard-label query supervision")
    torch.set_num_threads(4); cv2.setNumThreads(0); torch.set_float32_matmul_precision("highest"); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    preparation = json.loads((args.prepared/"PREPARATION.json").read_text())
    if preparation["master_seed"]!=42 or preparation["status"]!="prepared_no_training": parser.error("Unexpected prepared protocol")
    for relative,expected in preparation["files"].items():
        path=(args.prepared/relative).resolve()
        if not path.is_relative_to(args.prepared.resolve()) or file_hash(path)!=expected: parser.error(f"Prepared input changed: {relative}")
    roots = json.loads(args.roots.read_text(encoding="utf-8-sig"))
    source_files = [Path(__file__), *[Path(__file__).with_name(name) for name in ("source_position.py","source_protocol.py","source_augmentation.py","reader_grades.py","model.py","data.py","backbone.py","layers.py")]]
    config = dict(seed=42,fold=args.fold,stage=args.stage,arm=args.arm,query_label_smoothing=args.query_label_smoothing,device=args.device,
                  preparation_sha256=file_hash(args.prepared/"PREPARATION.json"),roots_sha256=file_hash(args.roots),
                  initialization_sha256=file_hash(args.backbone_weights or args.base_checkpoint),
                  code_hashes={str(path.resolve()):file_hash(path) for path in source_files})
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    binding=dict(**config,binding_sha256=config_hash,owner=dict(pid=os.getpid(),created=psutil.Process().create_time()))
    if args.output.exists():
        if not args.resume: parser.error("Use a new output directory, or explicitly resume its bound checkpoint")
        old=json.loads((args.output/"RUN.json").read_text())
        if old["binding_sha256"]!=config_hash: parser.error("Resume configuration or inputs differ")
        status=json.loads((args.output/"STATUS.json").read_text()); owner=status["owner"]
        try:
            process=psutil.Process(owner["pid"])
            if process.is_running() and abs(process.create_time()-owner["created"])<.01: parser.error("The previous owner is still running")
        except psutil.NoSuchProcess: pass
        if status["status"]=="complete": parser.error("This phase is already complete")
    else:
        if args.resume: parser.error("There is no existing run to resume")
        args.output.mkdir(parents=True)
    write_json(args.output/"RUN.json",binding)
    write_json(args.output/"STATUS.json",dict(status="preparing",utc=utc(),owner=binding["owner"]))
    try:
        data=SourceData(args.prepared,roots,args.fold,args.arm)
        write_json(args.output/"IMAGE_BINDINGS.json",data.input_hashes)
        train_stage(args,data,binding,args.resume)
    except Exception as error:
        write_json(args.output/"STATUS.json",dict(status="failed",utc=utc(),error=repr(error),owner=binding["owner"]))
        raise


if __name__=="__main__":
    main()
