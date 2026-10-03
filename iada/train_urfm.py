"""Train URFM-L/16 IADA on prepared public source folds.

This entry uses only fold training images, paired auxiliary images, and the
fold's source selection partition. It does not read external or clinical data.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path, PurePosixPath
import random

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import albumentations as A
import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from torch.nn import functional as F

from .data import image_tensor, read_rgb, transform
from .urfm import URFMIADA, URFMLarge


SEED = 42
SOURCES = ('busi', 'udiat', 'arc')
METRICS = ('auc', 'f1', 'accuracy', 'precision', 'sensitivity', 'specificity')


class NoVerticalFlip(A.VerticalFlip):
    """Keep the vertical-flip random draw, but do not reflect the image."""

    def apply(self, img, **params):
        return img


def source_transform(stage):
    result = transform('base' if stage == 'base' else 'adapt', SEED)
    if stage == 'base':
        indices = [i for i, op in enumerate(result.transforms) if type(op) is A.VerticalFlip]
        if len(indices) != 1 or result.transforms[indices[0]].p != .2:
            raise ValueError('Expected the source base augmentation with one 0.2 vertical flip')
        result.transforms[indices[0]] = NoVerticalFlip(p=.2)
    return result


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def cpu_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


class SourceData:
    def __init__(self, prepared, roots, fold, stage):
        prepared = Path(prepared)
        self.train = csv_rows(prepared / f'fold{fold}' / 'train.csv')
        self.selection = csv_rows(prepared / f'fold{fold}' / 'selection.csv')
        self.auxiliary = csv_rows(prepared / 'auxiliary.csv')
        if not self.train or not self.selection or not self.auxiliary:
            raise ValueError('Prepared source partition is empty')
        if {row.get('dataset') for row in self.train + self.selection} != set(SOURCES):
            raise ValueError('Expected BUSI, UDIAT, and ARC source partitions')
        if {row.get('site') for row in self.auxiliary} != {'GDPH', 'SYSUCC'}:
            raise ValueError('Unexpected auxiliary source')
        for row in self.train + self.selection + self.auxiliary:
            if row.get('label') not in ('0', '1'):
                raise ValueError('Source labels must be 0 or 1')
        self.n = len(self.train)
        self.labels = np.asarray([int(row['label']) for row in self.train], dtype=np.int64)
        self.stepsize = 16

        def resolve(row):
            key = row['root_key']
            if key not in roots:
                raise ValueError(f'Missing source root: {key}')
            relative = row['relative_path']
            pure = PurePosixPath(relative)
            if pure.is_absolute() or '..' in pure.parts or '\\' in relative or ':' in relative:
                raise ValueError('Source image paths must be relative to their named root')
            root = Path(roots[key]).expanduser().resolve()
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f'Source image is missing or outside its root: {relative}')
            return path

        paths = [resolve(row) for row in self.train + self.auxiliary]
        selection_paths = [resolve(row) for row in self.selection]
        schedule = np.load(prepared / f'fold{fold}' / f'{stage}_schedule.npz', allow_pickle=False)
        try:
            self.anchor = schedule['anchor']
            self.auxiliary_index = schedule['auxiliary']
        finally:
            schedule.close()
        rounds = 100 if stage == 'base' else 40
        if (self.anchor.ndim != 2 or self.anchor.shape[0] < rounds or
                self.anchor.shape[1] != 2 * self.n or self.auxiliary_index.shape != self.anchor.shape):
            raise ValueError(f'{stage} schedule needs at least {rounds} epochs of {2 * self.n} observations')
        self.anchor = self.anchor[:rounds]
        self.auxiliary_index = self.auxiliary_index[:rounds]
        if ((self.anchor < 0) | (self.anchor >= self.n)).any():
            raise ValueError('Schedule contains an invalid anchor')
        if ((self.auxiliary_index < -1) | (self.auxiliary_index >= len(self.auxiliary))).any():
            raise ValueError('Schedule contains an invalid auxiliary index')
        auxiliary_labels = np.asarray([int(row['label']) for row in self.auxiliary])
        for anchor, auxiliary in zip(self.anchor, self.auxiliary_index):
            original = auxiliary < 0
            if original.sum() != self.n or not np.array_equal(np.sort(anchor[original]), np.arange(self.n)):
                raise ValueError('Each original training image must appear once per epoch')
            paired = ~original
            if not np.array_equal(self.labels[anchor[paired]], auxiliary_labels[auxiliary[paired]]):
                raise ValueError('Auxiliary views must match anchor diagnosis labels')
        self.augment = source_transform(stage)
        self.cache = [cv2.resize(read_rgb(path), (224, 224), interpolation=cv2.INTER_LINEAR) for path in paths]
        self.valid = torch.stack([image_tensor(path) for path in selection_paths])

    def batch(self, stage, epoch, start):
        anchor = self.anchor[epoch - 1, start:start + self.stepsize]
        auxiliary = self.auxiliary_index[epoch - 1, start:start + self.stepsize]
        index = np.where(auxiliary >= 0, self.n + auxiliary, anchor)
        images = []
        for slot, i in enumerate(index):
            seed = SEED + (0 if stage == 'base' else 100000000) + epoch * 1000003 + start + slot
            self.augment.set_random_seed(seed)
            images.append(self.augment(image=self.cache[int(i)])['image'])
        return torch.stack(images), torch.as_tensor(self.labels[anchor].copy(), dtype=torch.long)


def metric_values(labels, scores):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    if set(labels) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError('Selection requires both classes and finite scores')
    pred, positive = scores >= 0, labels == 1
    return dict(auc=float(roc_auc_score(labels, scores)), f1=float(f1_score(labels, pred, zero_division=0)),
                accuracy=float(accuracy_score(labels, pred)),
                precision=float(precision_score(labels, pred, zero_division=0)),
                sensitivity=float(pred[positive].mean()), specificity=float((~pred[~positive]).mean()))


def selection_metrics(records, scores):
    scores = np.asarray(scores, float)
    if len(records) != len(scores):
        raise ValueError('Selection-score length mismatch')
    parts = {}
    for dataset in SOURCES:
        indices = [i for i, row in enumerate(records) if row['dataset'] == dataset]
        parts[dataset] = metric_values([int(records[i]['label']) for i in indices], scores[indices])
    return dict(macro={key: float(np.mean([part[key] for part in parts.values()])) for key in METRICS},
                by_dataset=parts)


def improves(value, best):
    if best is None:
        return True
    current, previous = value['macro'], best['macro']
    return (current['f1'] > previous['f1'] + 1e-12 or
            (abs(current['f1'] - previous['f1']) <= 1e-12 and current['auc'] > previous['auc'] + 1e-12))


def ranking_loss(logits, labels):
    scores = logits.float()[:, 1] - logits.float()[:, 0]
    positive, negative = scores[labels == 1], scores[labels == 0]
    return (F.softplus(negative[None, :] - positive[:, None]).mean()
            if len(positive) and len(negative) else scores.sum() * 0)


def classification_loss(logits, labels, weight, stage):
    if stage == 'base':
        return F.cross_entropy(logits.float(), labels, weight=weight, label_smoothing=.1)
    return (F.cross_entropy(logits.float(), labels, weight=weight, label_smoothing=0., reduction='none').mean()
            + .1 * ranking_loss(logits, labels))


def learning_rate(stage, epoch):
    if stage == 'query':
        return 1e-5 + .5 * 9e-5 * (1 + math.cos(math.pi * (epoch - 1) / 39))
    return (1e-5 * epoch / 5 if epoch <= 5 else
            1e-6 + .5 * 9e-6 * (1 + math.cos(math.pi * (epoch - 6) / 94)))


@torch.no_grad()
def predict_scores(model, images, device):
    model.eval()
    scores = []
    for start in range(0, len(images), 32):
        logits = model(images[start:start + 32].to(device)).float()
        scores.extend((logits[:, 1].double() - logits[:, 0].double()).cpu().tolist())
    return np.asarray(scores)


def build_model(stage, device, backbone_weights=None, base_checkpoint=None, fold=None):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if stage == 'base':
        model = URFMIADA(URFMLarge(weights=backbone_weights, weights_key='ema_state_dict'), 'fixed')
    else:
        model = URFMIADA(variant='query')
        checkpoint = torch.load(base_checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if checkpoint.get('stage') != 'base' or checkpoint.get('fold') != fold:
            raise ValueError('Query requires a selected URFM base checkpoint from the same fold')
        model.load_base_state(checkpoint['model_state_dict'])
        model.enable_adaptation()
    return model.to(device)


def rng_state():
    state = np.random.get_state()
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python=random.getstate(), numpy=(state[0], torch.as_tensor(state[1].copy()), *state[2:]))


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], numpy[1].numpy(), *numpy[2:]))


def run(args, config, data):
    stage, device = config['stage'], torch.device(args.device)
    rounds = config['epochs']
    model = build_model(stage, device, args.backbone_weights, args.base_checkpoint, args.fold)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5 if stage == 'base' else 1e-4,
                                  weight_decay=.01, fused=device.type == 'cuda')
    counts = np.bincount(data.labels, minlength=2)
    if (counts == 0).any():
        raise ValueError('Training partition requires both classes')
    weight = torch.as_tensor(data.n / (2 * counts), device=device, dtype=torch.float32)
    output = args.output
    history, best, best_scores, selected_epoch, start_epoch = [], None, None, -1, 1
    if args.resume:
        state = torch.load(output / 'resume.pt', map_location='cpu', weights_only=True)
        if state['fold'] != args.fold or state['stage'] != stage or state['epochs'] != rounds:
            raise ValueError('Resume stage, fold, or duration differs')
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        history, best, best_scores = state['history'], state['best'], state['best_scores'].numpy()
        selected_epoch, start_epoch = state['selected_epoch'], state['epoch'] + 1
        restore_rng(state['rng'])
    elif stage == 'query':
        best_scores = predict_scores(model, data.valid, device)
        best = selection_metrics(data.selection, best_scores)
        selected_epoch = 0
        history.append(dict(epoch=0, selection=best, loss=None))
        save_checkpoint(output / 'best.pt', dict(model_state_dict=cpu_state(model), epoch=0,
                                                  selection=best, stage='query', fold=args.fold))

    for epoch in range(start_epoch, rounds + 1):
        rate = learning_rate(stage, epoch)
        for group in optimizer.param_groups:
            group['lr'] = rate
        model.train()
        total, seen = 0., 0
        for start in range(0, 2 * data.n, data.stepsize):
            images, labels = data.batch(stage, epoch, start)
            torch.manual_seed(10000000 + SEED + (0 if stage == 'base' else 200000000) + epoch * 10000 + start)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda' and stage == 'query'):
                logits = model(images)
                loss = classification_loss(logits, labels, weight, stage)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(labels)
            seen += len(labels)
        if seen != 2 * data.n:
            raise RuntimeError('Source observation count changed')
        scores = predict_scores(model, data.valid, device)
        values = selection_metrics(data.selection, scores)
        if improves(values, best):
            best, best_scores, selected_epoch = values, scores, epoch
            save_checkpoint(output / 'best.pt', dict(model_state_dict=cpu_state(model), epoch=epoch,
                                                      selection=best, stage=stage, fold=args.fold))
        history.append(dict(epoch=epoch, selection=values, loss=total / seen, lr=rate))
        write_json(output / 'history.json', history)
        save_checkpoint(output / 'resume.pt', dict(model_state_dict=cpu_state(model), optimizer=optimizer.state_dict(),
                                                    epoch=epoch, fold=args.fold, stage=stage, epochs=rounds,
                                                    history=history, best=best, best_scores=torch.as_tensor(best_scores),
                                                    selected_epoch=selected_epoch, rng=rng_state()))
        print(json.dumps(dict(epoch=epoch, selected_epoch=selected_epoch, macro=values['macro'])), flush=True)

    fields = ('sample_id', 'dataset', 'image_family_id', 'label')
    selected = [{**{key: row[key] for key in fields}, 'score': format(best_scores[i], '.17g')}
                for i, row in enumerate(data.selection)]
    write_csv(output / 'selection.csv', selected)
    write_json(output / 'RESULT.json', dict(status='complete', fold=args.fold, stage=stage,
                                            epochs=rounds, selected_epoch=selected_epoch, selected=best))
    (output / 'resume.pt').unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--roots', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=range(1, 6), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backbone-weights', type=Path)
    parser.add_argument('--base-checkpoint', type=Path)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    stage = config.get('stage')
    expected = dict(stage=stage, epochs=100 if stage == 'base' else 40,
                    encoder='urfm_l16_ema', head='iada', seed=SEED,
                    batch_size=16, optimizer='audited')
    if stage not in ('base', 'query') or config != expected:
        parser.error('Config must match the fixed URFM-IADA source recipe')
    if stage == 'base' and (not args.backbone_weights or args.base_checkpoint):
        parser.error('Base requires --backbone-weights only')
    if stage == 'query' and (not args.base_checkpoint or args.backbone_weights):
        parser.error('Query requires --base-checkpoint only')
    if args.resume:
        if not (args.output / 'resume.pt').is_file() or (args.output / 'RESULT.json').exists():
            parser.error('Resume needs an incomplete run with resume.pt')
    elif args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new empty output directory or --resume')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    roots = json.loads(args.roots.read_text(encoding='utf-8-sig'))
    data = SourceData(args.prepared, roots, args.fold, stage)
    run(args, config, data)


if __name__ == '__main__':
    main()
