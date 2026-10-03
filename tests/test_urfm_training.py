import csv
import io

import numpy as np
import pytest
import torch
from PIL import Image

from iada.train_urfm import (SourceData, classification_loss, improves, learning_rate,
                             restore_rng, rng_state, selection_metrics, source_transform)


def test_source_macro_selection_uses_f1_then_auc_and_keeps_earliest():
    records = [dict(dataset=dataset, label=str(label))
               for dataset in ('busi', 'udiat', 'arc') for label in (0, 1)]
    values = selection_metrics(records, [-2, 2, -1, 1, -3, 3])
    assert values['macro']['f1'] == 1.
    assert values['macro']['auc'] == 1.
    assert not improves(values, values)
    better = {'macro': {**values['macro'], 'auc': .9}}
    assert improves(values, better)
    better['macro']['f1'] = 1. + 2e-12
    assert not improves(values, better)


def test_r9_losses_and_endpoints():
    logits = torch.tensor([[0., 2.], [1., -1.], [0., -0.5]])
    labels = torch.tensor([1, 0, 1])
    weight = torch.tensor([1.2, .8])
    base = torch.nn.functional.cross_entropy(logits, labels, weight=weight, label_smoothing=.1)
    torch.testing.assert_close(classification_loss(logits, labels, weight, 'base'), base)
    assert classification_loss(logits, labels, weight, 'query') > 0
    assert learning_rate('base', 5) == pytest.approx(1e-5)
    assert learning_rate('base', 100) == pytest.approx(1e-6)
    assert learning_rate('query', 1) == pytest.approx(1e-4)
    assert learning_rate('query', 40) == pytest.approx(1e-5)


def test_source_data_reads_relative_roots_and_fixed_schedule(tmp_path):
    root, prepared = tmp_path / 'images', tmp_path / 'prepared'
    root.mkdir()
    fold = prepared / 'fold1'
    fold.mkdir(parents=True)
    for name in ('a.png', 'b.png', 'c.png', 'd.png'):
        Image.new('RGB', (8, 8), (80, 100, 120)).save(root / name)

    def write(path, rows):
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    train = [dict(dataset=dataset, label=str(i % 2), root_key='source', relative_path=name)
             for i, (dataset, name) in enumerate(zip(('busi', 'udiat', 'arc'), ('a.png', 'b.png', 'c.png')))]
    selected = [dict(dataset=dataset, label=str(i % 2), root_key='source', relative_path=name)
                for i, (dataset, name) in enumerate(zip(('busi', 'udiat', 'arc'), ('a.png', 'b.png', 'c.png')))]
    auxiliary = [dict(site=site, label=str(i), root_key='source', relative_path='d.png')
                 for i, site in enumerate(('GDPH', 'SYSUCC'))]
    write(fold / 'train.csv', train)
    write(fold / 'selection.csv', selected)
    write(prepared / 'auxiliary.csv', auxiliary)
    anchor = np.tile(np.arange(3, dtype=np.int32), (100, 2))
    aux = np.full_like(anchor, -1)
    aux[:, 3:] = [0, 1, 0]
    np.savez(fold / 'base_schedule.npz', anchor=anchor, auxiliary=aux)
    data = SourceData(prepared, {'source': str(root)}, 1, 'base')
    images, labels = data.batch('base', 1, 0)
    assert images.shape == (6, 3, 224, 224)
    assert labels.tolist() == [0, 1, 0, 0, 1, 0]
    assert any(type(op).__name__ == 'NoVerticalFlip' for op in source_transform('base').transforms)


def test_resume_rng_payload_loads_with_tensor_only_mode():
    stream = io.BytesIO()
    torch.save({'rng': rng_state()}, stream)
    stream.seek(0)
    restore_rng(torch.load(stream, weights_only=True)['rng'])
