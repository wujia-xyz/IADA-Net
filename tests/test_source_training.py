from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from torch import nn

from iada import train_source as training
from iada.source_position import BicubicPosition


def test_fixed_rates_and_selection_priorities():
    assert training.learning_rate("base",1) == pytest.approx(2e-6)
    assert training.learning_rate("base",5) == pytest.approx(1e-5)
    assert training.learning_rate("base",6) == pytest.approx(1e-5)
    assert training.learning_rate("base",100) == pytest.approx(1e-6)
    assert training.learning_rate("query",1) == pytest.approx(1e-4)
    assert training.learning_rate("query",40) == pytest.approx(1e-5)
    score = lambda f1,auc: {"macro":{"f1":f1,"auc":auc}}
    best = score(.8,.9)
    assert training.improves(score(.81,.7),best)
    assert training.improves(score(.8,.91),best)
    assert not training.improves(score(.79,.99),best)
    assert not training.improves(score(.8+1e-13,.9+1e-13),best)


def test_positional_transpose_matches_native_first_derivative():
    torch.manual_seed(42)
    x = torch.randn(1,3,9,9,dtype=torch.float64,requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    scale = 4.1/9
    actual = BicubicPosition.apply(x,scale,scale)
    native = torch.nn.functional.interpolate(y,scale_factor=(scale,scale),mode="bicubic",align_corners=False,antialias=False)
    torch.testing.assert_close(actual,native,rtol=0,atol=0)
    cotangent = torch.randn_like(actual)
    left, = torch.autograd.grad(actual,x,cotangent)
    right, = torch.autograd.grad(native,y,cotangent)
    torch.testing.assert_close(left,right,rtol=1e-12,atol=1e-12)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(.3)
        self.classifier = nn.Linear(3,2)

    def forward(self, images):
        return self.classifier(self.dropout(images))


class TinySource:
    def __init__(self, output=None):
        self.n = 6
        self.labels = np.asarray([0,1,0,1,0,1])
        self.paths = [f"synthetic-{i}" for i in range(12)]
        self.valid = torch.linspace(-1,1,18).reshape(6,3)
        self.selection = [dict(sample_id=f"synthetic-{i}",dataset=("busi","udiat","arc")[i//2],image_family_id=f"family-{i}",label=str(i%2)) for i in range(6)]
        self.output = output

    def batch(self, phase, epoch, start):
        assert start == 0
        if self.output is not None and epoch == 11:
            (self.output/"STOP_AFTER_ROUND").touch()
        return self.valid.repeat(2,1),torch.tensor(np.tile(self.labels,2)),np.arange(12),np.arange(12,dtype=np.int64)+epoch


def test_round_boundary_resume_matches_uninterrupted_training(tmp_path,monkeypatch):
    # A tiny synthetic classifier exercises checkpoint, optimizer and dropout
    # restoration; it is not an additional ultrasound experiment.
    def make_model(*args,**kwargs):
        torch.manual_seed(42)
        return TinyModel()
    monkeypatch.setattr(training,"build_model",make_model)
    binding={"binding_sha256":"synthetic-test","code_hashes":{},"owner":{"pid":0,"created":0.}}
    args=SimpleNamespace(stage="base",arm="binary",device="cpu",fold=1,backbone_weights=Path("unused"),base_checkpoint=None,query_label_smoothing=0.)
    full=tmp_path/"full";full.mkdir();args.output=full
    training.train_stage(args,TinySource(),binding,False)
    resumed=tmp_path/"resumed";resumed.mkdir();args.output=resumed
    training.train_stage(args,TinySource(resumed),binding,False)
    assert json.loads((resumed/"STATUS.json").read_text())["status"]=="stopped_on_request"
    assert json.loads((resumed/"RESUME.json").read_text())["epoch"]==11
    (resumed/"STOP_AFTER_ROUND").unlink()
    training.train_stage(args,TinySource(),binding,True)
    for name in ("best.pt","final.pt"):
        a=torch.load(full/name,weights_only=True);b=torch.load(resumed/name,weights_only=True)
        assert a["epoch"]==b["epoch"]
        for key in a["model_state_dict"]:
            torch.testing.assert_close(a["model_state_dict"][key],b["model_state_dict"][key],rtol=0,atol=0)
    assert (full/"selection_all_epochs.csv").read_bytes()==(resumed/"selection_all_epochs.csv").read_bytes()
    assert (full/"exposures.csv").read_bytes()==(resumed/"exposures.csv").read_bytes()
    a=json.loads((full/"RESULT.json").read_text());b=json.loads((resumed/"RESULT.json").read_text())
    assert a["selected"]==b["selected"] and a["updates"]==b["updates"]==100
