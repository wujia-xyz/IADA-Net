import copy
from pathlib import Path

import albumentations as A
import numpy as np
import pytest
import torch
from torch import nn

from iada.data import transform
from iada.source_augmentation import without_vertical_application
from iada import train_source as training


def equal(left, right):
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def test_no_vertical_preserves_draws_and_replays_every_other_operation():
    original = transform("base", 42)
    original.save_applied_params = True
    candidate = without_vertical_application(original)
    assert type(original.transforms[1]) is A.VerticalFlip
    image = np.arange(224*224*3, dtype=np.uint8).reshape(224,224,3)
    seen = {True:0, False:0}
    for position in range(64):
        seed = 42 + 1000003 + position
        original.set_random_seed(seed); candidate.set_random_seed(seed)
        left = original(image=image.copy()); right = candidate(image=image.copy())
        trace = [("VerticalFlip" if name == "NoVerticalFlip" else name, params) for name, params in right["applied_transforms"]]
        assert equal(left["applied_transforms"], trace)
        for a, b in zip([original,*original.transforms], [candidate,*candidate.transforms]):
            assert equal(a.py_random.getstate(), b.py_random.getstate())
            assert equal(a.random_generator.bit_generator.state, b.random_generator.bit_generator.state)
        vertical = any(name == "VerticalFlip" for name, _ in trace)
        operators = {op.__class__.__name__:op for op in original.transforms}
        expected = image.copy()
        for name, params in left["applied_transforms"]:
            if name != "VerticalFlip":
                expected = operators[name].apply(expected, **copy.deepcopy(params))
        assert torch.equal(expected, right["image"])
        if not vertical:
            assert torch.equal(left["image"], right["image"])
        seen[vertical] += 1
    assert all(seen.values())


class TinyIADA(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dinov2 = nn.Identity()
        self.classifier = nn.Linear(2,2)

    def load_base_state(self, state):
        self.load_state_dict(state, strict=True)

    def enable_adaptation(self):
        return self


def stub_models(monkeypatch):
    monkeypatch.setattr(training, "IADANet", TinyIADA)
    monkeypatch.setattr(training, "install", lambda model:None)
    def forbid_reader(model):
        raise AssertionError("A no-vertical model must not receive a reader head")
    monkeypatch.setattr(training, "attach_reader_head", forbid_reader)


def test_no_vertical_is_binary_and_carries_its_own_base_tag(tmp_path, monkeypatch):
    stub_models(monkeypatch)
    model = training.build_model("base", "no_vertical", torch.device("cpu"), backbone_weights=Path("synthetic"))
    path = tmp_path/"base.pt"
    torch.save(dict(model_state_dict=model.state_dict(),phase="base",fold=1,arm="no_vertical"),path)
    query = training.build_model("query","no_vertical",torch.device("cpu"),base_checkpoint=path,fold=1)
    assert not hasattr(model,"reader_ordinal") and not hasattr(query,"reader_ordinal")
    for key,value in model.state_dict().items():
        torch.testing.assert_close(query.state_dict()[key],value,rtol=0,atol=0)


@pytest.mark.parametrize("tag", [None,"binary"])
def test_no_vertical_rejects_unbound_or_other_arm_bases(tmp_path, monkeypatch, tag):
    stub_models(monkeypatch)
    payload = dict(model_state_dict=TinyIADA().state_dict(),phase="base",fold=1)
    if tag is not None:
        payload["arm"] = tag
    path = tmp_path/"base.pt"; torch.save(payload,path)
    with pytest.raises(ValueError):
        training.build_model("query","no_vertical",torch.device("cpu"),base_checkpoint=path,fold=1)


def test_legacy_binary_base_remains_accepted(tmp_path, monkeypatch):
    stub_models(monkeypatch)
    path = tmp_path/"base.pt"
    torch.save(dict(model_state_dict=TinyIADA().state_dict(),phase="base",fold=1),path)
    assert isinstance(training.build_model("query","binary",torch.device("cpu"),base_checkpoint=path,fold=1),TinyIADA)
