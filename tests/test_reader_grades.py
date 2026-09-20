import copy

import pytest
import torch
from torch import nn

from iada.model import IADANet
from iada.reader_grades import (
    ReaderOrdinalHead, attach_reader_head, diagnosis_state,
    ordinal_loss, reader_grade_loss, remove_reader_head,
)


class FeatureBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("features", torch.randn(1, 256, 768))

    def forward_features(self, images):
        return {"x_norm_patchtokens": self.features.expand(len(images), -1, -1)}


def test_cumulative_probabilities_and_missing_reader_mask():
    head = ReaderOrdinalHead().double()
    features = torch.randn(3, 256, dtype=torch.float64, requires_grad=True)
    grades = torch.tensor([[0, 5], [2, -1], [-1, 4]])
    available = grades >= 0
    assert sum(p.numel() for p in head.parameters()) == 266
    torch.testing.assert_close(head.cutpoints(), torch.tensor([[-2., -1., 0., 1., 2.]] * 2).double())
    probabilities = head(features).sigmoid()
    assert bool((probabilities[:, :, 1:] < probabilities[:, :, :-1]).all())
    expected = []
    logits = head(features)
    for sample in range(3):
        for reader in range(2):
            if available[sample, reader]:
                for boundary in range(5):
                    sign = -1 if grades[sample, reader] > boundary else 1
                    expected.append(torch.nn.functional.softplus(sign * logits[sample, reader, boundary]))
    loss = ordinal_loss(head, features, grades, available)
    torch.testing.assert_close(loss, torch.stack(expected).mean())
    altered = grades.clone(); altered[~available] = 999
    torch.testing.assert_close(ordinal_loss(head, features, altered, available), loss)
    with torch.no_grad():
        head.severity.weight.fill_(.01)
        head.first_cutpoint[1].add_(.75)
    assert bool((head.cutpoints()[:, 1:] > head.cutpoints()[:, :-1]).all())
    ordinal_loss(head, features, grades, available).backward()
    assert bool(torch.isfinite(features.grad).all()) and features.grad.abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())


def test_all_missing_batch_does_not_update_head_or_optimizer_clock():
    head = ReaderOrdinalHead(feature_dim=4)
    features = nn.Parameter(torch.randn(2, 4))
    optimizer = torch.optim.AdamW([features, *head.parameters()], lr=.01, weight_decay=.1)
    available = torch.ones(2, 2, dtype=torch.bool)
    grades = torch.tensor([[1, 4], [2, 5]])
    optimizer.zero_grad(set_to_none=True)
    (features.square().mean() + ordinal_loss(head, features, grades, available)).backward()
    optimizer.step()
    before = {name: parameter.detach().clone() for name, parameter in head.named_parameters()}
    clock = {name: optimizer.state[parameter]["step"].clone() for name, parameter in head.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    zero = ordinal_loss(head, features, torch.full((2, 2), -1), torch.zeros(2, 2, dtype=torch.bool))
    assert zero.item() == 0
    (features.square().mean() + zero).backward()
    assert all(parameter.grad is None for parameter in head.parameters())
    optimizer.step()
    for name, parameter in head.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[parameter]["step"], clock[name], rtol=0, atol=0)
    assert optimizer.state[features]["step"].item() == 2


def test_adding_and_removing_head_preserves_diagnosis_and_checkpoint():
    torch.manual_seed(42)
    model = IADANet("fixed", backbone=FeatureBackbone()).eval()
    original = copy.deepcopy(model.state_dict())
    images = torch.zeros(2, 3, 224, 224)
    expected = model(images).detach()
    attach_reader_head(model)
    with pytest.raises(ValueError, match="already attached"):
        attach_reader_head(model)
    torch.testing.assert_close(model(images), expected, rtol=0, atol=0)
    loss = reader_grade_loss(model, torch.tensor([[1, 5], [3, 4]]), torch.ones(2, 2, dtype=torch.bool))
    loss.backward()
    assert model.reader_ordinal.severity.weight.grad.abs().sum() > 0
    plain = diagnosis_state(model.state_dict())
    assert plain.keys() == original.keys()
    for name in original:
        torch.testing.assert_close(plain[name], original[name], rtol=0, atol=0)
    restored = IADANet("fixed", backbone=FeatureBackbone()).eval()
    restored.load_state_dict(plain, strict=True)
    torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
    assert remove_reader_head(model) is model
    torch.testing.assert_close(model(images), expected, rtol=0, atol=0)
    assert not model.classifier._forward_pre_hooks
    with pytest.raises(ValueError, match="Attach a reader head"):
        reader_grade_loss(model, torch.zeros(2, 2, dtype=torch.long), torch.ones(2, 2, dtype=torch.bool))


def test_rejects_invalid_available_grades_and_partial_state():
    head = ReaderOrdinalHead(feature_dim=4)
    features = torch.randn(1, 4)
    for grades in (torch.tensor([[6, 1]]), torch.tensor([[-1, 1]]), torch.tensor([[1.5, 1.]])):
        with pytest.raises(ValueError):
            ordinal_loss(head, features, grades, torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="shape"):
        ordinal_loss(head, features, torch.ones(1), torch.ones(1))
    with pytest.raises(ValueError, match="schema"):
        diagnosis_state({"reader_ordinal.severity.weight": head.severity.weight})
