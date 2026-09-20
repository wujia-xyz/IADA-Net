"""Training-only ordinal reader supervision; no grade is needed at inference.

The cumulative-link objective is established ordinal-regression machinery,
separate from IADA's row-aggregation analysis. See docs/READER_GRADES.md.
"""
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


GRADE_CATEGORIES = ("2", "3", "4A", "4B", "4C", "5")
PREFIX = "reader_ordinal."
HEAD_STATE_KEYS = frozenset(PREFIX + key for key in (
    "severity.weight", "first_cutpoint", "raw_gaps"))


class ReaderOrdinalHead(nn.Module):
    """One image severity and ordered, reader-specific category cutpoints.

    With the study defaults, the head has 266 parameters. Its outputs have shape
    [batch, reader, category_boundary] and represent logits of grade > boundary.
    """

    def __init__(self, feature_dim=256, readers=2, categories=6):
        super().__init__()
        if feature_dim < 1 or readers < 1 or categories < 2:
            raise ValueError("Need positive feature/readers counts and at least two categories")
        self.readers = readers
        self.categories = categories
        self.severity = nn.Linear(feature_dim, 1, bias=False)
        nn.init.zeros_(self.severity.weight)
        self.first_cutpoint = nn.Parameter(torch.full((readers,), -2.))
        self.raw_gaps = nn.Parameter(torch.full((readers, categories - 2), math.log(math.expm1(1.))))

    def cutpoints(self):
        gaps = F.softplus(self.raw_gaps)
        cumulative = torch.cat([gaps.new_zeros(self.readers, 1), gaps.cumsum(dim=1)], dim=1)
        return self.first_cutpoint[:, None] + cumulative

    def forward(self, features):
        return self.severity(features)[:, None, :] - self.cutpoints()[None, :, :]


def ordinal_loss(head, features, grade_index, available):
    """Mean cumulative BCE across thresholds and available reader annotations.

    Indices 0..5 correspond to GRADE_CATEGORIES. Unavailable entries may use -1.
    If the batch has no available grades, the zero remains connected to the
    diagnosis features but not to any head parameter. Call optimizer.zero_grad
    with set_to_none=True so AdamW does not advance an unused auxiliary group.
    """
    if grade_index.shape != available.shape or tuple(grade_index.shape) != (len(features), head.readers):
        raise ValueError("Grades and availability must have shape [batch, readers]")
    if grade_index.device != features.device or available.device != features.device:
        raise ValueError("Features, grades and availability must be on the same device")
    available = available.bool()
    if not bool(available.any()):
        return features.sum() * 0
    valid = grade_index[available]
    if not bool(torch.isfinite(valid).all() and ((valid >= 0) & (valid < head.categories)).all()):
        raise ValueError("Available grade indices are outside the category range")
    if valid.is_floating_point() and not torch.equal(valid, valid.round()):
        raise ValueError("Available grade indices must be integers")
    logits = head(features)
    levels = torch.arange(head.categories - 1, device=features.device)
    targets = (grade_index[:, :, None] > levels).to(logits.dtype)
    per_reader = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=-1)
    return per_reader[available].mean()


def attach_reader_head(model):
    """Attach the head to the existing classifier input without changing logits.

    Attach before constructing the optimizer. The normal model state names are
    unchanged; only reader_ordinal.* is added. The pre-hook captures the latest
    classifier input for reader_grade_loss, including its gradient connection.
    """
    if hasattr(model, "reader_ordinal") or hasattr(model, "_reader_hook_handle"):
        raise ValueError("A reader head is already attached")
    if not isinstance(getattr(model, "classifier", None), nn.Linear):
        raise ValueError("Expected a linear model.classifier")
    parameter = next(model.parameters())
    model.reader_ordinal = ReaderOrdinalHead(model.classifier.in_features).to(parameter.device)

    def capture(module, args):
        model._reader_pooled_features = args[0]

    model._reader_hook_handle = model.classifier.register_forward_pre_hook(capture)
    return model


def reader_grade_loss(model, grade_index, available):
    """Evaluate the auxiliary loss after the corresponding model forward call."""
    if not hasattr(model, "reader_ordinal") or not hasattr(model, "_reader_pooled_features"):
        raise ValueError("Attach a reader head and run the model before computing its loss")
    return ordinal_loss(model.reader_ordinal, model._reader_pooled_features, grade_index, available)


def diagnosis_state(state: Mapping):
    """Remove exactly the known head tensors before loading a diagnosis model."""
    auxiliary = {key for key in state if key.startswith(PREFIX)}
    if auxiliary != HEAD_STATE_KEYS:
        raise ValueError(f"Reader-head state schema mismatch: {sorted(auxiliary)}")
    return {key: value for key, value in state.items() if key not in auxiliary}


def remove_reader_head(model):
    """Remove the training head and its hook without altering diagnosis tensors."""
    if not hasattr(model, "reader_ordinal") or not hasattr(model, "_reader_hook_handle"):
        raise ValueError("No reader head attached by attach_reader_head")
    model._reader_hook_handle.remove()
    del model._reader_hook_handle
    del model.reader_ordinal
    if hasattr(model, "_reader_pooled_features"):
        del model._reader_pooled_features
    return model
