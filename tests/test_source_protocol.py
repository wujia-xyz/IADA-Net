from pathlib import Path
import numpy as np
import pytest

from iada.source_protocol import load_protocol, paired_schedule


def test_published_roles_keep_families_and_cover_every_outer_test():
    folder = Path(__file__).resolve().parents[1] / "data/paired_source_seed42"
    protocol, original, auxiliary, partitions = load_protocol(folder)
    assert protocol["master_seed"] == 42 and len(original) == 1052 and len(auxiliary) == 2301
    assert len(partitions) == 15
    assert sum(len(partitions[fold, "test"]) for fold in range(1, 6)) == 1052
    assert {key: sum(r["dataset"] == key for r in original) for key in ("busi", "udiat", "arc")} == {"busi": 645, "udiat": 159, "arc": 248}


def test_schedule_preserves_every_original_and_auxiliary_label():
    auxiliary = [{"site": site, "label": label} for site in ("GDPH", "SYSUCC") for label in (0, 1) for _ in range(3)]
    labels = np.asarray([0, 0, 1, 1])
    schedule = paired_schedule(labels, auxiliary, 10, 42)
    selected = []
    for anchor, aux, flag in zip(schedule["anchor"], schedule["auxiliary"], schedule["is_auxiliary"]):
        assert sorted(anchor[~flag]) == list(range(len(labels)))
        assert int(flag.sum()) == len(labels) and bool((aux[~flag] == -1).all())
        for a, b in zip(anchor[flag], aux[flag]):
            assert auxiliary[b]["label"] == labels[a]
        selected.extend(aux[flag])
    assert set(selected) == set(range(len(auxiliary)))
    with pytest.raises(ValueError, match="Both classes"):
        paired_schedule(labels, [r for r in auxiliary if r["site"] == "GDPH"], 2, 42)
