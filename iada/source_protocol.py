"""Recreate the fixed source manifests and schedules without fitting models.

Only public-source metadata is accepted here. The clinical cohort is not an
input to preparation, source splitting, auxiliary sampling, or reader controls.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np


SEED = 42
SOURCES = ("busi", "udiat", "arc")
GRADE_CATEGORIES = ("2", "3", "4A", "4B", "4C", "5")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, records):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(records)


def _unique(records, column):
    values = {row[column]: row for row in records}
    if len(values) != len(records):
        raise ValueError(f"Duplicate {column}")
    return values


def load_protocol(directory):
    directory = Path(directory)
    specification = json.loads((directory / "PROTOCOL.json").read_text(encoding="utf-8"))
    if specification["master_seed"] != SEED:
        raise ValueError("This protocol has one fixed master seed, 42")
    for filename, expected in specification["files"].items():
        path = directory / filename
        if path.resolve().parent != directory.resolve() or not path.is_file():
            raise ValueError(f"Protocol file is missing or outside the metadata directory: {filename}")
    original = rows(directory / "original.csv")
    auxiliary = rows(directory / "auxiliary.csv")
    partitions = rows(directory / "partitions.csv")
    identities = _unique(original, "sample_id")
    _unique(auxiliary, "image_id")
    if len(original) != 1052 or len(auxiliary) != 2301:
        raise ValueError("Unexpected source population")
    for row in original + auxiliary:
        relative = PurePosixPath(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in row["relative_path"] or ":" in row["relative_path"]:
            raise ValueError("Source paths must be relative to their named dataset root")
        if row["label"] not in ("0", "1"):
            raise ValueError("Invalid diagnosis label")
    if {r["dataset"] for r in original} != set(SOURCES):
        raise ValueError("Unexpected original dataset")
    if {r["site"] for r in auxiliary} != {"GDPH", "SYSUCC"}:
        raise ValueError("Unexpected auxiliary source")
    result = {}
    test_occurrences = Counter()
    for fold in range(1, 6):
        groups = {}
        all_ids = []
        for role in ("train", "selection", "test"):
            selected = sorted((r for r in partitions if int(r["fold"]) == fold and r["role"] == role), key=lambda r: int(r["position"]))
            if [int(r["position"]) for r in selected] != list(range(len(selected))):
                raise ValueError("Partition order is missing or duplicated")
            part = [identities[r["sample_id"]] for r in selected]
            if {r["dataset"] for r in part} != set(SOURCES):
                raise ValueError("A source is absent from a partition")
            for dataset in SOURCES:
                if {r["label"] for r in part if r["dataset"] == dataset} != {"0", "1"}:
                    raise ValueError("A source partition needs both diagnosis classes")
            groups[role] = {r["image_family_id"] for r in part}
            all_ids.extend(r["sample_id"] for r in part)
            result[fold, role] = part
            if role == "test":
                test_occurrences.update(r["sample_id"] for r in part)
        if len(all_ids) != len(original) or set(all_ids) != set(identities):
            raise ValueError("A fold omits or duplicates an original image")
        for a, b in (("train", "selection"), ("train", "test"), ("selection", "test")):
            if groups[a] & groups[b]:
                raise ValueError("An image family crosses partition roles")
    if test_occurrences != Counter({identity: 1 for identity in identities}) or len(partitions) != 5 * len(original):
        raise ValueError("Every original image must be tested exactly once")
    return specification, original, auxiliary, result


def paired_schedule(labels, auxiliary, rounds, phase_seed):
    """Exact source order: originals once plus same-class auxiliary views once."""
    labels = np.asarray(labels, dtype=np.int64)
    sites = np.asarray([row["site"] for row in auxiliary])
    aux_labels = np.asarray([int(row["label"]) for row in auxiliary], dtype=np.int64)
    n = len(labels)
    pool_rng = np.random.default_rng(phase_seed)
    pools, position = {}, {}
    site_cursor, recorded = {0: 0, 1: 0}, []
    for site in ("GDPH", "SYSUCC"):
        for label in (0, 1):
            key = (site, label)
            pools[key] = pool_rng.permutation(np.flatnonzero((sites == site) & (aux_labels == label)))
            if not len(pools[key]):
                raise ValueError("Both classes must be present in each auxiliary site")
            position[key] = 0
    for epoch in range(rounds):
        rng = np.random.default_rng(phase_seed + 10000 + epoch)
        order = rng.permutation(2 * n)
        anchor = (order % n).astype(np.int32)
        is_aux = order >= n
        indices = np.full(2 * n, -1, np.int32)
        for j in np.flatnonzero(is_aux):
            label = int(labels[anchor[j]])
            site = ("GDPH", "SYSUCC")[site_cursor[label] % 2]
            site_cursor[label] += 1
            key = (site, label)
            if position[key] == len(pools[key]):
                pools[key] = pool_rng.permutation(pools[key]); position[key] = 0
            indices[j] = int(pools[key][position[key]])
            position[key] += 1
        np.testing.assert_array_equal(np.bincount(anchor[~is_aux], minlength=n), np.ones(n, dtype=int))
        np.testing.assert_array_equal(aux_labels[indices[is_aux]], labels[anchor[is_aux]])
        recorded.append((anchor, indices, is_aux))
    return {key: np.stack([record[i] for record in recorded]) for i, key in enumerate(("anchor", "auxiliary", "is_auxiliary"))}


def reader_targets(workbook, auxiliary, expected_workbook_hash=None):
    """Read the provider's local workbook; never distribute its grade values."""
    import openpyxl
    book = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    try:
        if book.sheetnames != ["prediction"]:
            raise ValueError("Unexpected reader workbook sheets")
        values = list(book["prediction"].values)
    finally:
        book.close()
    if values[0] != ("ID", "BIRADS-reader1", "BIRADS-reader2", "fold") or len(values) != 2406:
        raise ValueError("Unexpected reader workbook schema")
    original = {}
    for row in values[1:]:
        key = str(row[0]).strip()
        if key in original:
            raise ValueError("Duplicate reader image ID")
        original[key] = row[1:3]

    def category(value):
        text = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value).strip().upper()
        return GRADE_CATEGORIES.index(text) if text in GRADE_CATEGORIES else None

    grades = np.full((len(auxiliary), 2), -1, dtype=np.int64)
    available = np.zeros_like(grades)
    for i, row in enumerate(auxiliary):
        aliases = row["alias_released_ids"].split("|")
        for reader in range(2):
            recorded = [category(original[identity][reader]) for identity in aliases]
            if all(value is not None for value in recorded) and len(set(recorded)) == 1:
                grades[i, reader] = recorded[0]; available[i, reader] = 1
    strata = [(row["site"], int(row["label"]), *map(int, available[i])) for i, row in enumerate(auxiliary)]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 2]))
    source_index = np.arange(len(auxiliary))
    for group in sorted(set(strata)):
        indices = np.asarray([i for i, value in enumerate(strata) if value == group])
        permuted = rng.permutation(indices)
        source_index[indices] = permuted
        if Counter(map(tuple, grades[indices])) != Counter(map(tuple, grades[permuted])):
            raise ValueError("Reader joint distribution changed")
        np.testing.assert_array_equal(available[indices], available[permuted])
    return [dict(image_id=row["image_id"], reader1_available=int(available[i, 0]), reader2_available=int(available[i, 1]),
                 true_grade1=int(grades[i, 0]), true_grade2=int(grades[i, 1]),
                 shuffled_grade1=int(grades[source_index[i], 0]), shuffled_grade2=int(grades[source_index[i], 1]),
                 shuffle_source_image_id=auxiliary[source_index[i]]["image_id"]) for i, row in enumerate(auxiliary)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reader-workbook", type=Path)
    args = parser.parse_args(argv)
    specification, original, auxiliary, partitions = load_protocol(args.protocol)
    if args.output.resolve() == args.protocol.resolve():
        parser.error("Prepared outputs must be separate from the published source metadata")
    grades = reader_targets(args.reader_workbook, auxiliary, specification["reader_workbook_sha256"]) if args.reader_workbook else None
    args.output.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 6):
        folder = args.output / f"fold{fold}"; folder.mkdir(exist_ok=True)
        for role in ("train", "selection", "test"):
            write_rows(folder / f"{role}.csv", partitions[fold, role])
        labels = [int(row["label"]) for row in partitions[fold, "train"]]
        for phase, rounds, seed in (("base", 250, SEED), ("query", 40, 100000 + SEED)):
            arrays = paired_schedule(labels, auxiliary, rounds, seed)
            np.savez_compressed(folder / f"{phase}_schedule.npz", **arrays)
    write_rows(args.output / "auxiliary.csv", auxiliary)
    if grades is not None:
        write_rows(args.output / "grade_targets.csv", grades)
    report = dict(status="prepared_no_training", master_seed=SEED, original_images=len(original), auxiliary_images=len(auxiliary),
                  outer_folds=5, grade_targets_created=grades is not None, clinical_or_external_input_used=False)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
