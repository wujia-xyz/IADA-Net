# Fixed source protocol

`data/paired_source_seed42` records the public-source image IDs, image hashes, relative locations, known image families, and exact nested partition roles used by the later pooled-source study. It contains 1,052 original images (645 BUSI, 159 UDIAT, and 248 ARC) and 2,301 admitted GDPH/SYSUCC auxiliary images. The files contain metadata only: no images, weights, clinical records, external-cohort records, patient-ID column, or source reader grade values.

These partitions are distinct from the older `data/splits/` manifests. They assign each original image to one outer test fold and give it a fixed training or inner-selection role in each of the other folds. Related image families stay together. The family grouping captures known image relationships; BUSI and UDIAT do not establish complete patient identity. Each CSV's row order is part of the protocol. `PROTOCOL.json` binds the three CSVs by SHA256, and `.gitattributes` preserves their LF line endings across checkouts.

## Obtain the source images

Download images from the providers linked in the main README. GDPH/SYSUCC and the reader workbook are available through the [HoVer-Trans author repository](https://github.com/yuhaomo/HoVerTrans). The source providers retain their data terms. Map the `root_key` and `relative_path` columns to your local downloads:

| Root key | Local root expected by `relative_path` |
| --- | --- |
| `busi` | Folder containing `benign/` and `malignant/` |
| `udiat` | UDIAT `original/` folder |
| `arc` | ARC `BD3M/` folder containing the numbered case folders |
| `auxiliary` | Folder containing `GDPH/` and `SYSUCC/` |

An image's SHA256 must match `image_sha256` before it is used to replay the study. Auxiliary duplicate aliases are listed by released image ID so the reader-annotation consensus can be reconstructed locally.

## Recreate the source schedules

From the repository root:

```bash
python -m iada.source_protocol --protocol data/paired_source_seed42 --output outputs/source_protocol
```

The output must be a new directory. Preparation validates the published metadata hashes, all five fold memberships, within-fold family separation, original-image coverage, and both classes within each dataset/role. It does not fit a model or read any clinical or external cohort.

Each source round presents every original training image once and adds one auxiliary image with the same diagnosis label per original observation. Auxiliary selection alternates GDPH and SYSUCC within each diagnosis class and cycles through a shuffled site/class pool. All methods use the same original/auxiliary image order. The base schedule has 250 rounds so the full registered comparator budgets can use it; IADA consumes its first 100. The query schedule has 40 rounds. All deterministic substreams derive from master seed 42; they are not additional seed repetitions.

The command writes `fold1/` through `fold5/`, each with `train.csv`, `selection.csv`, `test.csv`, `base_schedule.npz` and `query_schedule.npz`. For a schedule row, `anchor` indexes that fold's original training table. `auxiliary` indexes the common auxiliary table, with -1 indicating that the original anchor image is used. `is_auxiliary` explicitly records the role. Arrays are verified to preserve the original observations and their diagnosis labels. `PREPARATION.json` hashes the generated files.

## Recreate reader targets locally

Install the optional workbook dependency and provide the unchanged official workbook:

```bash
pip install -e '.[source]'
python -m iada.source_protocol --protocol data/paired_source_seed42 --output outputs/graded_source_protocol --reader-workbook "path/to/BIRADS&FOLD.xlsx"
```

The file must match the registered workbook SHA256. Each reader's grade is retained only when all exact-image aliases provide the same valid grade. The two readers are masked independently. Classification images and pathology labels remain unchanged. This produces 2,280 and 2,271 available annotations for the two readers.

The correspondence control uses one joint-reader permutation within site, diagnosis class, and availability strata. It preserves both readers' joint grade distribution and changes 1,912 grade pairs. `grade_targets.csv` is generated locally from your workbook; it is not included in the repository. The module implements preparation, not an additional fit or a target-dependent choice.

## Training and evaluation scope

This command prepares the later study's source inputs. Use the separate [iada.train_source entry point](TRAIN_SOURCE.md) to consume these nested partitions and paired schedules. The historical `iada.train` command remains the older training wrapper. The [reader head](READER_GRADES.md) and [patient-series scorer](PATIENT_SERIES.md) follow the same explicit separation. Do not combine archived results with a newly prepared run or describe preparation alone as a complete experimental reproduction.

In the later study, selected checkpoints are chosen only on the inner selection partition by equal-dataset F1 at 0.5, then equal-dataset AUC, then the earliest candidate within tolerance 1e-12. Outer test folds are reserved for internal evaluation. External and clinical images do not enter source scheduling, reader-grade preparation, or checkpoint selection.
