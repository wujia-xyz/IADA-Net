# Fixed source protocol

`data/paired_source_seed42` records the public-source image IDs, image hashes, relative locations, known image families, and exact nested partition roles used by the later pooled-source study. It contains 1,052 original images (645 BUSI, 159 UDIAT, and 248 ARC) and 2,301 admitted GDPH/SYSUCC auxiliary images. The files contain metadata only: no images, weights, clinical records, external-cohort records, patient-ID column, or source reader grade values.

These partitions are distinct from the older `data/splits/` manifests. They assign each original image to one outer test fold and give it a fixed training or inner-selection role in each of the other folds. Related image families stay together. The family grouping captures known image relationships; BUSI and UDIAT do not establish complete patient identity. Each CSV's row order is part of the protocol. Historical hashes remain stored as metadata; the final workflow does not calculate or require them. `.gitattributes` preserves CSV line endings across checkouts.

## Obtain the source images

Download images from the providers linked in the main README. GDPH/SYSUCC and the reader workbook are available through the [HoVer-Trans author repository](https://github.com/yuhaomo/HoVerTrans). The source providers retain their data terms. Map the `root_key` and `relative_path` columns to your local downloads:

| Root key | Local root expected by `relative_path` |
| --- | --- |
| `busi` | Folder containing `benign/` and `malignant/` |
| `udiat` | UDIAT `original/` folder |
| `arc` | ARC `BD3M/` folder containing the numbered case folders |
| `auxiliary` | Folder containing `GDPH/` and `SYSUCC/` |

Use the recorded relative paths and image IDs to map the provider downloads. Auxiliary duplicate aliases are listed by released image ID so the historical reader-annotation consensus can be reconstructed locally. Final IADA uses diagnosis labels, without reader-grade supervision.

## Recreate the source schedules

From the repository root:

```bash
python -m iada.source_protocol --protocol data/paired_source_seed42 --output outputs/source_protocol
```

Preparation writes to the specified output directory, replacing its prepared files in place when it already exists. It checks all five fold memberships, within-fold family separation, original-image coverage, and both classes within each dataset/role. It does not fit a model, hash files, create audit receipts, or read clinical/external images.

Each source round presents every original training image once and adds one auxiliary image with the same diagnosis label per original observation. Auxiliary selection alternates GDPH and SYSUCC within each diagnosis class and cycles through a shuffled site/class pool. All methods use the same original/auxiliary image order. The base schedule has 250 rounds so the full registered comparator budgets can use it; IADA consumes its first 100. The query schedule has 40 rounds. All deterministic substreams derive from master seed 42; they are not additional seed repetitions.

The command writes `auxiliary.csv` and `fold1/` through `fold5/`, each with `train.csv`, `selection.csv`, `test.csv`, `base_schedule.npz` and `query_schedule.npz`. For a schedule row, `anchor` indexes that fold's original training table. `auxiliary` indexes the common auxiliary table, with -1 indicating that the original anchor image is used. `is_auxiliary` records the role. Structural checks preserve the original observations and their diagnosis labels.

## Recreate reader targets locally

Install the optional workbook dependency and provide the unchanged official workbook:

```bash
pip install -e '.[source]'
python -m iada.source_protocol --protocol data/paired_source_seed42 --output outputs/graded_source_protocol --reader-workbook "path/to/BIRADS&FOLD.xlsx"
```

Each reader's grade is retained only when all exact-image aliases provide the same valid grade. The two readers are masked independently. Classification images and pathology labels remain unchanged. The recorded workbook produces 2,280 and 2,271 available annotations for the two readers. This optional historical control is not required by final URFM-IADA training.

The correspondence control uses one joint-reader permutation within site, diagnosis class, and availability strata. It preserves both readers' joint grade distribution and changes 1,912 grade pairs. `grade_targets.csv` is generated locally from your workbook; it is not included in the repository. The module implements preparation, not an additional fit or a target-dependent choice.

## Training and evaluation scope

This command prepares the final study's source inputs. Use [iada.train_urfm](TRAIN_SOURCE.md) to consume these nested partitions and paired schedules. The historical `iada.train` and `iada.train_source` commands belong to earlier DINOv2 workflows. The [reader head](READER_GRADES.md) is an optional historical control; the [patient-series scorer](PATIENT_SERIES.md) implements final target aggregation.

In the later study, selected checkpoints are chosen only on the inner selection partition by equal-dataset F1 at 0.5, then equal-dataset AUC, then the earliest candidate within tolerance 1e-12. Outer test folds are reserved for internal evaluation. External and clinical images do not enter source scheduling, reader-grade preparation, or checkpoint selection.
