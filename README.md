# IADA-Net

PyTorch code for **Image-Adaptive Depth Aggregation for Whole-Image Breast Ultrasound Classification** (Jia Wu and Dongjing Shan).

IADA-Net combines a DINOv2 ViT-B/14 encoder, row-attention pooling, depth encoding, and bidirectional attention. A second training stage adapts the pooling query to each image while retaining the base classifier. Inputs are complete B-mode frames and image-level benign/malignant labels.

## Installation

Use Python 3.10 or newer. Install matching PyTorch and torchvision builds for your CPU/CUDA environment using the [PyTorch instructions](https://pytorch.org/get-started/locally/), then run:

```bash
git clone https://github.com/wujia-xyz/IADA-Net.git
cd IADA-Net
pip install -e '.[test]'
pytest -q
```

The tests use synthetic features and do not download models or patient images.

## Data and pretrained initialization

Download datasets from their providers: [BUSI](https://doi.org/10.1016/j.dib.2019.104863), [UDIAT](https://doi.org/10.1109/JBHI.2017.2731873), [ARC/TDF-Net](https://doi.org/10.1016/j.inffus.2024.102592), [BrEaST](https://doi.org/10.1038/s41597-024-02984-z), and [BUS-BRA](https://doi.org/10.1002/mp.16812). Images and pretrained/trained weights are not bundled.

The retained internal fold assignments are in `data/splits/`. Supply `--data-root` as the BUSI directory containing `benign/` and `malignant/`, the UDIAT directory containing `original/`, or the ARC B-mode directory containing numbered patient folders. Each CSV includes the original sample identifier and image hash. See [data preparation](docs/DATA.md).

For base training, obtain the official [DINOv2 ViT-B/14 backbone weights](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth). The model retains DINOv2's original 37 x 37 positional table and interpolation rule, allowing existing full checkpoints to load directly.

## Train IADA-Net

Train the base classifier for each fold, then adapt its query using the same fold:

```bash
python -m iada.train --config configs/base.json --manifest data/splits/busi.csv --data-root data/raw/BUSI --backbone-weights weights/dinov2_vitb14_pretrain.pth --fold 1 --output outputs/busi/fold1/base --device cuda

python -m iada.train --config configs/query.json --manifest data/splits/busi.csv --data-root data/raw/BUSI --base-checkpoint outputs/busi/fold1/base/best.pt --fold 1 --output outputs/busi/fold1/query --device cuda
```

Repeat for folds 1–5 and for the UDIAT/ARC manifests. `configs/readout.json` and `configs/both.json` define the adaptation controls. Base checkpoints are selected by validation F1; adaptation includes the initial model and uses AUC to resolve F1 ties. All metrics for a fold come from one selected checkpoint.

The training configurations use one fixed seed, **42**. The five folds are data partitions, not five random-seed repetitions.

## Predict and evaluate

```bash
python -m iada.predict --checkpoint outputs/busi/fold1/query/best.pt --image path/to/image.png --device cpu

python -m iada.evaluate --checkpoints outputs/busi/fold1/query/best.pt outputs/busi/fold2/query/best.pt outputs/busi/fold3/query/best.pt outputs/busi/fold4/query/best.pt outputs/busi/fold5/query/best.pt --manifest data/breast.csv --dataset breast --data-root data/raw/BrEaST --output outputs/breast_metrics.json --device cuda
```

The historical `iada.evaluate` command averages fold probabilities per image and then images per patient. It assigns exact 0.5 ties to benign and retains the original dataset-specific bootstrap convention. These settings belong to the archived release results.

For a retained compact adapter, pass the original base checkpoint and `--adapter path/to/adapter.pt` to prediction. Evaluation accepts one `--adapters` entry per base checkpoint. The loader checks the adapter's base-checkpoint hash before applying its two tensors. A full model checkpoint already contains its encoder and requires no separate initialization file for inference.

### Patient-series scoring

The newer patient-series protocol averages the five fold probabilities **for each image**, then takes the **maximum image probability per patient**. A probability of at least 0.5 is positive. Score saved fold probabilities with:

```bash
python -m iada.score_patient_series --manifest patient_manifest.csv --fold-predictions fold1.csv fold2.csv fold3.csv fold4.csv fold5.csv --cohort-kind malignant-only --output outputs/series_metrics.json
```

The manifest requires `sample_id`, `patient_id`, and `patient_label`. Each prediction file requires `sample_id` and `probability`; files are joined by ID and must cover the complete manifest. Patient labels do not assert pathology for every image. A malignant-only series reports TP, FN, sensitivity and an exact 95% interval. For a mixed-class public cohort, use `--cohort-kind binary`; its patient bootstrap defaults to 2,000 valid resamples and seed 42. See [patient-series evaluation](docs/PATIENT_SERIES.md) for paired comparisons and optional patient-level output.

This command scores existing predictions. It does not train a model, select a checkpoint, alter its probabilities or make the historical and newer protocols interchangeable.

## Controlled aggregation study

```bash
python -m iada.cache_features --manifest data/splits/busi.csv --data-root data/raw/BUSI --dataset busi --weights weights/dinov2_vitb14_pretrain.pth --output cache/busi.npy --views 8 --device cuda

python -m iada.train_matched --features cache/busi.npy --manifest data/splits/busi.csv --dataset busi --kind depth --fold 1 --output outputs/matched/depth/busi/fold1 --device cuda
```

Run all four head kinds (`gap`, `gated`, `cls_mean`, `depth`) on all five folds. They use the same frozen encoder features, view draws, and approximately 6.3 million trainable parameters. Cache external images with `--views 1`, then use `python -m iada.evaluate_heads --help` for patient-level evaluation and feature-order probes. Query-content control functions are in `iada/probes.py`.

## Results and reproduction

`results/` contains the retained aggregate results underlying the main external evaluation and controlled aggregation study. [Reproduction notes](docs/REPRODUCIBILITY.md) describe the original checkpoints, protocol distinctions, and the relationship between the release and historical runs. The repository is a code release; complete training from new initialization is required when the original task-trained checkpoints are not available.

The [aggregation analysis](docs/THEORY.md) states the encoded-feature scope of the row-compression separation and contextual derivative. It is separate from empirical performance claims. New experimental results are not supplied by the patient-series scoring utility.

The optional [source-reader grade head](docs/READER_GRADES.md) adds training-only ordinal supervision and can be removed without changing diagnosis outputs. It is supplied as a separate component for the ongoing source-supervision study; it does not alter the historical runner or provide that study's uncompleted results.

The later study's [fixed source protocol](docs/SOURCE_PROTOCOL.md) supplies public-source metadata and a command that recreates the exact nested roles, paired image schedules, and optional reader correspondence control. The metadata and preparation command are separate from the older `data/splits/` training workflow. Images and reader workbook values remain with their original providers.

For the later training workflow, use [iada.train_source](docs/TRAIN_SOURCE.md). It trains one fixed-seed fold/stage, supports the reader controls and explicit resume, and keeps the historical runner intact. `iada.predict_manifest` writes full per-image probabilities for the patient-series scorer; it also accepts graded base checkpoints by removing their training-only reader head.

The earlier [DABI-Net repository](https://github.com/wujia-xyz/DABI-Net) provides the predecessor model and comparison implementations. This repository contains the current IADA-Net model and controlled heads, with portable dataset and checkpoint paths.

## Citation and license

Use `CITATION.cff` for this software. The accompanying paper is a manuscript; a publication DOI will be added when available. This code is distributed under the MIT license. DINOv2, timm, and the datasets retain their respective licenses; see [third-party notices](docs/THIRD_PARTY.md).
