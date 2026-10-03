# IADA-Net

PyTorch implementation of **Image-Adaptive Depth Aggregation with Context-Conditioned Row Pooling for Whole-Image Breast Ultrasound Classification**.

The final model is **URFM-L/16 + IADA**. It accepts a complete B-mode image at 224 × 224, compresses its 14 × 14 patch grid into an ordered row sequence, adds explicit depth encoding, and applies bidirectional depth interaction. Query adaptation trains only the image-conditioned pooling adapter while retaining the selected Base encoder and classifier. Inference requires no lesion crop, segmentation mask, Doppler, or elastography image.

## Installation

Use Python 3.10 or newer. Install matching PyTorch and torchvision builds using the [PyTorch instructions](https://pytorch.org/get-started/locally/), then:

```bash
git clone https://github.com/wujia-xyz/IADA-Net.git
cd IADA-Net
pip install -e '.[test]'
```

## Data and initialization

Obtain images from their providers: [BUSI](https://doi.org/10.1016/j.dib.2019.104863), [UDIAT](https://doi.org/10.1109/JBHI.2017.2731873), [ARC/TDF-Net](https://doi.org/10.1016/j.inffus.2024.102592), [BrEaST](https://doi.org/10.1038/s41597-024-02984-z), and [BUS-BRA](https://doi.org/10.1002/mp.16812). GDPH/SYSUCC auxiliary images are available through the [HoVer-Trans author repository](https://github.com/yuhaomo/HoVerTrans).

Download **URFM ViT-L/16 pretraining**, `mae_vit_large_patch16_dec768d8b_all_biomedclip_1199.pth`, from the [URFM authors](https://github.com/sonovision-ai/URFM) / [official weight repository](https://huggingface.co/QingboKang/URFM). Base training loads `ema_state_dict`. The initialization is the pretraining checkpoint, rather than a downstream task-finetuned model. Images and pretrained/task-trained weights are not bundled.

`data/paired_source_seed42/` contains the final nested source partitions: 1,052 original and 2,301 auxiliary images. Prepare the fixed paired schedules:

```bash
python -m iada.source_protocol --protocol data/paired_source_seed42 --output outputs/source_protocol
```

Create a local `roots.json` with your dataset locations:

```json
{
  "busi": "/data/BUSI",
  "udiat": "/data/UDIAT/original",
  "arc": "/data/ARC/BD3M",
  "auxiliary": "/data/auxiliary"
}
```

The auxiliary root contains `GDPH/` and `SYSUCC/`. [Source preparation](docs/SOURCE_PROTOCOL.md) describes the exact roles and schedules. Preparation and the final training, prediction and scoring commands use structural checks without calculating file hashes or writing audit receipts.

## Train the final model

For each fold, fit Base for 100 epochs and Query for 40:

```bash
python -m iada.train_urfm --config configs/urfm_base.json --prepared outputs/source_protocol --roots roots.json --backbone-weights weights/mae_vit_large_patch16_dec768d8b_all_biomedclip_1199.pth --fold 1 --output outputs/urfm/fold1/base --device cuda

python -m iada.train_urfm --config configs/urfm_query.json --prepared outputs/source_protocol --roots roots.json --base-checkpoint outputs/urfm/fold1/base/best.pt --fold 1 --output outputs/urfm/fold1/query --device cuda
```

Repeat for folds 1–5. Both stages use seed 42, the fixed original/auxiliary schedule, and source-only selection: equal-source macro F1 at 0.5, then macro AUC, then earliest. Query includes epoch zero. Base disables vertical reflection while retaining its random draw; Query uses the original mild augmentation. Outputs are `best.pt`, `history.json`, `selection.csv`, and `RESULT.json`. Use `--resume` for an incomplete run.

This portable entry point implements selected Base → Query. It does not implement the historical passive SWA branch. [Training details](docs/TRAIN_SOURCE.md) specify the optimizer, losses, precision and selection rules.

## Predict and score

The loader supports the final R9 full-model checkpoints and the earlier DINOv2 models. A full task-trained checkpoint includes its encoder and requires no separate pretraining file for inference.

```bash
python -m iada.predict --checkpoint outputs/urfm/fold1/query/best.pt --image path/to/image.png --device cuda

python -m iada.predict_manifest --checkpoint outputs/urfm/fold1/query/best.pt --manifest evaluation_manifest.csv --data-root /data/evaluation --output outputs/predictions/fold1.csv --device cuda --batch-size 32
```

The manifest supplies a unique `sample_id` and `image_path` per image. Patient scoring also requires `patient_id` and explicit `patient_label`. Produce one prediction CSV per fold, then:

```bash
python -m iada.score_patient_series --manifest evaluation_manifest.csv --fold-predictions outputs/predictions/fold1.csv outputs/predictions/fold2.csv outputs/predictions/fold3.csv outputs/predictions/fold4.csv outputs/predictions/fold5.csv --cohort-kind binary --output outputs/evaluation.json
```

The final rule is **mean fold probability per image → maximum image probability per patient**, with `probability >= 0.5` positive. Use `--cohort-kind malignant-only` for the clinical series, which reports sensitivity and TP/FN with an exact interval. [Patient-series documentation](docs/PATIENT_SERIES.md) describes paired comparisons and complete input alignment. The earlier `iada.evaluate` command uses a different historical aggregation rule.

## Saved five-fold results

These are saved R9 results, not experiments rerun during the repository update:

| URFM-L/16 readout | Internal macro AUC | Internal macro F1 | BrEaST AUC | BUS-BRA AUC | External mean AUC | Clinical FN / 120 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **IADA-Net** | 0.9286 | **0.8536** | **0.9065** | 0.9246 | **0.9155** | 15 |
| CLS | **0.9341** | 0.8467 | 0.8999 | **0.9273** | 0.9136 | 16 |
| GAP | 0.9291 | 0.8319 | 0.9019 | 0.9183 | 0.9101 | **14** |
| w/o explicit depth | 0.9168 | 0.8307 | 0.9009 | 0.9127 | 0.9068 | **14** |
| w/o DBI | 0.9258 | 0.8292 | 0.8958 | 0.9243 | 0.9100 | 17 |

Internal values average the three source datasets' five-fold means; per-dataset sample SDs are in the saved tables. External AUCs use the five selected fold models under the patient rule above. External mean AUC gives the two external cohorts equal weight.

Against the nine reproduced methods, final IADA has the highest AUC point estimate on both external cohorts and ties TDF-Net for fewest clinical misses (15/120). Saved paired 95% AUC-difference intervals exclude zero for all nine BUS-BRA comparisons and eight BrEaST comparisons; BrEaST versus CAM-QUS includes zero. These are exploratory comparisons without multiple-comparison correction. The full ablation table retains the actual tradeoffs, including metrics where an ablation is higher.

[Final result files](results/urfm_l16/) contain same-backbone and encoder controls, CQ × vertical-flip results, external/clinical aggregates, paired statistics, and all 18 RTX 5090 efficiency measurements. Private images, patient IDs and clinical per-image predictions are not published. Efficiency uses FP32, TF32 off, 10 warm-ups and 50 synchronized measurements; uncounted FLOP operators are identified per row.

## Earlier implementations

The DINOv2-B/14 implementation and controlled-head study remain available for their archived results. Files directly under `results/` belong to that earlier study; final results are under `results/urfm_l16/`. [Reproduction notes](docs/REPRODUCIBILITY.md) distinguish the protocols. The [DABI-Net predecessor](https://github.com/wujia-xyz/DABI-Net) contains earlier comparison implementations.

## Citation and license

See [CITATION.cff](CITATION.cff) for the software and accompanying manuscript citation. This code uses the MIT license. URFM, DINOv2, timm and datasets retain their own terms; see [third-party notices](docs/THIRD_PARTY.md).