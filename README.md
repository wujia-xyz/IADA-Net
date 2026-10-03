# IADA-Net

PyTorch implementation of **Image-Adaptive Depth Aggregation with Context-Conditioned Row Pooling for Whole-Image Breast Ultrasound Classification**.

The final model is **URFM-L/16 + IADA**. It accepts a complete B-mode image at 224 × 224, uses context-conditioned row pooling (CCRP) to compress its 14 × 14 grid of 1,024-dimensional patch tokens into 14 ordered row descriptors, adds explicit depth codes, and applies two bidirectional interaction (DBI) layers. The two streams are aligned and fused before averaging over depth and classifying. The final readout uses patch tokens. Query adaptation trains only the image-conditioned pooling adapter while retaining the selected Base encoder and classifier. Inference requires no lesion crop, segmentation mask, Doppler, or elastography image.

## Installation

Use Python 3.10 or newer. Install matching PyTorch and torchvision builds using the [PyTorch instructions](https://pytorch.org/get-started/locally/), then:

```bash
git clone https://github.com/wujia-xyz/IADA-Net.git
cd IADA-Net
pip install -e '.[test]'
```

## Data and initialization

Obtain images from their providers: [BUSI](https://doi.org/10.1016/j.dib.2019.104863), [UDIAT](https://doi.org/10.1109/JBHI.2017.2731873), [ARC/TDF-Net](https://doi.org/10.1016/j.inffus.2024.102592), [BrEaST](https://doi.org/10.1038/s41597-024-02984-z), and [BUS-BRA](https://doi.org/10.1002/mp.16812). GDPH/SYSUCC auxiliary images are available through the [HoVer-Trans author repository](https://github.com/yuhaomo/HoVerTrans).

Download **URFM ViT-L/16 pretraining**, `mae_vit_large_patch16_dec768d8b_all_biomedclip_1199.pth`, from the [URFM authors](https://github.com/sonovision-ai/URFM) / [official weight repository](https://huggingface.co/QingboKang/URFM). Base training loads `ema_state_dict`. The initialization is the pretraining checkpoint, rather than a downstream task-finetuned model. Obtain the encoder checkpoint from its authors. Trained IADA-Net weights and the full study image lists are available from the corresponding authors upon reasonable request, as stated in the manuscript.

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

The auxiliary root contains `GDPH/` and `SYSUCC/`. [Source preparation](docs/SOURCE_PROTOCOL.md) describes the exact roles and schedules. The preparation command preserves the recorded source memberships and schedules.

## Train the final model

For each fold, fit Base for 100 epochs and Query for 40:

```bash
python -m iada.train_urfm --config configs/urfm_base.json --prepared outputs/source_protocol --roots roots.json --backbone-weights weights/mae_vit_large_patch16_dec768d8b_all_biomedclip_1199.pth --fold 1 --output outputs/urfm/fold1/base --device cuda

python -m iada.train_urfm --config configs/urfm_query.json --prepared outputs/source_protocol --roots roots.json --base-checkpoint outputs/urfm/fold1/base/best.pt --fold 1 --output outputs/urfm/fold1/query --device cuda
```

Repeat for folds 1–5. Both stages use seed 42, the fixed original/auxiliary schedule, and source-only selection: equal-source macro F1 at 0.5, then macro AUC, then earliest. Query includes epoch zero. Base disables vertical reflection while retaining its random draw; Query uses the original mild augmentation. Outputs are `best.pt`, `history.json`, `selection.csv`, and `RESULT.json`. Use `--resume` for an incomplete run.

[Training details](docs/TRAIN_SOURCE.md) specify the selected Base → Query workflow, optimizer, losses, precision and checkpoint rule used in the paper.

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

## Comparators and training resources

The paper groups the nine reproduced methods by design and training resources:

| Group | Methods | Training resources |
| --- | --- | --- |
| Generic image classifiers | ResNet-18, ViT-B/16, ConvNeXt-B | Original and auxiliary B-mode images |
| Specialized image-level classifiers | SW-ForkNet, HoVer-Trans, MsGoF | Original and auxiliary B-mode images |
| Lesion-annotation-assisted classifiers | CAM-QUS, MB-DCNN | The same images plus available BUSI/UDIAT lesion masks |
| Multimodal-trained classifier | TDF-Net | ARC B-mode, Doppler and elastography triplets |

All methods receive B-mode input at evaluation. The first eight comparators share IADA-Net's original/auxiliary images, outer folds and checkpoint rule, while retaining their method-specific preprocessing and training recipes. TDF-Net reconstructs missing-modality features and scores each BUSI, UDIAT and ARC image with its corresponding outer-fold model. IADA-Net uses whole images and diagnosis labels throughout training and inference.

## Results reported in the manuscript

The final configuration is URFM-L/16 + IADA, without vertical reflection and with the conditioned query (R9 u16_iada). Development results are image-level five-fold means ± sample SD:

| Dataset | Images | AUC | F1 |
| --- | --- | --- | --- |
| BUSI | 645 | 0.975 ± 0.014 | 0.877 ± 0.030 |
| UDIAT | 159 | 0.910 ± 0.087 | 0.854 ± 0.053 |
| ARC | 248 | 0.901 ± 0.058 | 0.830 ± 0.085 |

External and clinical comparisons (paper Table 3):

| Method | BrEaST AUC | BrEaST F1 | BUS-BRA AUC | BUS-BRA F1 | Clinical missed / 120 |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 | 0.869 | 0.749 | 0.834 | 0.661 | 24 |
| ViT-B/16 | 0.860 | 0.758 | 0.843 | 0.659 | 17 |
| ConvNeXt-B | 0.867 | 0.756 | <ins>0.871</ins> | <ins>0.662</ins> | <ins>16</ins> |
| SW-ForkNet | 0.856 | 0.731 | 0.841 | 0.604 | 20 |
| HoVer-Trans | 0.686 | 0.597 | 0.763 | 0.593 | 23 |
| MsGoF | 0.638 | 0.059 | 0.666 | 0.274 | 82 |
| CAM-QUS† | <ins>0.871</ins> | <ins>0.767</ins> | 0.834 | 0.604 | 18 |
| MB-DCNN† | 0.856 | 0.712 | 0.834 | 0.659 | 26 |
| TDF-Net‡ | 0.850 | 0.698 | 0.800 | 0.614 | **15** |
| IADA-Net (ours) | **0.907** | **0.782** | **0.925** | **0.707** | **15** |

† Uses BUSI/UDIAT lesion masks in training. ‡ Trains on ARC modality triplets and is evaluated with B-mode. IADA-Net has the highest external AUC and F1 point estimates among the nine reproduced comparators, and ties TDF-Net at 15 missed clinical patients (sensitivity 0.875, exact 95% CI 0.802–0.928).

The [six manuscript tables](results/urfm_l16/PAPER_RESULTS.md) include datasets, all development comparisons (including the completed TDF-Net BUSI/UDIAT inference), external/clinical results, component ablations, encoder controls and the vertical-reflection × CQ study. They retain the paper's displayed precision. [development.csv](results/urfm_l16/development.csv) retains the saved unrounded development means and SDs; the other [result files](results/urfm_l16/) contain unrounded controls, paired statistics and all 18 RTX 5090 efficiency measurements. CLS and the A/C extensions are additional saved controls; the final method remains u16_iada.

External and clinical scoring follows **mean fold probability per image → maximum image probability per patient**, with 0.5 positive. Paired AUC intervals use 2,000 class-stratified patient bootstrap resamples; clinical comparisons use exact McNemar tests. Saved AUC-difference intervals exclude zero for nine BUS-BRA and eight BrEaST comparisons; BrEaST versus CAM-QUS includes zero. These comparisons are exploratory and use no multiple-comparison correction.

The [CCRP proposition](docs/THEORY.md) analyzes context before row compression on rectangular feature grids. Its empirical contributions and tradeoffs are reported in the ablation tables. Clinical images, patient identifiers and per-patient predictions are private.

## Earlier implementations

The DINOv2-B/14 implementation and controlled-head study remain available for their archived results. Files directly under `results/` belong to that earlier study; final results are under `results/urfm_l16/`. [Reproduction notes](docs/REPRODUCIBILITY.md) distinguish the protocols. The [DABI-Net predecessor](https://github.com/wujia-xyz/DABI-Net) contains historical implementations; use the URFM commands above for this paper.

## Citation and license

See [CITATION.cff](CITATION.cff) for the software and accompanying manuscript citation. This code uses the MIT license. URFM, DINOv2, timm and datasets retain their own terms; see [third-party notices](docs/THIRD_PARTY.md).