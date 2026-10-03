# Results reported in the manuscript

*Image-Adaptive Depth Aggregation with Context-Conditioned Row Pooling for Whole-Image Breast Ultrasound Classification*.

The tables below retain the manuscript's displayed values and method groups. Bold marks the best displayed value; underlining marks the second best where used in the paper. Values are taken from the current manuscript and saved aggregate results.

## Table 1. Datasets

| Dataset | Role | Unit | Total | Benign | Malignant |
| --- | --- | --- | --- | --- | --- |
| BUSI | Development | Images | 645 | 436 | 209 |
| UDIAT | Development | Images | 159 | 107 | 52 |
| ARC | Development | Cases | 248 | 145 | 103 |
| GDPH&SYSUCC | Auxiliary | Images | 2,301 | 851 | 1,450 |
| BrEaST | External | Patients | 252 | 154 | 98 |
| BUS-BRA | External | Patients | 1,064¹ | 722 | 342 |
| Clinical series | Clinical | Patients | 120² | 0 | 120 |

¹ BUS-BRA includes 1,875 images. ² The clinical series includes 640 views. The three development datasets contain 1,052 images, grouped into 934 image families; GDPH/SYSUCC supplies 2,301 auxiliary training images.

## Table 2. Development comparisons

| Group | Method | BUSI AUC | BUSI F1 | UDIAT AUC | UDIAT F1 | ARC AUC | ARC F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Generic image classifiers | ResNet-18 | 0.956 ± 0.022 | <ins>0.827</ins> ± 0.052 | 0.874 ± 0.044 | 0.735 ± 0.091 | <ins>0.895</ins> ± 0.072 | <ins>0.825</ins> ± 0.068 |
| Generic image classifiers | ViT-B/16 | 0.945 ± 0.012 | 0.808 ± 0.035 | 0.882 ± 0.040 | 0.773 ± 0.103 | 0.882 ± 0.055 | 0.754 ± 0.094 |
| Generic image classifiers | ConvNeXt-B | 0.946 ± 0.034 | 0.799 ± 0.081 | 0.897 ± 0.049 | 0.735 ± 0.091 | 0.878 ± 0.091 | 0.741 ± 0.093 |
| Specialized image-level classifiers | SW-ForkNet | <ins>0.958</ins> ± 0.013 | 0.818 ± 0.035 | 0.849 ± 0.064 | 0.745 ± 0.071 | 0.878 ± 0.064 | 0.762 ± 0.092 |
| Specialized image-level classifiers | HoVer-Trans | 0.811 ± 0.073 | 0.658 ± 0.074 | 0.700 ± 0.096 | 0.548 ± 0.106 | 0.777 ± 0.072 | 0.649 ± 0.067 |
| Specialized image-level classifiers | MsGoF | 0.673 ± 0.081 | 0.289 ± 0.261 | 0.521 ± 0.124 | 0.175 ± 0.262 | 0.580 ± 0.047 | 0.184 ± 0.215 |
| Lesion-annotation-assisted classifiers | CAM-QUS† | 0.937 ± 0.036 | 0.814 ± 0.041 | **0.922** ± 0.040 | <ins>0.778</ins> ± 0.056 | 0.870 ± 0.081 | 0.788 ± 0.062 |
| Lesion-annotation-assisted classifiers | MB-DCNN† | 0.930 ± 0.016 | 0.795 ± 0.027 | 0.884 ± 0.058 | 0.763 ± 0.091 | 0.877 ± 0.050 | 0.802 ± 0.053 |
| Multimodal-trained classifier (B-mode inference) | TDF-Net‡ | 0.850 ± 0.027 | 0.650 ± 0.045 | 0.825 ± 0.112 | 0.609 ± 0.085 | 0.839 ± 0.064 | 0.674 ± 0.114 |
| Proposed image-level method | IADA-Net (ours) | **0.975** ± 0.014 | **0.877** ± 0.030 | <ins>0.910</ins> ± 0.087 | **0.854** ± 0.053 | **0.901** ± 0.058 | **0.830** ± 0.085 |

Image-level AUC and F1 at threshold 0.5, mean ± sample SD over five outer folds. † CAM-QUS and MB-DCNN use available BUSI/UDIAT lesion masks during training. ‡ TDF-Net trains on ARC B-mode/Doppler/elastography triplets and uses B-mode input at evaluation. BUSI and UDIAT are scored with the corresponding outer-fold ARC-trained TDF model.

## Table 3. External and clinical comparisons

| Group | Method | BrEaST AUC | BrEaST F1 | BrEaST SEN | BrEaST SPE | BUS-BRA AUC | BUS-BRA F1 | BUS-BRA SEN | BUS-BRA SPE | Clinical missed / 120 | Clinical SEN (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Generic image classifiers | ResNet-18 | 0.869 | 0.749 | 0.806 | 0.779 | 0.834 | 0.661 | 0.830 | 0.677 | 24 | 0.800 (0.717–0.867) |
| Generic image classifiers | ViT-B/16 | 0.860 | 0.758 | 0.816 | 0.786 | 0.843 | 0.659 | 0.848 | 0.657 | 17 | 0.858 (0.783–0.915) |
| Generic image classifiers | ConvNeXt-B | 0.867 | 0.756 | <ins>0.867</ins> | 0.727 | <ins>0.871</ins> | <ins>0.662</ins> | <ins>0.906</ins> | 0.605 | <ins>16</ins> | 0.867 (0.793–0.922) |
| Specialized image-level classifiers | SW-ForkNet | 0.856 | 0.731 | 0.816 | 0.734 | 0.841 | 0.604 | 0.892 | 0.497 | 20 | 0.833 (0.754–0.895) |
| Specialized image-level classifiers | HoVer-Trans | 0.686 | 0.597 | 0.582 | 0.766 | 0.763 | 0.593 | 0.664 | <ins>0.729</ins> | 23 | 0.808 (0.726–0.874) |
| Specialized image-level classifiers | MsGoF | 0.638 | 0.059 | 0.031 | **0.994** | 0.666 | 0.274 | 0.181 | **0.932** | 82 | 0.317 (0.235–0.408) |
| Lesion-annotation-assisted classifiers | CAM-QUS† | <ins>0.871</ins> | <ins>0.767</ins> | 0.857 | 0.760 | 0.834 | 0.604 | <ins>0.906</ins> | 0.482 | 18 | 0.850 (0.773–0.909) |
| Lesion-annotation-assisted classifiers | MB-DCNN† | 0.856 | 0.712 | 0.806 | 0.708 | 0.834 | 0.659 | 0.792 | 0.711 | 26 | 0.783 (0.699–0.853) |
| Multimodal-trained classifier (B-mode inference) | TDF-Net‡ | 0.850 | 0.698 | 0.673 | <ins>0.838</ins> | 0.800 | 0.614 | 0.886 | 0.526 | **15** | 0.875 (0.802–0.928) |
| Proposed image-level method | IADA-Net (ours) | **0.907** | **0.782** | **0.878** | 0.766 | **0.925** | **0.707** | **0.944** | 0.655 | **15** | 0.875 (0.802–0.928) |

External and clinical scores use mean fold probability per image followed by the maximum over each patient's images, with threshold 0.5. Clinical SEN uses an exact 95% Clopper–Pearson interval. Training-resource markers have the same meaning as Table 2.

## Table 4. Component ablations

| Variant | RP | DC | DBI | CQ | BUSI AUC | UDIAT AUC | ARC AUC | Macro F1 | BrEaST AUC | BUS-BRA AUC | Clinical missed / 120 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GAP readout | × | × | × | × | 0.960 | **0.914** | <ins>0.913</ins> | 0.832 | 0.902 | 0.918 | **14** |
| w/o explicit depth | ✓ | × | ✓ | ✓ | 0.962 | 0.894 | 0.894 | 0.831 | 0.901 | 0.913 | **14** |
| w/o DBI | ✓ | ✓ | × | ✓ | 0.959 | 0.897 | **0.921** | 0.829 | 0.896 | <ins>0.924</ins> | 17 |
| w/o CQ | ✓ | ✓ | ✓ | × | <ins>0.974</ins> | <ins>0.910</ins> | 0.903 | <ins>0.851</ins> | **0.909** | <ins>0.924</ins> | <ins>15</ins> |
| IADA-Net | ✓ | ✓ | ✓ | ✓ | **0.975** | <ins>0.910</ins> | 0.901 | **0.854** | <ins>0.907</ins> | **0.925** | <ins>15</ins> |

RP: row pooling; DC: explicit depth codes and depth biases; DBI: bidirectional interaction; CQ: conditioned query. All variants use URFM-L/16 and matched five-fold training. Removing CQ sets context_proj to zero. Per-source development entries are AUC; macro F1 gives the three sources equal weight.

## Table 5. Pretrained encoders

| Encoder and readout | Macro AUC | Macro F1 | BrEaST AUC | BUS-BRA AUC | External mean AUC | Clinical missed / 120 |
| --- | --- | --- | --- | --- | --- | --- |
| DINOv2-B/14 + GAP | 0.913 | 0.786 | <ins>0.902</ins> | 0.882 | 0.892 | **14** |
| DINOv2-B/14 + IADA-Net | 0.907 | 0.785 | 0.879 | 0.901 | 0.890 | **14** |
| DINOv2-L/14 + GAP | 0.914 | 0.801 | 0.888 | 0.903 | 0.896 | 17 |
| USFM + GAP | <ins>0.915</ins> | 0.805 | 0.859 | 0.861 | 0.860 | 21 |
| URFM-L/16 + GAP | **0.929** | <ins>0.832</ins> | <ins>0.902</ins> | <ins>0.918</ins> | <ins>0.910</ins> | **14** |
| URFM-L/16 + IADA-Net (ours) | **0.929** | **0.854** | **0.907** | **0.925** | **0.916** | <ins>15</ins> |

Macro AUC and F1 give BUSI, UDIAT and ARC equal weight. External AUC is patient-level. CLS, A and C are additional controls in variants.csv; they are not the final method or rows in this manuscript table.

## Table 6. Vertical reflection and conditioned query

| VR | CQ | Macro AUC | Macro F1 | BrEaST AUC | BUS-BRA AUC | Clinical missed / 120 |
| --- | --- | --- | --- | --- | --- | --- |
| ✓ | ✓ | 0.920 | 0.849 | 0.904 | 0.917 | 16 |
| ✓ | × | 0.920 | 0.845 | 0.907 | 0.917 | 16 |
| × | × | **0.929** | 0.851 | **0.909** | 0.924 | **15** |
| × | ✓ | **0.929** | **0.854** | 0.907 | **0.925** | **15** |

VR is vertical reflection at probability 0.2 in Base training. The proposed setting is no VR with CQ. All other training and evaluation settings are matched.

## Efficiency (Figure 7)

The [18-model RTX 5090 measurements](efficiency_5090.csv) use FP32, TF32 disabled, eval plus inference_mode, batch sizes 1 and 32, 10 warm-ups and 50 synchronized measurements. IADA-Net has 312.13 M parameters, 124.09 GFLOPs, 16.90 ms batch-1 latency and 312.42 images/s batch-32 throughput. Each method retains its own input size and complete inference path; uncounted FLOP operators are listed in the CSV.
