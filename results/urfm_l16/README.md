# Results for the URFM-L/16 manuscript

The final model is R9 `u16_iada`: URFM-L/16 + IADA-Net, without vertical reflection and with the conditioned query.

| File | Contents |
| --- | --- |
| [PAPER_RESULTS.md](PAPER_RESULTS.md) | The six manuscript tables at their displayed precision and with the same method groups. |
| [development.csv](development.csv) | Unrounded saved AUC/F1 means and sample SDs for BUSI, UDIAT and ARC for the nine comparators and final IADA-Net. Includes the completed TDF-Net BUSI/UDIAT B-mode inference. |
| [variants.csv](variants.csv) | Saved same-backbone and encoder controls, per-source five-fold means/SDs, external metrics and clinical FN. Final: `urfm_iada`. CLS, A and C are additional controls outside the manuscript's main tables. |
| [flip_cq.csv](flip_cq.csv) | The four vertical-reflection × conditioned-query settings of manuscript Table 6, including per-source values. |
| [external_clinical.csv](external_clinical.csv) | Aggregate counts and metrics for the nine reproduced methods and final IADA. Historical internal method identifiers are retained; the manuscript table gives their display names. |
| [statistics.json](statistics.json) | Saved class-stratified paired AUC bootstrap intervals, exact clinical McNemar tests and specificity-matching analyses. |
| [efficiency_5090.csv](efficiency_5090.csv) | The 18-model RTX 5090 measurements used in Figure 7, with full input paths and operator-omission notes. |

Development metrics use image-level mean and sample SD over five held-out folds, with ddof=1. Macro metrics give BUSI, UDIAT and ARC equal weight. TDF-Net uses the corresponding outer-fold ARC-triplet-trained model for every BUSI, UDIAT and ARC development image.

External and clinical aggregation is five-fold mean probability per image followed by the patient maximum, with probability >= 0.5 malignant. AUC-difference intervals use 2,000 class-stratified paired patient resamples. Clinical truth is patient-level for 120 malignant patients with 640 views; its reported FN counts missed patients.

Specificity-matching thresholds describe operating points and do not replace the deployment threshold. Paired comparisons use no multiple-comparison correction, and the repeatedly observed target cohorts are exploratory.

The four CSVs directly under `results/` belong to the earlier DINOv2 study and its evaluation protocol. Use this subdirectory for the URFM manuscript.
