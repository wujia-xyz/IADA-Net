# Reproduction notes

## Final URFM-L/16 model

`iada.urfm.URFMIADA` is the final R9 `u16_iada` architecture: URFM ViT-L/16, conditional row pooling on a 14 × 14 grid, explicit depth encoding, two bidirectional interaction layers, and a two-class classifier. State names match the selected full checkpoints, including `encoder.net.*` and `row_pooling.context_proj.*`.

The Query model contains 312,133,266 parameters. Query trains only the 1,049,600 parameters of `context_proj`; the encoder and remaining head stay fixed in evaluation mode. Zero initialization preserves the selected Base logits. Gradients pass through the fixed readout to the adapter.

The portable selected Base → Query training path and full-checkpoint inference were checked on CPU, including original full-model and official EMA loading. This release update did not rerun model fitting, five-fold inference, or saved statistical calculations. The passive SWA branch is not implemented in the new entry point.

## Source training

Exact source memberships and order are in `data/paired_source_seed42/`: 1,052 original images and 2,301 auxiliary images. Each outer fold has a separate inner selection partition. Known image families do not cross roles; this establishes image-family grouping, not complete patient identity for datasets that lack it.

The Base schedule contains 250 epochs for historical comparator budgets; final IADA consumes the first 100. Query uses 40. Each epoch observes every original training image once and one same-label auxiliary image per original, alternating GDPH/SYSUCC within each diagnosis class. Master seed 42 and original augmentation/dropout substreams are retained.

The final workflow checks paths, labels and schedule structure without hashing files or generating audit receipts. Stored historical metadata hashes are not prerequisites. [Training settings](TRAIN_SOURCE.md) and [source preparation](SOURCE_PROTOCOL.md) describe the executable workflow.

## Final evaluation and saved results

Internal results use mean and sample SD (`ddof=1`) over five held-out folds per source. Macro metrics give the three source datasets equal weight. Checkpoints are selected on inner macro F1, then macro AUC, then earliest within 1e-12; outer, external and clinical labels do not enter that selection.

External and clinical evaluation averages five fold probabilities per image, then takes each patient's maximum image probability. Exact 0.5 ties are malignant. Clinical truth exists only at the patient level for 120 malignant patients and 640 views. Clinical summaries therefore report TP/FN and sensitivity, rather than per-view FN, AUC, or specificity.

`results/urfm_l16/` copies existing aggregate results, including the full CQ × vertical-flip ablation and unsuccessful A/C extensions. `statistics.json` retains the original paired bootstrap settings, AUC differences, McNemar tests, and specificity-matching analyses. Matching thresholds are descriptive operating-point analyses; deployment remains at 0.5. Repeatedly observed external and clinical cohorts provide exploratory evidence.

## RTX 5090 efficiency

`efficiency_5090.csv` retains the original 18-model measurement: each experimental graph and its own input size, FP32, TF32 off, `eval()` plus `inference_mode()`, batches 1/32, 10 warm-ups, and 50 synchronized timing measurements summarized by the median. Parameter counts include all registered parameters. GFLOPs are batch-1 PyTorch `FlopCounterMode` counts, without manual supplements. ViT-B and TDF-Net have uncounted native multihead attention; other omissions appear per row. Peak memory uses PyTorch allocated MiB.

MB-DCNN includes coarse segmentation and classification along the original crop/view path. TDF-Net takes only B-mode, including its missing-feature reconstruction path. MsGoF uses retained fold-1 weights because random FP32 initialization overflowed; no efficiency training was performed.

## Archived DINOv2 study

The earlier DINOv2-B/14 implementation uses a 16 × 16 grid. Its training, evaluation and controlled-head tools retain historical conventions: the old evaluator uses patient means, negative 0.5 ties, and population SD in archived aggregates. Use `iada.train_urfm`, `iada.predict_manifest` and `iada.score_patient_series` for the final study.

The optional reader-grade and older `iada.train_source` modules are historical controls. The final classifier requires no reader grades or external/clinical training input.