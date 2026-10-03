# Final URFM-L/16 results

These files export previously saved results. The repository update did not retrain models, run cohort inference, or recalculate statistics.

| File | Contents |
| --- | --- |
| `variants.csv` | Same-backbone and encoder comparisons, per-source five-fold means/sample SDs, external patient AUCs, and clinical FN. `urfm_iada` is the final R9 model; A/C are unsuccessful extensions. |
| `flip_cq.csv` | All four combinations of vertical flip and context-conditioned query (CQ). The final setting has no vertical flip and has CQ. |
| `external_clinical.csv` | Aggregate confusion counts/metrics for the nine reproduced methods and final IADA. Clinical rows retain TP/FN and sensitivity; mixed-class metrics are blank. |
| `statistics.json` | Original paired bootstrap AUC differences/95% intervals, exact McNemar tests, specificity-matching analyses, and matched-model summaries. Individual records and local filesystem paths are omitted. |
| `efficiency_5090.csv` | All 18 models on RTX 5090, retaining PyTorch FLOP-counter values and the original operator-omission notes. |

Internal mean/SD uses the five held-out folds; sample SD has `ddof=1`. Macro metrics give BUSI, UDIAT and ARC equal weight. External/clinical aggregation is five-fold mean probability per image followed by the patient maximum, with threshold `>=0.5`.

The clinical cohort comprises 120 malignant patients with 640 views. Its recorded truth is patient-level; clinical FN is not a per-view quantity. Specificity-matching thresholds are descriptive and do not replace the fixed deployment threshold. Pairwise intervals have no multiple-comparison correction, and the repeatedly observed external/clinical cohorts are exploratory.

The four older CSVs directly under `results/` use earlier DINOv2/controlled-head protocols. They are distinct from this final result set.
