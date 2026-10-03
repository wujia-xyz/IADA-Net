# Training-only source reader grades

`iada.reader_grades` supplies the ordinal auxiliary head used in the registered source-reader study. This is a training component. It is not a trained classifier, an independent diagnostic output, or evidence of improved external or clinical performance. The study is still running; its results are not included in this release.

The head reads the existing 256-dimensional input to IADA's diagnosis classifier. It learns one scalar image severity and five increasing cutpoints for each of two readers, adding 266 parameters. Grade indices 0 through 5 correspond to BI-RADS 2, 3, 4A, 4B, 4C, and 5. Each output is the logit that a reader's grade exceeds a category boundary. The objective averages binary cross-entropy over boundaries and available reader annotations. Reader-specific softplus gaps keep cutpoints ordered.

Ordinal supervision is established prior work, including [BVA-Net](https://doi.org/10.1109/JBHI.2020.3034804) and [CORAL](https://doi.org/10.1016/j.patrec.2020.11.008). The head is separate from the row-aggregation analysis in [THEORY.md](THEORY.md); the auxiliary loss is not presented as a new theoretical contribution.

## Use in a training loop

```python
from iada.reader_grades import attach_reader_head, reader_grade_loss

attach_reader_head(model)  # Before constructing the optimizer.

optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = classification_loss(logits, diagnosis_labels)
loss = loss + 0.1 * reader_grade_loss(model, grade_indices, grade_available)
loss.backward()
# Apply the study's common gradient bound, then optimizer.step().
```

`grade_indices` and `grade_available` have shape `[batch, 2]` on the same device as the model features. The indices must represent grades, not binary pathology labels. Unknown entries may use -1 and must have a false availability mask. Unavailable grades do not remove an image or its diagnosis label from classification training. Ambiguous exact-image aliases are handled independently for each reader before training; the module does not resolve or guess source annotations.

When no grades are available in a batch, the auxiliary loss does not connect to any auxiliary parameter. With `zero_grad(set_to_none=True)`, AdamW leaves its parameters, weight decay, moments, and step counters unchanged for that batch. It still updates the diagnosis model from its classification loss. The tests cover this case after a previous annotated optimizer step.

## Fixed study settings and control

The source study uses one master seed 42 with five outer folds and separate inner checkpoint selection. Each epoch retains the original source images and the same-class auxiliary observations. Source pathology labels and image schedules are identical in both auxiliary-supervision arms. One arm uses the available grades. The correspondence control jointly shuffles the two readers' grades within hospital, pathology class, and availability strata; it preserves the joint grade distribution. This permutation is a declared control, not a correction to the source annotations.

The base stage uses 100 passes and an auxiliary coefficient of 0.1. Its existing classification objective and core optimizer remain unchanged. The new head follows 100 times the core learning rate, with weight decay 0.01 for its severity weights and zero for its cutpoints. The common gradient-norm bound is 1. After selection by inner-source classification F1, AUC for ties, and earliest candidate thereafter, the reader head is removed. The query stage uses the ordinary 40-pass hard-label objective, ranking coefficient 0.1, and a selectable epoch-zero checkpoint. Neither the reader grade nor a grading branch is used during diagnosis.

This module does not change `iada.train`, the historical data partitions, or the archived release results. Running that historical command is not a replay of the later pooled-source and paired-grade experiment. The separate [iada.train_source entry point](TRAIN_SOURCE.md) uses the fixed source admissions, nested partitions and common image/augmentation schedules. The component tests verify behavior, not a complete retraining or cross-hardware equality.

## Remove the auxiliary head

For inference with the same in-memory model:

```python
from iada.reader_grades import remove_reader_head
remove_reader_head(model)
model.eval()
```

For a saved state dictionary:

```python
from iada.reader_grades import diagnosis_state
diagnosis_model.load_state_dict(diagnosis_state(training_state), strict=True)
```

The filter removes exactly the three expected `reader_ordinal.*` tensors and rejects an incomplete or unknown auxiliary schema. It does not alter any diagnosis tensor. Checkpoints and patient-level data are not distributed here.
