# Data for the manuscript

The final study uses the following curated collections.

| Dataset | Role | Evaluation unit | Count | Benign | Malignant |
| --- | --- | --- | ---: | ---: | ---: |
| BUSI | Development | Images | 645 | 436 | 209 |
| UDIAT | Development | Images | 159 | 107 | 52 |
| ARC | Development | Cases / B-mode images | 248 | 145 | 103 |
| GDPH/SYSUCC | Auxiliary training | Images | 2,301 | 851 | 1,450 |
| BrEaST | External | Patients | 252 | 154 | 98 |
| BUS-BRA | External | Patients (1,875 images) | 1,064 | 722 | 342 |
| Clinical series | Clinical | Patients (640 views) | 120 | 0 | 120 |

The 1,052 development images form 934 image families. Normal images and exact duplicates are removed, including the BUSI duplicate pair with conflicting labels. Known related images stay together within the five outer folds and their nested training, selection and test roles. BUSI and UDIAT do not supply complete patient identifiers, so development metrics are image-level.

## Source training inputs

Use the released metadata in `data/paired_source_seed42/` and prepare it with `python -m iada.source_protocol`. [Source preparation](SOURCE_PROTOCOL.md) documents dataset-root mapping, exact memberships and row order, and the original/auxiliary schedules. Source labels are 0=benign and 1=malignant. Each original training observation is paired with one same-label auxiliary observation; auxiliary images are used only in training.

The final model consumes a complete B-mode image resized to 224 × 224 with three channels and ImageNet normalization. Public images enter as stored. Clinical DICOM screens are cropped to their B-mode field at native pixels before resizing. IADA-Net does not consume lesion masks, lesion crops, Doppler or elastography.

## Evaluation inputs

Each external or clinical manifest supplies image paths, patient membership and explicit patient-level truth:

```csv
sample_id,image_path,patient_id,patient_label
image_a1,images/image_a1.png,patient_a,0
image_a2,images/image_a2.png,patient_a,0
image_b1,images/image_b1.png,patient_b,1
```

These are illustrative filenames. Run `iada.predict_manifest` once for each selected fold model and pass the five probability files to `iada.score_patient_series`. The rule is **mean fold probability per image, then maximum image probability per patient**, with probability >= 0.5 malignant. Patient labels must be consistent within a patient.

On the clinical series, truth is available for the 120 malignant patients. Their 640 views have no separate pathology labels. Use `--cohort-kind malignant-only` to report missed patients and sensitivity with an exact interval. Use `--cohort-kind binary` for BrEaST and BUS-BRA. [Patient-series evaluation](PATIENT_SERIES.md) describes class-stratified paired bootstrap intervals and exact clinical McNemar comparisons.

## Comparator resources and availability

CAM-QUS and MB-DCNN additionally use available BUSI/UDIAT lesion masks during training. TDF-Net trains on ARC B-mode, Doppler and elastography triplets and uses its missing-modality recovery for B-mode inference. It scores every development image with the TDF model corresponding to that image's outer fold.

Obtain images from their original providers under their terms. Clinical images and per-patient predictions are private. Trained weights and the full study image lists are available from the corresponding authors upon reasonable request. The older `data/splits/` manifests belong to the archived DINOv2 study; the final study uses `data/paired_source_seed42/`.
