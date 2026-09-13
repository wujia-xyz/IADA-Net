# Data preparation

Training manifests use these CSV fields:

```csv
sample_id,image_path,label,fold,patient_id,image_sha256
example-1,benign/example-1.png,0,1,,
example-2,malignant/example-2.png,1,2,,
```

`image_path` is relative to the supplied dataset root. Labels are 0=benign and 1=malignant. Internal fold numbers are 1–5. The optional hash identifies the original released image; the example filenames above are illustrative.

For external evaluation, include `case_id` or `patient_id` so that multiple images of one patient are averaged together:

```csv
sample_id,image_path,label,case_id
image-1,images/image-1.png,0,1
image-2,images/image-2.png,0,1
image-3,images/image-3.png,1,2
```

Conflicting labels within one patient are rejected. Patient IDs are sorted numerically when possible to retain the original bootstrap ordering. Use dataset keys `breast` and `busbra` for the recorded seed conventions. For BrEaST, use the 252 lesion-containing images, excluding the four normal cases. BUS-BRA contains 1,875 images from 1,064 patients in the reported external evaluation.

The internal manifests preserve the published development partitions. BUSI and UDIAT contain cross-fold duplicates, including a conflicting-label pair in BUSI, as disclosed in the paper. Do not treat the retained internal scores as an independent patient-level test. For a new independent experiment, review duplicate groups and labels before repartitioning; that produces a new protocol and new results. Images are supplied by their original providers under their access and use conditions.
