# Train the fixed-source IADA models

`iada.train_source` is the separate training entry point for the pooled-source study. It consumes the exact inputs prepared by [source_protocol](SOURCE_PROTOCOL.md), rather than the older `data/splits/` workflow. It supports the binary-only base, its soft-label and hard-label query variants, and the true/shuffled source-reader controls. It does not start an additional seed repetition.

## Prepare paths and inputs

Install the optional source dependencies with `pip install -e '.[source,test]'`. Download the source images and official DINOv2 initialization from the providers linked in the README. Prepare the manifests and schedules as described in [SOURCE_PROTOCOL.md](SOURCE_PROTOCOL.md); use the reader workbook when running either graded arm.

Create a local `roots.json`, adjusting these locations to your downloads:

```json
{
  "busi": "data/raw/BUSI",
  "udiat": "data/raw/UDIAT/original",
  "arc": "data/raw/ARC/BD3M",
  "auxiliary": "data/raw/GDPH_SYSUCC"
}
```

Root paths may be absolute or relative to the command's working directory. Before fitting, the runner verifies the prepared file hashes and the original image hashes. It loads only the training and inner-selection images. Outer-test images are not used for checkpoint selection; clinical and external images are not inputs to training.

## Run one fold and one stage

For the binary-only model:

```bash
python -m iada.train_source --prepared outputs/source_protocol --roots roots.json --fold 1 --stage base --arm binary --backbone-weights weights/dinov2_vitb14_pretrain.pth --output outputs/source_binary/fold1/base --device cuda

python -m iada.train_source --prepared outputs/source_protocol --roots roots.json --fold 1 --stage query --arm binary --base-checkpoint outputs/source_binary/fold1/base/best.pt --query-label-smoothing 0.0 --output outputs/source_binary/fold1/hard_query --device cuda
```

To reproduce the soft-label query control, use the **same base checkpoint** with `--query-label-smoothing 0.1` and a new query output directory. The base stage has 100 source passes. The query stage has 40 and includes the initial zero adapter as a selection candidate. Only the 590,592 query-adapter parameters update in that stage. The seed is fixed at 42; repeat the commands for folds 2 through 5 using their corresponding bases.

For reader supervision, prepare inputs with the official workbook, then set `--arm true_grade` or `--arm shuffled_grade` in both stages and use separate output directories. Each query must use its own arm's selected base. Graded arms fix query label smoothing to zero. The base's 266-parameter ordinal head follows the settings in [READER_GRADES.md](READER_GRADES.md); it is removed before ordinary query adaptation and needs no grade at inference.

All training uses the fixed paired image/augmentation/dropout streams. Base training uses weighted cross-entropy with smoothing 0.1; query training uses the mean of per-image weighted cross-entropies plus ranking weight 0.1. These two reductions are intentionally distinct. The core uses AdamW, weight decay 0.01, and the registered warmup/cosine schedule. The shared gradient-norm bound is 1. CUDA query training uses BF16 autocast; evaluation is FP32. The deterministic positional operator preserves the native bicubic forward and uses its spatial transpose for the first derivative. CPU execution is supported but does not claim the same trajectory as the CUDA benchmark.

Checkpoint selection uses equal-dataset inner F1 at 0.5, then AUC for ties, then the earliest candidate within tolerance 1e-12. A single selected checkpoint supplies every metric for a fold. The command records model/code/input bindings, image exposures, per-round transformation hashes, source selection predictions, selected/final checkpoints, and resumable optimizer states.

## Interrupt and resume

An output directory must be new unless `--resume` is supplied. To request a clean stop, create an empty file named `STOP_AFTER_ROUND` inside that stage's output directory. The runner stops at the next round boundary and writes a snapshot. Remove that marker before resuming with the same command plus `--resume`.

Resume verifies the checkpoint hash, configuration and input bindings, and the previous process's PID/creation time. It refuses to resume an owner that is still running or a phase already marked complete. An unclean interruption resumes from the last recorded ten-round snapshot; later partial history is retained in an interruption folder. The completed-round schedules and dropout seeds are replayed. An elapsed monitoring timeout is not a reason to restart a live process.

## Predict and score

Generate complete image probabilities from each selected fold checkpoint:

```bash
python -m iada.predict_manifest --checkpoint outputs/source_binary/fold1/hard_query/best.pt --manifest images.csv --output outputs/predictions/fold1.csv --device cuda
```

The image manifest needs `sample_id` and `image_path`; relative paths can use `--data-root`. Prepared source manifests instead use `root_key`/`relative_path` with `--roots roots.json`. A provided `image_sha256` is checked. Diagnosis labels are ignored during inference. The loader automatically removes only the exact known reader-head tensors when loading a graded base checkpoint.

Probabilities use the sigmoid of the float64 difference between FP32 class logits, matching the source study. Each file preserves the complete manifest order and includes a checkpoint/manifest hash receipt. It does not apply an image decision threshold or perform patient aggregation. Pass all five probability files and the separate patient-label manifest to `iada.score_patient_series`, as described in [PATIENT_SERIES.md](PATIENT_SERIES.md). Its malignant-only mode reports patient sensitivity, TP/FN and the exact interval.

The archived `iada.predict` and `iada.evaluate` commands retain their historical decision and aggregation conventions. Use the new manifest/scorer pair for the patient-series protocol.

## Verification scope

The portable training primitives were checked against the registered implementation with real source images at batch size 16. Seven binary/graded base/query configurations matched initial tensors, outputs, losses, all computed gradients, clipping, every scheduled learning rate and optimizer parameter group. Twelve first/last-round data batches matched pixels, labels, indices and seeds; all ten full source schedules and all fifteen partitions were separately verified. A synthetic interruption test reproduced the uninterrupted final and selected tensors and all per-epoch predictions.

These are implementation checks, not new classification results or a fresh fivefold retraining. Hardware and library changes can still change a full optimization trajectory. The ongoing study's empirical results must be read from its completed evaluation and independent audit, not inferred from these tests.
