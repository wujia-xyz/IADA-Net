# Train the final URFM-IADA model

Use `python -m iada.train_urfm` after preparing `data/paired_source_seed42/` with `iada.source_protocol` and mapping the four dataset roots in `roots.json`.

```bash
python -m iada.train_urfm --config configs/urfm_base.json --prepared outputs/source_protocol --roots roots.json --fold 1 --output outputs/urfm/fold1/base --backbone-weights weights/mae_vit_large_patch16_dec768d8b_all_biomedclip_1199.pth --device cuda
python -m iada.train_urfm --config configs/urfm_query.json --prepared outputs/source_protocol --roots roots.json --fold 1 --output outputs/urfm/fold1/query --base-checkpoint outputs/urfm/fold1/base/best.pt --device cuda
```

Both stages use seed 42, batch 16, prepared image schedules, class weights from original fold training labels, AdamW (weight decay 0.01; fused on CUDA), gradient clipping at 1, and the recorded augmentation/dropout seeds. TF32 is disabled; deterministic algorithms are enabled.

| Setting | Base | Query |
| --- | --- | --- |
| Duration | 100 epochs | 40 epochs; epoch zero eligible |
| Initialization | URFM ViT-L/16 `ema_state_dict` | Selected same-fold Base + zero adapter |
| Trainable parameters | Encoder and fixed-query IADA head | `context_proj` only |
| Training precision | FP32 | CUDA BF16; CPU FP32 |
| Loss | Weighted CE, smoothing 0.1 | Mean per-example weighted CE, smoothing 0, + 0.1 ranking loss |
| Learning rate | 5-epoch warm-up to 1e-5; cosine to 1e-6 | Cosine from 1e-4 to 1e-5 |
| Augmentation | Original Base recipe with vertical reflection disabled | Original mild Query recipe |

Base uses PyTorch's weighted-mean CE reduction. Query averages weighted per-example CE without renormalizing by the batch sum of weights. Ranking applies softplus to negative-minus-positive logit margins, and is zero for single-class batches.

Selection gives the three source datasets equal weight: macro F1 at 0.5, macro AUC for ties, then earliest within 1e-12. Base considers epochs 1–100; Query also considers epoch zero. Evaluation uses FP32 and float64 logit differences.

Outputs are `best.pt`, `history.json`, `selection.csv`, and `RESULT.json`. An incomplete run keeps `resume.pt` with optimizer/RNG states; `--resume` continues it. A completed run removes the temporary resume checkpoint. No hashes or audit receipts are generated. External and clinical images are excluded from this source-only entry point.

This implements selected Base → Query, without the historical passive SWA branch. The older `iada.train_source` and reader-grade modules retain a separate historical DINOv2 workflow and are not the final training command.