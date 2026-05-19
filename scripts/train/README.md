
# Training Scripts

**Note:** LapIRN requires its own training script, see https://github.com/cwmok/LapIRN

## Loss1: Standard Loss (Lsim + Lreg)

- `train_loss1std_1stage.py`: Train with standard loss (MSE or NCC) for 1-stage registration (half/full resolution)
- `train_loss1std_2stage.py`: Train with standard loss (MSE or NCC) for 2-stage registration (stage1: half resolution/weight freezed, stage2: full resolution)

## Loss2: Standard Loss + Dice (Weak) Supervision (Lsim + Lreg + Lseg)

- `train_loss2dice_1stage.py`: Train with Dice loss for 1-stage registration (half/full resolution)
- `train_loss2dice_2stage.py`: Train with Dice loss for 2-stage registration (half resolution/weight freezed, stage2: full resolution)

## Loss3: Standard Loss + MUSA Loss Regularization (Lsim + Lreg + Lmusa)

- `train_loss3musa_1stage.py`: Train with MUSA loss for 1-stage registration (half resolution, full resolution is unstable)
- `train_loss3musa_2stage.py`: Train with MUSA loss for 2-stage registration (half resolution/weight freezed, stage2: full resolution)

> **Note:** For `train_loss3musa_2stage.py`, stage 2 uses standard loss (Lsim + Lreg). In other words, MUSA loss is only used in training the half-resolution models in `train_loss3musa_1stage.py`

## MUSA+: Stage-3 Small-OAR Local Refinement

- `train_musa_plus_stage3.py`: Freeze a trained two-stage DIR-MUSA model and train a lightweight Stage-3 residual U-Net around small OARs.

This is the Phase-1 low-risk prototype from `docs/musa_plus_research_plan.md`. It uses:

- rule-based pair difficulty from initial bone Dice, OAR Dice, and image MSE;
- difficulty-adaptive ROI dilation, small-OAR loss weight, residual scale, and smoothness;
- rule-based anatomy-conditioned regularization maps for bone, small-OAR ROI, and ROI boundary.

The Stage-3 information policy is controlled by `--stage3-input-mode`:

- `full`: current segmentation-guided prototype, using fixed small-OAR and fixed bone masks.
- `no-fixed-small`: zeros the fixed small-OAR channel and builds ROI from the Stage-2 warped moving small-OAR mask.
- `no-fixed-seg`: zeros fixed small-OAR and fixed bone channels, builds ROI from warped moving small-OAR, and uses CT/flow-only difficulty.

Example:

```bash
python scripts/train/train_musa_plus_stage3.py \
  --trn-list data/lists/trn_list_inter.txt \
  --val-list data/lists/val_list_inter.txt \
  --vol-path data/images \
  --seg-path-o data/seg_o \
  --seg-path-b data/seg_b \
  --metadata-path data/metadata \
  --model-type 05dualprnet-v1 \
  --model-load-stage1 /path/to/m05_stage1_musa_r2.pth \
  --model-load-stage2 /path/to/m05_stage2_r1.pth \
  --out-dir runs/musa_plus_stage3_m05 \
  --batch-size 1 \
  --epochs 200 \
  --steps-per-epoch 100 \
  --stage3-input-mode full
```

If metadata is unavailable, pass explicit labels:

```bash
--small-oar-labels 3,4,7,8,12
```

After training, run one prepared pair with:

```bash
python scripts/infer/infer_musa_plus_prepared_pair.py \
  --moving-id segrap_0000 \
  --fixed-id segrap_0001 \
  --data-root data \
  --model-type 05dualprnet-v1 \
  --checkpoint-stage1 /path/to/m05_stage1_musa_r2.pth \
  --checkpoint-stage2 /path/to/m05_stage2_r1.pth \
  --checkpoint-stage3 runs/musa_plus_stage3_m05/best_stage3.pth \
  --output-dir runs/musa_plus_stage3_m05/infer_pair
```

For leakage ablation with an existing checkpoint, rerun inference with:

```bash
--stage3-input-mode no-fixed-small
```

or the stricter:

```bash
--stage3-input-mode no-fixed-seg
```

The MUSA+ metrics JSON reports:

- `leakage_audit`: whether fixed labels were used as Stage-3 input, ROI source, or difficulty features.
- `small_oar_per_label`: integer-label small-OAR Dice, not just binary soft Dice.
- `large_oar_per_label` and `bone_per_label`: worst label drop and counts of drops larger than 0.02 / 0.05.
- `stage3.residual_magnitude`: global and ROI mean/p95/max residual magnitude.
- `stage3.final_jacobian`: global and ROI Jacobian minimum, low percentiles, and non-positive ratio.

Run the whole validation pair CSV:

```bash
python scripts/infer/eval_musa_plus_prepared_pairs.py \
  --pairs-csv data/lists/val_pairs.csv \
  --data-root data \
  --model-type 05dualprnet-v1 \
  --checkpoint-stage1 /path/to/m05_stage1_musa_r2.pth \
  --checkpoint-stage2 /path/to/m05_stage2_r1.pth \
  --checkpoint-stage3 runs/musa_plus_stage3_m05/best_stage3.pth \
  --output-dir runs/musa_plus_stage3_m05/eval_full \
  --stage3-input-mode full
```
