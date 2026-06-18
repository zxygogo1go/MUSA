# HaN-Seg M05+MUSA/MUSA+ Experiment

This is the recommended pilot protocol for retraining on the 42-case HaN-Seg
CT set and comparing `M05+MUSA Stage2` with `M05+MUSA+ adaptive Stage3 safe`.

HaN-Seg is not a native longitudinal follow-up dataset; it contains CT/MR scans
from 42 patients with CT OAR labels. The split below treats it as an
inter-subject head-and-neck CT registration dataset.

## 1. Preprocess HaN-Seg

```bash
python scripts/preprocess/prepare_hanseg_batch.py \
  --cases-root HaN-Seg/set_1 \
  --out-root data_hanseg \
  --write-paper-split

python scripts/preprocess/validate_prepared_data.py \
  --data-root data_hanseg
```

The fixed split is under `data_hanseg/lists/paper_split`:

- train: 22 cases, including `hanseg_0019`
- val: 5 pairs, excluding case 19 because `OAR_OpticChiasm` is missing
- test: 5 held-out pairs from cases 01-10

## 2. Train M05+MUSA Stage1

```bash
python scripts/train/train_loss3musa_1stage.py \
  --trn-list data_hanseg/lists/paper_split/trn_list_inter.txt \
  --val-list data_hanseg/lists/paper_split/val_list_inter.txt \
  --vol-path data_hanseg/images \
  --seg-path-o data_hanseg/seg_o \
  --seg-path-b data_hanseg/seg_b \
  --model-resolution r2 \
  --model-type 05dualprnet-v1 \
  --lr 1e-4 \
  --loss-sim-type mse \
  --lambda 1.0 \
  --alpha 1.0 \
  --gpu 0 \
  --batch-size 1 \
  --epochs 500 \
  --steps-per-epoch 100 \
  --epoch-save 10 \
  --epoch-val 10 \
  --out-dir outputs_hanseg/paper_split_loss3_m05_stage1_r2
```

## 3. Train M05+MUSA Stage2

```bash
python scripts/train/train_loss3musa_2stage.py \
  --trn-list data_hanseg/lists/paper_split/trn_list_inter.txt \
  --val-list data_hanseg/lists/paper_split/val_list_inter.txt \
  --vol-path data_hanseg/images \
  --seg-path-o data_hanseg/seg_o \
  --seg-path-b data_hanseg/seg_b \
  --model-type 05dualprnet-v1 \
  --model-load-stage1 outputs_hanseg/paper_split_loss3_m05_stage1_r2/checkpoint/0500.pth \
  --model-load-stage2 from-scratch \
  --lr 1e-4 \
  --loss-sim-type mse \
  --lambda 1.0 \
  --gpu 0 \
  --batch-size 1 \
  --epochs 500 \
  --steps-per-epoch 100 \
  --epoch-save 10 \
  --epoch-val 10 \
  --out-dir outputs_hanseg/paper_split_loss3_m05_stage2_r1
```

## 4. Train Adaptive Stage3 Safe

```bash
export SMALL_HANSEG="OAR_Cochlea_L,OAR_Cochlea_R,OAR_OpticNrv_L,OAR_OpticNrv_R,OAR_OpticChiasm,OAR_Pituitary,OAR_Glnd_Lacrimal_L,OAR_Glnd_Lacrimal_R"

python scripts/train/train_musa_plus_stage3.py \
  --trn-list data_hanseg/lists/paper_split/trn_list_inter.txt \
  --val-list data_hanseg/lists/paper_split/val_list_inter.txt \
  --vol-path data_hanseg/images \
  --seg-path-o data_hanseg/seg_o \
  --seg-path-b data_hanseg/seg_b \
  --metadata-path data_hanseg/metadata \
  --small-oar-names "$SMALL_HANSEG" \
  --model-type 05dualprnet-v1 \
  --model-load-stage1 outputs_hanseg/paper_split_loss3_m05_stage1_r2/checkpoint/0500.pth \
  --model-load-stage2 outputs_hanseg/paper_split_loss3_m05_stage2_r1/checkpoint/0500.pth \
  --out-dir outputs_hanseg/paper_split_musa_plus_stage3_jac_safe \
  --lr 1e-4 \
  --batch-size 1 \
  --epochs 200 \
  --steps-per-epoch 100 \
  --epoch-save 10 \
  --epoch-val 10 \
  --gpu 0 \
  --stage3-input-mode full \
  --roi-radius-min 3 \
  --roi-radius-max 3 \
  --roi-smooth-steps 2 \
  --residual-scale-min 0.25 \
  --residual-scale-max 1.00 \
  --lambda-local-img 1.0 \
  --lambda-small-min 1.0 \
  --lambda-small-max 3.0 \
  --lambda-smooth 0.10 \
  --lambda-smooth-extra 0.20 \
  --lambda-mag 0.01 \
  --lambda-jacobian 1.0 \
  --jacobian-margin 0.05 \
  --jacobian-roi-weight 5.0 \
  --lambda-preserve-large 0.75 \
  --lambda-preserve-bone 0.75 \
  --best-policy noharm \
  --best-jacobian-penalty 10.0 \
  --best-large-drop-penalty 4.0 \
  --best-bone-drop-penalty 4.0
```

Use `best_stage3_noharm.pth` for the locked safe checkpoint.

## 5. Evaluate Val And Test

```bash
python scripts/infer/eval_musa_plus_prepared_pairs.py \
  --pairs-csv data_hanseg/lists/paper_split/val_pairs.csv \
  --data-root data_hanseg \
  --model-type 05dualprnet-v1 \
  --checkpoint-stage1 outputs_hanseg/paper_split_loss3_m05_stage1_r2/checkpoint/0500.pth \
  --checkpoint-stage2 outputs_hanseg/paper_split_loss3_m05_stage2_r1/checkpoint/0500.pth \
  --checkpoint-stage3 outputs_hanseg/paper_split_musa_plus_stage3_jac_safe/best_stage3_noharm.pth \
  --metadata-path data_hanseg/metadata \
  --small-oar-names "$SMALL_HANSEG" \
  --output-dir outputs_hanseg/paper_split_eval_musa_plus_stage3_jac_safe_val \
  --save-pair-metrics

python scripts/infer/eval_musa_plus_prepared_pairs.py \
  --pairs-csv data_hanseg/lists/paper_split/test_pairs.csv \
  --data-root data_hanseg \
  --model-type 05dualprnet-v1 \
  --checkpoint-stage1 outputs_hanseg/paper_split_loss3_m05_stage1_r2/checkpoint/0500.pth \
  --checkpoint-stage2 outputs_hanseg/paper_split_loss3_m05_stage2_r1/checkpoint/0500.pth \
  --checkpoint-stage3 outputs_hanseg/paper_split_musa_plus_stage3_jac_safe/best_stage3_noharm.pth \
  --metadata-path data_hanseg/metadata \
  --small-oar-names "$SMALL_HANSEG" \
  --output-dir outputs_hanseg/paper_split_eval_musa_plus_stage3_jac_safe_test \
  --save-pair-metrics
```

## 6. Build Paper Tables

```bash
python scripts/infer/compare_musa_plus_stage2_stage3.py \
  --eval-dir outputs_hanseg/paper_split_eval_musa_plus_stage3_jac_safe_test \
  --output-dir outputs_hanseg/paper_split_compare_stage2_stage3_safe_test_full \
  --stage2-name "HaN-Seg M05+MUSA Stage2" \
  --stage3-name "HaN-Seg M05+MUSA+ adaptive Stage3 safe"

python scripts/infer/summarize_musa_plus_per_label.py \
  --eval-dir outputs_hanseg/paper_split_eval_musa_plus_stage3_jac_safe_test \
  --data-root data_hanseg \
  --output-dir outputs_hanseg/paper_split_compare_stage2_stage3_safe_test_per_label
```

After these finish, compare the HaN-Seg test summary with the SegRap test
summary using the same metrics: all-OAR Dice, small-OAR Dice, large-OAR
preservation, bone Dice, ROI Jacobian non-positive ratio, and residual DVF p95.

## 7. Render Visual Comparisons

The wrapper below selects representative test pairs from
`stage2_vs_stage3_by_pair.csv`, runs single-pair inference to save warped
images/segmentations, then renders MUSA Stage2 vs adaptive Stage3 PNG figures.

```bash
export SMALL_HANSEG="OAR_Cochlea_L,OAR_Cochlea_R,OAR_OpticNrv_L,OAR_OpticNrv_R,OAR_OpticChiasm,OAR_Pituitary,OAR_Glnd_Lacrimal_L,OAR_Glnd_Lacrimal_R"

python scripts/infer/make_musa_plus_visual_comparison.py \
  --data-root data_hanseg \
  --model-type 05dualprnet-v1 \
  --checkpoint-stage1 outputs_hanseg/paper_split_loss3_m05_stage1_r2/checkpoint/0500.pth \
  --checkpoint-stage2 outputs_hanseg/paper_split_loss3_m05_stage2_r1/checkpoint/0500.pth \
  --checkpoint-stage3 outputs_hanseg/paper_split_musa_plus_stage3_jac_safe/best_stage3_noharm.pth \
  --metadata-path data_hanseg/metadata \
  --small-oar-names "$SMALL_HANSEG" \
  --compare-csv outputs_hanseg/paper_split_compare_stage2_stage3_safe_test_full/stage2_vs_stage3_by_pair.csv \
  --selection top-small,worst-large,worst-jac \
  --num-pairs 3 \
  --output-dir outputs_hanseg/paper_split_visual_musa_vs_stage3_safe_test \
  --gpu 0
```

Open `outputs_hanseg/paper_split_visual_musa_vs_stage3_safe_test/index.md` to
inspect the generated figures.

Color convention:

- fixed label: green
- original moving label: red
- MUSA Stage2 warped label: cyan
- adaptive Stage3 warped label: yellow
