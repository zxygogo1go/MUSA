# Reproduce MUSA

This repository already contains the core model code, training scripts,
inference script, and pretrained weights for M01, M04, and M05. The main missing
piece for a full reproduction is the preprocessed dataset.

## 1. Environment

Install PyTorch first using the CUDA version that matches your machine, then
install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Quick smoke test:

```bash
python -m unittest tests/test_smoke.py
```

## 2. Data Preparation

Prepare CT images, organ segmentations, and bone segmentations according to
`docs/data_format.md`.

Expected inputs:

```text
data/images/*.npy
data/seg_o/*.npy
data/seg_b/*.npy
data/lists/trn_list_inter.txt
data/lists/val_list_inter.txt
```

Preprocessing steps from the paper/README:

1. Remove background, scanner bed, and immobilization devices.
2. Reorient images to the project convention.
3. Rigidly align images to a common template.
4. Clip CT intensities to `[-1024, 3000]` HU and normalize to `[0, 1]`.
5. Resample to 2 mm isotropic spacing and crop to `(160, 160, 192)`.
6. Save processed arrays as `.npy`.

For aligned NIfTI inputs, use:

```bash
python scripts/preprocess/prepare_batch.py \
  --manifest data/manifest.csv \
  --out-root data
```

For SegRap-style case folders with per-structure masks, use:

```bash
python scripts/preprocess/prepare_segrap_batch.py \
  --cases-root /path/to/SegRap2023_Training_Set_120cases \
  --out-root data
```

Then create training and validation lists:

```bash
python scripts/preprocess/validate_prepared_data.py \
  --data-root data

python scripts/preprocess/make_lists.py \
  --image-dir data/images \
  --train-out data/lists/trn_list_inter.txt \
  --val-out data/lists/val_list_inter.txt \
  --val-pairs data/lists/val_pairs.csv
```

## 3. Stage 1: MUSA Loss at Half Resolution

Example with VoxelMorph:

```bash
python scripts/train/train_loss3musa_1stage.py \
  --trn-list data/lists/trn_list_inter.txt \
  --val-list data/lists/val_list_inter.txt \
  --vol-path data/images \
  --seg-path-o data/seg_o \
  --seg-path-b data/seg_b \
  --model-resolution r2 \
  --model-type 01voxelmorph-v1 \
  --lr 1e-4 \
  --loss-sim-type mse \
  --lambda 1.0 \
  --alpha 1000 \
  --gpu 0 \
  --batch-size 1 \
  --epochs 500 \
  --steps-per-epoch 100 \
  --epoch-save 10 \
  --epoch-val 10 \
  --out-dir outputs/loss3_m01_stage1
```

## 4. Stage 2: Frozen Stage 1 Plus Full-Resolution Residual Model

```bash
python scripts/train/train_loss3musa_2stage.py \
  --trn-list data/lists/trn_list_inter.txt \
  --val-list data/lists/val_list_inter.txt \
  --vol-path data/images \
  --seg-path-o data/seg_o \
  --seg-path-b data/seg_b \
  --model-type 01voxelmorph-v1 \
  --model-load-stage1 outputs/loss3_m01_stage1/final_model.pth \
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
  --out-dir outputs/loss3_m01_2stage
```

Repeat the same two-stage recipe with:

- `04transmorph-v1`
- `05dualprnet-v1`

## 5. Inference with Pretrained Weights

For prepared `.npy` data produced by this repository, run:

```bash
python scripts/infer/infer_prepared_pair.py \
  --moving-id segrap_0000 \
  --fixed-id segrap_0001 \
  --data-root data \
  --model-type 01voxelmorph-v1 \
  --checkpoint-stage1 outputs/loss3_m01_stage1_r2/checkpoint-cont0050/0500.pth \
  --checkpoint-stage2 outputs/loss3_m01_stage2_r1_from_stage1_0500/checkpoint/0500.pth \
  --output-dir outputs/infer_m01_loss3/segrap_0000_to_segrap_0001 \
  --gpu 0
```

This saves `.npy` outputs for the deformed image, composed DVF, warped
segmentations, and a JSON metrics file with Dice before/after registration.

To evaluate all validation pairs and write aggregate metrics:

```bash
python scripts/infer/eval_prepared_pairs.py \
  --pairs-csv data/lists/val_pairs.csv \
  --data-root data \
  --model-type 01voxelmorph-v1 \
  --checkpoint-stage1 outputs/loss3_m01_stage1_r2/checkpoint-cont0050/0500.pth \
  --checkpoint-stage2 outputs/loss3_m01_stage2_r1_from_stage1_0500/checkpoint/0500.pth \
  --output-dir outputs/eval_m01_loss3 \
  --gpu 0
```

Add `--save-pair-outputs` if you also want per-pair warped images, DVFs, and
warped segmentations saved under `outputs/eval_m01_loss3/pairs/`.

After preprocessing a moving/fixed pair to `(160, 160, 192)`, run:

```bash
python scripts/infer/infer_registration.py \
  --moving-img path/to/moving_img.nii.gz \
  --fixed-img path/to/fixed_img.nii.gz \
  --model M01 \
  --loss loss3 \
  --stage 2 \
  --output-dir outputs/infer_case001 \
  --output-prefix case001
```

If `--checkpoint-stage1` and `--checkpoint-stage2` are omitted, the script tries
to find matching checkpoints under `pretrained_models/`.
