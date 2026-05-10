# Preprocessing Helpers

The full paper preprocessing pipeline includes background removal, rigid
template alignment, reorientation, resampling, cropping, and intensity
normalization. Those clinical-data-specific steps still need to be implemented
or performed with your local tooling.

This folder contains small repository-specific helpers for the final prepared
arrays.

## Prepare One Case

Use `prepare_case.py` after upstream background removal, orientation
standardization, and rigid template alignment:

```bash
python scripts/preprocess/prepare_case.py \
  --case-id case001 \
  --ct raw_aligned/case001_ct.nii.gz \
  --seg-o raw_aligned/case001_oar.nii.gz \
  --seg-b raw_aligned/case001_bone.nii.gz \
  --out-root data
```

Outputs:

```text
data/images/case001.npy
data/seg_o/case001.npy
data/seg_b/case001.npy
data/metadata/case001.json
```

The script resamples to 2 mm isotropic spacing, center crops/pads to
`(160, 160, 192)`, normalizes CT intensities from `[-1024, 3000]` to `[0, 1]`,
and uses nearest-neighbor interpolation for labels.

## Prepare a Batch

Create a CSV manifest:

```text
case_id,ct,seg_o,seg_b
case001,raw_aligned/case001_ct.nii.gz,raw_aligned/case001_oar.nii.gz,raw_aligned/case001_bone.nii.gz
case002,raw_aligned/case002_ct.nii.gz,raw_aligned/case002_oar.nii.gz,raw_aligned/case002_bone.nii.gz
```

Then run:

```bash
python scripts/preprocess/prepare_batch.py \
  --manifest data/manifest.csv \
  --out-root data
```

## Make Train/Validation Lists

After saving preprocessed arrays to `data/images/*.npy`, generate list files:

```bash
python scripts/preprocess/make_lists.py \
  --image-dir data/images \
  --train-out data/lists/trn_list_inter.txt \
  --val-out data/lists/val_list_inter.txt \
  --val-pairs data/lists/val_pairs.csv
```

`val_pairs.csv` should contain one moving/fixed pair per line:

```text
moving001,fixed001
moving002,fixed002
```

The generated `val_list_inter.txt` follows the repository dataloader convention:
all moving IDs first, then all fixed IDs.
