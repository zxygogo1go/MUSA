# Data Format

This project expects preprocessed head-and-neck CT volumes and segmentations.
The processed dataset is not included in the repository.

## Directory Layout

Recommended layout:

```text
data/
  images/
    case001.npy
    case002.npy
  seg_o/
    case001.npy
    case002.npy
  seg_b/
    case001.npy
    case002.npy
  lists/
    trn_list_inter.txt
    val_list_inter.txt
```

The training dataloader appends `.npy` by default, so list files should contain
case IDs without suffixes:

```text
case001
case002
case003
```

Raw-to-array conversion helpers live in `scripts/preprocess/`. The batch helper
expects a CSV manifest:

```text
case_id,ct,seg_o,seg_b
case001,raw_aligned/case001_ct.nii.gz,raw_aligned/case001_oar.nii.gz,raw_aligned/case001_bone.nii.gz
```

## Array Requirements

Full-resolution arrays:

- Shape: `(160, 160, 192)`
- Spacing: 2 mm isotropic
- Orientation: `i`: right-to-left, `j`: anterior-to-posterior, `k`: inferior-to-superior
- CT intensity: clipped to `[-1024, 3000]` HU and normalized to `[0, 1]`

Half-resolution arrays are produced inside the training scripts by downsampling
to `(80, 80, 96)`.

## Segmentations

`seg_o` contains multi-class organ-at-risk labels used for validation Dice.

`seg_b` contains bone labels. For MUSA loss training, the dataloader binarizes
this mask with `seg_b > 0`.

Both segmentation folders must use the same case IDs as `images`.

## Validation List Pairing

`myDataset_val` splits `val_list_inter.txt` in half and pairs corresponding
entries:

```text
moving001
moving002
fixed001
fixed002
```

This produces validation pairs:

```text
moving001 -> fixed001
moving002 -> fixed002
```
