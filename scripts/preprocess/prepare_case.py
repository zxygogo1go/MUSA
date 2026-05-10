"""Prepare one DIR-MUSA case from aligned NIfTI files.

This script assumes orientation standardization, background removal, and rigid
template alignment have already been handled upstream. It performs the final
repository-specific conversion to `.npy`: spacing resampling, center crop/pad,
CT intensity normalization, and label-preserving segmentation resampling.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


TARGET_SHAPE = (160, 160, 192)
TARGET_SPACING = (2.0, 2.0, 2.0)
CT_CLIP = (-1024.0, 3000.0)


def parse_tuple(value: str, length: int, cast):
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != length:
        raise argparse.ArgumentTypeError(f"Expected {length} comma-separated values, got {value!r}")
    try:
        return tuple(cast(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one DIR-MUSA case from aligned NIfTI files.")
    parser.add_argument("--case-id", required=True, help="Case ID used for output filenames.")
    parser.add_argument("--ct", required=True, help="Input CT image (.nii or .nii.gz).")
    parser.add_argument("--seg-o", required=True, help="Input OAR/organ segmentation (.nii or .nii.gz).")
    parser.add_argument("--seg-b", required=True, help="Input bone segmentation (.nii or .nii.gz).")
    parser.add_argument("--out-root", required=True, help="Output root containing images, seg_o, and seg_b.")
    parser.add_argument(
        "--target-shape",
        default=TARGET_SHAPE,
        type=lambda value: parse_tuple(value, 3, int),
        help="Output array shape as x,y,z. Default: 160,160,192.",
    )
    parser.add_argument(
        "--target-spacing",
        default=TARGET_SPACING,
        type=lambda value: parse_tuple(value, 3, float),
        help="Target spacing in mm as x,y,z. Default: 2,2,2.",
    )
    parser.add_argument(
        "--ct-clip",
        default=CT_CLIP,
        type=lambda value: parse_tuple(value, 2, float),
        help="CT clipping range as min,max. Default: -1024,3000.",
    )
    parser.add_argument(
        "--skip-shape-check",
        action="store_true",
        help="Allow outputs with a shape different from --target-shape.",
    )
    parser.add_argument(
        "--metadata-out",
        default=None,
        help="Optional metadata JSON path. Default: <out-root>/metadata/<case-id>.json.",
    )
    return parser.parse_args()


def load_nifti(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return data, spacing


def resample_to_spacing(
    array: np.ndarray,
    source_spacing: Sequence[float],
    target_spacing: Sequence[float],
    order: int,
) -> np.ndarray:
    factors = tuple(source / target for source, target in zip(source_spacing, target_spacing))
    return zoom(array, factors, order=order)


def center_crop_or_pad(array: np.ndarray, target_shape: Sequence[int], pad_value: float = 0) -> np.ndarray:
    output = np.full(tuple(target_shape), pad_value, dtype=array.dtype)

    src_slices = []
    dst_slices = []
    for src_size, dst_size in zip(array.shape, target_shape):
        if src_size >= dst_size:
            src_start = (src_size - dst_size) // 2
            src_end = src_start + dst_size
            dst_start = 0
            dst_end = dst_size
        else:
            src_start = 0
            src_end = src_size
            dst_start = (dst_size - src_size) // 2
            dst_end = dst_start + src_size

        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    output[tuple(dst_slices)] = array[tuple(src_slices)]
    return output


def normalize_ct(array: np.ndarray, clip_range: Sequence[float]) -> np.ndarray:
    clip_min, clip_max = clip_range
    if clip_max <= clip_min:
        raise ValueError(f"Invalid CT clip range: {clip_range}")
    clipped = np.clip(array.astype(np.float32), clip_min, clip_max)
    return ((clipped - clip_min) / (clip_max - clip_min)).astype(np.float32)


def prepare_image(
    path: Path,
    target_shape: Sequence[int],
    target_spacing: Sequence[float],
    clip_range: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, object]]:
    array, spacing = load_nifti(path)
    resampled = resample_to_spacing(array, spacing, target_spacing, order=1)
    cropped = center_crop_or_pad(resampled, target_shape, pad_value=clip_range[0])
    normalized = normalize_ct(cropped, clip_range)
    metadata = {
        "input_path": str(path),
        "input_shape": list(array.shape),
        "input_spacing": list(spacing),
        "resampled_shape": list(resampled.shape),
        "output_shape": list(normalized.shape),
        "interpolation": "linear",
    }
    return normalized, metadata


def prepare_segmentation(
    path: Path,
    target_shape: Sequence[int],
    target_spacing: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, object]]:
    array, spacing = load_nifti(path)
    resampled = resample_to_spacing(array, spacing, target_spacing, order=0)
    cropped = center_crop_or_pad(resampled, target_shape, pad_value=0)
    labels = np.rint(cropped).astype(np.int16)
    metadata = {
        "input_path": str(path),
        "input_shape": list(array.shape),
        "input_spacing": list(spacing),
        "resampled_shape": list(resampled.shape),
        "output_shape": list(labels.shape),
        "interpolation": "nearest",
        "labels": [int(value) for value in np.unique(labels)],
    }
    return labels, metadata


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def metadata_path(out_root: Path, case_id: str, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return out_root / "metadata" / f"{case_id}.json"


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    case_id = args.case_id
    target_shape = tuple(args.target_shape)
    target_spacing = tuple(args.target_spacing)
    ct_clip = tuple(args.ct_clip)

    image, image_meta = prepare_image(Path(args.ct), target_shape, target_spacing, ct_clip)
    seg_o, seg_o_meta = prepare_segmentation(Path(args.seg_o), target_shape, target_spacing)
    seg_b, seg_b_meta = prepare_segmentation(Path(args.seg_b), target_shape, target_spacing)

    for name, array in [("image", image), ("seg_o", seg_o), ("seg_b", seg_b)]:
        if not args.skip_shape_check and array.shape != target_shape:
            raise ValueError(f"{name} output shape {array.shape} does not match target shape {target_shape}")

    save_array(out_root / "images" / f"{case_id}.npy", image)
    save_array(out_root / "seg_o" / f"{case_id}.npy", seg_o)
    save_array(out_root / "seg_b" / f"{case_id}.npy", seg_b)

    meta = {
        "case_id": case_id,
        "target_shape": list(target_shape),
        "target_spacing": list(target_spacing),
        "ct_clip": list(ct_clip),
        "ct": image_meta,
        "seg_o": seg_o_meta,
        "seg_b": seg_b_meta,
    }
    meta_out = metadata_path(out_root, case_id, args.metadata_out)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {out_root / 'images' / f'{case_id}.npy'}")
    print(f"Wrote {out_root / 'seg_o' / f'{case_id}.npy'}")
    print(f"Wrote {out_root / 'seg_b' / f'{case_id}.npy'}")
    print(f"Wrote {meta_out}")


if __name__ == "__main__":
    main()
