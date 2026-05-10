"""Prepare one SegRap-style case directory for DIR-MUSA training.

Expected input layout:

    segrap_0000/
      image.nii.gz
      Brain.nii.gz
      BrainStem.nii.gz
      ...

The script merges per-structure binary masks into a multi-label `seg_o` array
and a binary `seg_b` bone mask, then applies the same final preprocessing used
by `prepare_case.py`.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import nibabel as nib
import numpy as np

from prepare_case import (
    CT_CLIP,
    TARGET_SHAPE,
    TARGET_SPACING,
    center_crop_or_pad,
    load_nifti,
    normalize_ct,
    parse_tuple,
    resample_to_spacing,
    save_array,
)


DEFAULT_EXCLUDE = {"image", "image_contrast"}
DEFAULT_BONE_STRUCTURES = {
    "ETbone_L",
    "ETbone_R",
    "Mandible_L",
    "Mandible_R",
    "Mastoid_L",
    "Mastoid_R",
    "TMjoint_L",
    "TMjoint_R",
}


def parse_name_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one SegRap-style case directory for DIR-MUSA.")
    parser.add_argument("--case-dir", required=True, help="Directory containing image.nii.gz and per-structure masks.")
    parser.add_argument("--case-id", default=None, help="Output case ID. Default: case directory name.")
    parser.add_argument("--out-root", required=True, help="Output root containing images, seg_o, seg_b, and metadata.")
    parser.add_argument(
        "--image-name",
        default="image.nii.gz",
        help="CT filename inside --case-dir. Use image_contrast.nii.gz if desired.",
    )
    parser.add_argument(
        "--bone-structures",
        default=",".join(sorted(DEFAULT_BONE_STRUCTURES)),
        help="Comma-separated structure names used to build binary seg_b.",
    )
    parser.add_argument(
        "--exclude-structures",
        default=",".join(sorted(DEFAULT_EXCLUDE)),
        help="Comma-separated stem names excluded from multi-label seg_o.",
    )
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
        "--skip-missing-bone",
        action="store_true",
        help="Continue when a listed bone structure is absent.",
    )
    parser.add_argument(
        "--include-bone-in-seg-o",
        action="store_true",
        help="Also include bone structures in the multi-label seg_o output.",
    )
    return parser.parse_args()


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def discover_structure_paths(case_dir: Path, image_name: str, excluded: Iterable[str]) -> Dict[str, Path]:
    excluded_names = set(excluded)
    excluded_names.add(nifti_stem(Path(image_name)))

    paths = {}
    for path in sorted(case_dir.glob("*.nii*")):
        stem = nifti_stem(path)
        if stem in excluded_names:
            continue
        paths[stem] = path
    if not paths:
        raise ValueError(f"No structure masks found in {case_dir}")
    return paths


def check_mask_geometry(mask: np.ndarray, reference_shape: Sequence[int], name: str) -> None:
    if tuple(mask.shape) != tuple(reference_shape):
        raise ValueError(f"{name} shape {mask.shape} does not match image shape {tuple(reference_shape)}")


def merge_oar_masks(structure_paths: Dict[str, Path], reference_shape: Sequence[int]) -> Tuple[np.ndarray, Dict[str, int]]:
    merged = np.zeros(tuple(reference_shape), dtype=np.int16)
    labels = {}

    for label, name in enumerate(sorted(structure_paths), start=1):
        mask, _ = load_nifti(structure_paths[name])
        check_mask_geometry(mask, reference_shape, name)
        mask_bool = mask > 0
        if not np.any(mask_bool):
            continue
        merged[mask_bool] = label
        labels[name] = label

    if not labels:
        raise ValueError("All structure masks are empty")
    return merged, labels


def merge_bone_masks(
    structure_paths: Dict[str, Path],
    bone_names: Sequence[str],
    reference_shape: Sequence[int],
    skip_missing: bool,
) -> Tuple[np.ndarray, List[str]]:
    missing = [name for name in bone_names if name not in structure_paths]
    if missing and not skip_missing:
        raise ValueError(f"Missing bone structures: {missing}")

    merged = np.zeros(tuple(reference_shape), dtype=np.int16)
    used = []
    for name in bone_names:
        path = structure_paths.get(name)
        if path is None:
            continue
        mask, _ = load_nifti(path)
        check_mask_geometry(mask, reference_shape, name)
        mask_bool = mask > 0
        if np.any(mask_bool):
            merged[mask_bool] = 1
            used.append(name)

    if not used:
        raise ValueError("No non-empty bone structures were found for seg_b")
    return merged, used


def prepare_array(
    array: np.ndarray,
    source_spacing: Sequence[float],
    target_shape: Sequence[int],
    target_spacing: Sequence[float],
    interpolation_order: int,
    pad_value: float,
) -> np.ndarray:
    resampled = resample_to_spacing(array, source_spacing, target_spacing, order=interpolation_order)
    return center_crop_or_pad(resampled, target_shape, pad_value=pad_value)


def main() -> None:
    args = parse_args()
    case_dir = Path(args.case_dir)
    case_id = args.case_id or case_dir.name
    out_root = Path(args.out_root)
    image_path = case_dir / args.image_name
    if not image_path.is_file():
        raise FileNotFoundError(f"CT image not found: {image_path}")

    image_raw, spacing = load_nifti(image_path)
    structure_paths = discover_structure_paths(
        case_dir,
        image_name=args.image_name,
        excluded=parse_name_list(args.exclude_structures),
    )
    bone_names = parse_name_list(args.bone_structures)

    seg_b_raw, used_bone_names = merge_bone_masks(
        structure_paths,
        bone_names=bone_names,
        reference_shape=image_raw.shape,
        skip_missing=args.skip_missing_bone,
    )
    if args.include_bone_in_seg_o:
        seg_o_paths = structure_paths
    else:
        seg_o_paths = {name: path for name, path in structure_paths.items() if name not in set(bone_names)}
    seg_o_raw, label_map = merge_oar_masks(seg_o_paths, image_raw.shape)

    target_shape = tuple(args.target_shape)
    target_spacing = tuple(args.target_spacing)
    ct_clip = tuple(args.ct_clip)

    image_prepared = prepare_array(image_raw, spacing, target_shape, target_spacing, 1, ct_clip[0])
    image_prepared = normalize_ct(image_prepared, ct_clip)
    seg_o_prepared = prepare_array(seg_o_raw, spacing, target_shape, target_spacing, 0, 0)
    seg_b_prepared = prepare_array(seg_b_raw, spacing, target_shape, target_spacing, 0, 0)

    seg_o_prepared = np.rint(seg_o_prepared).astype(np.int16)
    seg_b_prepared = (seg_b_prepared > 0).astype(np.int16)

    save_array(out_root / "images" / f"{case_id}.npy", image_prepared.astype(np.float32))
    save_array(out_root / "seg_o" / f"{case_id}.npy", seg_o_prepared)
    save_array(out_root / "seg_b" / f"{case_id}.npy", seg_b_prepared)

    meta = {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "image_path": str(image_path),
        "input_shape": list(image_raw.shape),
        "input_spacing": list(spacing),
        "target_shape": list(target_shape),
        "target_spacing": list(target_spacing),
        "ct_clip": list(ct_clip),
        "label_map": label_map,
        "include_bone_in_seg_o": args.include_bone_in_seg_o,
        "bone_structures_requested": bone_names,
        "bone_structures_used": used_bone_names,
        "seg_o_labels_after_preprocess": [int(value) for value in np.unique(seg_o_prepared)],
        "seg_b_labels_after_preprocess": [int(value) for value in np.unique(seg_b_prepared)],
    }
    meta_path = out_root / "metadata" / f"{case_id}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {out_root / 'images' / f'{case_id}.npy'}")
    print(f"Wrote {out_root / 'seg_o' / f'{case_id}.npy'}")
    print(f"Wrote {out_root / 'seg_b' / f'{case_id}.npy'}")
    print(f"Wrote {meta_path}")
    print(f"Merged {len(label_map)} OAR structures; used {len(used_bone_names)} bone structures.")


if __name__ == "__main__":
    main()
