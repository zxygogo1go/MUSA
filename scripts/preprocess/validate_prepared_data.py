"""Validate prepared DIR-MUSA `.npy` data before training."""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def parse_tuple(value: str, length: int, cast):
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != length:
        raise argparse.ArgumentTypeError(f"Expected {length} comma-separated values, got {value!r}")
    try:
        return tuple(cast(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prepared DIR-MUSA data folders.")
    parser.add_argument("--data-root", required=True, help="Root containing images, seg_o, and seg_b folders.")
    parser.add_argument(
        "--expected-shape",
        default=(160, 160, 192),
        type=lambda value: parse_tuple(value, 3, int),
        help="Expected array shape as x,y,z. Default: 160,160,192.",
    )
    parser.add_argument("--max-cases", type=int, default=None, help="Validate only the first N sorted cases.")
    parser.add_argument("--allow-empty-seg-o", action="store_true", help="Do not fail when seg_o is all background.")
    parser.add_argument("--allow-empty-seg-b", action="store_true", help="Do not fail when seg_b is all background.")
    return parser.parse_args()


def npy_ids(folder: Path) -> List[str]:
    return sorted(path.stem for path in folder.glob("*.npy"))


def require_same_ids(ids_by_folder: Dict[str, List[str]]) -> List[str]:
    id_sets = {name: set(ids) for name, ids in ids_by_folder.items()}
    common = set.intersection(*id_sets.values())
    for name, ids in id_sets.items():
        missing = sorted(common.symmetric_difference(ids))
        if missing:
            extras = sorted(ids.difference(common))
            absent = sorted(common.difference(ids))
            raise ValueError(f"{name} IDs do not match. extras={extras[:10]}, missing={absent[:10]}")
    return sorted(common)


def is_integer_array(array: np.ndarray) -> bool:
    if np.issubdtype(array.dtype, np.integer):
        return True
    return bool(np.allclose(array, np.rint(array)))


def validate_case(data_root: Path, case_id: str, expected_shape: Sequence[int], args: argparse.Namespace) -> List[str]:
    warnings = []
    image = np.load(data_root / "images" / f"{case_id}.npy", mmap_mode="r")
    seg_o = np.load(data_root / "seg_o" / f"{case_id}.npy", mmap_mode="r")
    seg_b = np.load(data_root / "seg_b" / f"{case_id}.npy", mmap_mode="r")

    for name, array in [("image", image), ("seg_o", seg_o), ("seg_b", seg_b)]:
        if tuple(array.shape) != tuple(expected_shape):
            raise ValueError(f"{case_id}: {name} shape {array.shape} != {tuple(expected_shape)}")

    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError(f"{case_id}: image dtype should be float, got {image.dtype}")
    image_min = float(np.min(image))
    image_max = float(np.max(image))
    if image_min < -1e-5 or image_max > 1.0 + 1e-5:
        raise ValueError(f"{case_id}: image values should be in [0, 1], got [{image_min}, {image_max}]")

    if not is_integer_array(seg_o):
        raise ValueError(f"{case_id}: seg_o should contain integer labels, got dtype {seg_o.dtype}")
    if not is_integer_array(seg_b):
        raise ValueError(f"{case_id}: seg_b should contain integer labels, got dtype {seg_b.dtype}")

    seg_o_unique = np.unique(seg_o)
    seg_b_unique = np.unique(seg_b)
    if len(seg_o_unique) <= 1 and not args.allow_empty_seg_o:
        raise ValueError(f"{case_id}: seg_o is empty or all background")
    if len(seg_b_unique) <= 1 and not args.allow_empty_seg_b:
        raise ValueError(f"{case_id}: seg_b is empty or all background")
    if not set(int(value) for value in seg_b_unique).issubset({0, 1}):
        raise ValueError(f"{case_id}: seg_b should be binary, got labels {seg_b_unique.tolist()}")

    if image_min == image_max:
        warnings.append(f"{case_id}: image has constant value {image_min}")
    return warnings


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    folders = {
        "images": data_root / "images",
        "seg_o": data_root / "seg_o",
        "seg_b": data_root / "seg_b",
    }
    for name, folder in folders.items():
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder}")

    ids_by_folder = {name: npy_ids(folder) for name, folder in folders.items()}
    case_ids = require_same_ids(ids_by_folder)
    if args.max_cases is not None:
        case_ids = case_ids[: args.max_cases]
    if not case_ids:
        raise ValueError(f"No cases found under {data_root}")

    warnings = []
    for index, case_id in enumerate(case_ids, start=1):
        warnings.extend(validate_case(data_root, case_id, args.expected_shape, args))
        if index % 10 == 0 or index == len(case_ids):
            print(f"Validated {index}/{len(case_ids)} cases")

    print(f"OK: validated {len(case_ids)} cases in {data_root}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
