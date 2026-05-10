"""Create DIR-MUSA training and validation list files from prepared arrays."""

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create DIR-MUSA list files.")
    parser.add_argument("--image-dir", required=True, help="Directory containing preprocessed .npy images.")
    parser.add_argument("--train-out", required=True, help="Output path for trn_list_inter.txt.")
    parser.add_argument("--val-out", required=True, help="Output path for val_list_inter.txt.")
    parser.add_argument(
        "--val-pairs",
        default=None,
        help="Optional CSV file with moving_id,fixed_id rows for validation.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fallback validation fraction when --val-pairs is omitted.",
    )
    parser.add_argument("--seed", type=int, default=2025, help="Seed for fallback train/validation split.")
    return parser.parse_args()


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def discover_case_ids(image_dir: Path) -> List[str]:
    case_ids = sorted(path.stem for path in image_dir.glob("*.npy"))
    if not case_ids:
        raise ValueError(f"No .npy files found in {image_dir}")
    return case_ids


def read_val_pairs(path: Path) -> List[Tuple[str, str]]:
    pairs = []
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.reader(file_obj)
        for row_idx, row in enumerate(reader, start=1):
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) != 2:
                raise ValueError(f"{path}:{row_idx} must have exactly two columns: moving_id,fixed_id")
            moving, fixed = row[0].strip(), row[1].strip()
            if not moving or not fixed:
                raise ValueError(f"{path}:{row_idx} contains an empty case id")
            pairs.append((moving, fixed))
    if not pairs:
        raise ValueError(f"No validation pairs found in {path}")
    return pairs


def fallback_split(case_ids: Sequence[str], val_fraction: float, seed: int) -> Tuple[List[str], List[Tuple[str, str]]]:
    if not 0 < val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    if len(case_ids) < 3:
        raise ValueError("At least three cases are needed for fallback train/validation splitting")

    import random

    rng = random.Random(seed)
    shuffled = list(case_ids)
    rng.shuffle(shuffled)

    val_count = max(2, round(len(shuffled) * val_fraction))
    val_ids = sorted(shuffled[:val_count])
    train_ids = sorted(shuffled[val_count:])
    if not train_ids:
        raise ValueError("Fallback split produced no training cases; reduce --val-fraction")

    fixed_ids = val_ids[1:] + val_ids[:1]
    val_pairs = list(zip(val_ids, fixed_ids))
    return train_ids, val_pairs


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    case_ids = discover_case_ids(image_dir)

    if args.val_pairs:
        val_pairs = read_val_pairs(Path(args.val_pairs))
        val_ids = {case_id for pair in val_pairs for case_id in pair}
        unknown = sorted(val_ids.difference(case_ids))
        if unknown:
            raise ValueError(f"Validation pairs reference case IDs not found in {image_dir}: {unknown}")
        train_ids = [case_id for case_id in case_ids if case_id not in val_ids]
        if not train_ids:
            raise ValueError("No training IDs remain after excluding validation pairs")
    else:
        train_ids, val_pairs = fallback_split(case_ids, args.val_fraction, args.seed)

    moving_ids = [moving for moving, _ in val_pairs]
    fixed_ids = [fixed for _, fixed in val_pairs]

    write_lines(Path(args.train_out), train_ids)
    write_lines(Path(args.val_out), moving_ids + fixed_ids)

    print(f"Wrote {len(train_ids)} training IDs to {args.train_out}")
    print(f"Wrote {len(val_pairs)} validation pairs to {args.val_out}")


if __name__ == "__main__":
    main()
