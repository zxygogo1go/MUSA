"""Prepare a batch of DIR-MUSA cases from a CSV manifest."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


REQUIRED_COLUMNS = ("case_id", "ct", "seg_o", "seg_b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare multiple DIR-MUSA cases from a CSV manifest.")
    parser.add_argument("--manifest", required=True, help="CSV with columns: case_id,ct,seg_o,seg_b.")
    parser.add_argument("--out-root", required=True, help="Output root containing images, seg_o, seg_b, and metadata.")
    parser.add_argument("--target-shape", default="160,160,192", help="Output array shape as x,y,z.")
    parser.add_argument("--target-spacing", default="2,2,2", help="Target spacing in mm as x,y,z.")
    parser.add_argument("--ct-clip", default="-1024,3000", help="CT clipping range as min,max.")
    parser.add_argument("--skip-shape-check", action="store_true", help="Forwarded to prepare_case.py.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue processing after a failed case.")
    return parser.parse_args()


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        rows = []
        for row_idx, row in enumerate(reader, start=2):
            cleaned = {key: (value or "").strip() for key, value in row.items()}
            missing_values = [column for column in REQUIRED_COLUMNS if not cleaned[column]]
            if missing_values:
                raise ValueError(f"{path}:{row_idx} has empty required values: {missing_values}")
            rows.append(cleaned)

    if not rows:
        raise ValueError(f"No cases found in {path}")
    return rows


def build_command(args: argparse.Namespace, row: Dict[str, str]) -> List[str]:
    script = Path(__file__).with_name("prepare_case.py")
    command = [
        sys.executable,
        str(script),
        "--case-id",
        row["case_id"],
        "--ct",
        row["ct"],
        "--seg-o",
        row["seg_o"],
        "--seg-b",
        row["seg_b"],
        "--out-root",
        args.out_root,
        "--target-shape",
        args.target_shape,
        "--target-spacing",
        args.target_spacing,
        "--ct-clip",
        args.ct_clip,
    ]
    if args.skip_shape_check:
        command.append("--skip-shape-check")
    return command


def main() -> None:
    args = parse_args()
    rows = read_manifest(Path(args.manifest))

    failures = []
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] Preparing {row['case_id']}")
        result = subprocess.run(build_command(args, row), check=False)
        if result.returncode != 0:
            failures.append(row["case_id"])
            if not args.continue_on_error:
                raise SystemExit(result.returncode)

    if failures:
        raise SystemExit(f"Failed cases: {', '.join(failures)}")

    print(f"Prepared {len(rows)} cases into {args.out_root}")


if __name__ == "__main__":
    main()
