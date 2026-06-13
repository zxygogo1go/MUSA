"""Prepare multiple HaN-Seg cases and optionally write a fixed paper split."""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from prepare_hanseg_case import hanseg_case_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare multiple HaN-Seg case directories.")
    parser.add_argument("--cases-root", required=True, help="Directory containing case_* folders, e.g. HaN-Seg/set_1.")
    parser.add_argument("--out-root", required=True, help="Output root containing images, seg_o, seg_b, metadata.")
    parser.add_argument("--case-glob", default="case_*", help="Glob for case directories. Default: case_*.")
    parser.add_argument("--case-id-prefix", default="hanseg", help="Prefix used when deriving case IDs.")
    parser.add_argument("--image-name", default=None, help="Forwarded CT filename. Default: *_IMG_CT.nrrd.")
    parser.add_argument("--structure-names", default=None, help="Forwarded comma-separated global structure names.")
    parser.add_argument("--bone-structures", default=None, help="Forwarded comma-separated bone structure names.")
    parser.add_argument("--include-bone-in-seg-o", action="store_true", help="Forwarded to prepare_hanseg_case.py.")
    parser.add_argument("--strict-structures", action="store_true", help="Forwarded to prepare_hanseg_case.py.")
    parser.add_argument("--skip-missing-bone", action="store_true", help="Forwarded to prepare_hanseg_case.py.")
    parser.add_argument("--target-shape", default="160,160,192", help="Output array shape as x,y,z.")
    parser.add_argument("--target-spacing", default="2,2,2", help="Target spacing in mm as x,y,z.")
    parser.add_argument("--ct-clip", default="-1024,3000", help="CT clipping range as min,max.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a case when all three output arrays exist.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue processing after a failed case.")
    parser.add_argument(
        "--write-paper-split",
        action="store_true",
        help="Write data_hanseg/lists/paper_split-style train/val/test files under --out-root.",
    )
    return parser.parse_args()


def discover_case_dirs(cases_root: Path, case_glob: str) -> List[Path]:
    case_dirs = sorted(path for path in cases_root.glob(case_glob) if path.is_dir())
    if not case_dirs:
        raise ValueError(f"No case directories found in {cases_root} with glob {case_glob!r}")
    return case_dirs


def case_outputs_exist(out_root: Path, case_id: str) -> bool:
    return all(
        (out_root / folder / f"{case_id}.npy").is_file()
        for folder in ("images", "seg_o", "seg_b")
    )


def build_command(args: argparse.Namespace, case_dir: Path) -> List[str]:
    script = Path(__file__).with_name("prepare_hanseg_case.py")
    command = [
        sys.executable,
        str(script),
        "--case-dir",
        str(case_dir),
        "--case-id-prefix",
        args.case_id_prefix,
        "--out-root",
        args.out_root,
        "--target-shape",
        args.target_shape,
        "--target-spacing",
        args.target_spacing,
        f"--ct-clip={args.ct_clip}",
    ]
    if args.image_name:
        command.extend(["--image-name", args.image_name])
    if args.structure_names:
        command.extend(["--structure-names", args.structure_names])
    if args.bone_structures:
        command.extend(["--bone-structures", args.bone_structures])
    if args.include_bone_in_seg_o:
        command.append("--include-bone-in-seg-o")
    if args.strict_structures:
        command.append("--strict-structures")
    if args.skip_missing_bone:
        command.append("--skip-missing-bone")
    return command


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def write_pairs(path: Path, pairs: Sequence[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerows(pairs)


def default_pair(ids_by_number: dict, numbers: Sequence[int]) -> Tuple[str, str]:
    moving, fixed = numbers
    try:
        return ids_by_number[moving], ids_by_number[fixed]
    except KeyError as exc:
        raise ValueError(f"Cannot write paper split; missing case number {exc.args[0]}") from exc


def case_number(case_id: str) -> int:
    match = re.search(r"(\d+)$", case_id)
    if not match:
        raise ValueError(f"Could not infer numeric case number from {case_id}")
    return int(match.group(1))


def write_paper_split(out_root: Path, case_ids: Sequence[str]) -> None:
    ids_by_number = {case_number(case_id): case_id for case_id in case_ids}
    test_pairs = [
        default_pair(ids_by_number, (1, 2)),
        default_pair(ids_by_number, (3, 4)),
        default_pair(ids_by_number, (5, 6)),
        default_pair(ids_by_number, (7, 8)),
        default_pair(ids_by_number, (9, 10)),
    ]
    val_pairs = [
        default_pair(ids_by_number, (11, 12)),
        default_pair(ids_by_number, (13, 14)),
        default_pair(ids_by_number, (15, 16)),
        default_pair(ids_by_number, (17, 18)),
        default_pair(ids_by_number, (20, 21)),
    ]
    held_out = {case_id for pair in test_pairs + val_pairs for case_id in pair}
    train_ids = [case_id for case_id in sorted(case_ids, key=case_number) if case_id not in held_out]
    if not train_ids:
        raise ValueError("Paper split produced no training cases")

    split_dir = out_root / "lists" / "paper_split"
    write_lines(split_dir / "trn_list_inter.txt", train_ids)
    write_lines(split_dir / "val_list_inter.txt", [moving for moving, _ in val_pairs] + [fixed for _, fixed in val_pairs])
    write_pairs(split_dir / "val_pairs.csv", val_pairs)
    write_pairs(split_dir / "test_pairs.csv", test_pairs)
    (split_dir / "README.md").write_text(
        "\n".join(
            [
                "# HaN-Seg Paper Split",
                "",
                "Fixed inter-subject CT registration split for the 42-case HaN-Seg set.",
                "",
                "- Test pairs: cases 01-10 as five adjacent moving/fixed pairs.",
                "- Validation pairs: cases 11-18 and 20-21 as five adjacent moving/fixed pairs.",
                "- Training IDs: all remaining cases, including case 19.",
                "",
                "case_19 is kept out of validation/test because the public HaN-Seg",
                "availability table and files omit `OAR_OpticChiasm`; keeping it in",
                "training avoids missing-label metrics while preserving all 42 cases.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote HaN-Seg paper split to {split_dir}")


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    case_dirs = discover_case_dirs(Path(args.cases_root), args.case_glob)

    failures = []
    prepared_case_ids: List[str] = []
    for index, case_dir in enumerate(case_dirs, start=1):
        case_id = hanseg_case_id(case_dir.name, args.case_id_prefix)
        prepared_case_ids.append(case_id)
        if args.skip_existing and case_outputs_exist(out_root, case_id):
            print(f"[{index}/{len(case_dirs)}] Skipping {case_dir.name} -> {case_id} (exists)")
            continue
        print(f"[{index}/{len(case_dirs)}] Preparing {case_dir.name} -> {case_id}")
        result = subprocess.run(build_command(args, case_dir), check=False)
        if result.returncode != 0:
            failures.append(case_dir.name)
            if not args.continue_on_error:
                raise SystemExit(result.returncode)

    if failures:
        raise SystemExit(f"Failed cases: {', '.join(failures)}")

    if args.write_paper_split:
        write_paper_split(out_root, prepared_case_ids)
    print(f"Prepared {len(case_dirs)} HaN-Seg cases into {out_root}")


if __name__ == "__main__":
    main()
