"""Prepare multiple SegRap-style case directories for DIR-MUSA training."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare multiple SegRap-style case directories.")
    parser.add_argument("--cases-root", required=True, help="Directory containing segrap_* case folders.")
    parser.add_argument("--out-root", required=True, help="Output root containing images, seg_o, seg_b, and metadata.")
    parser.add_argument("--case-glob", default="segrap_*", help="Glob for case directories. Default: segrap_*.")
    parser.add_argument("--image-name", default="image.nii.gz", help="CT filename inside each case directory.")
    parser.add_argument("--bone-structures", default=None, help="Forwarded comma-separated bone structure names.")
    parser.add_argument("--exclude-structures", default=None, help="Forwarded comma-separated excluded structure names.")
    parser.add_argument("--target-shape", default="160,160,192", help="Output array shape as x,y,z.")
    parser.add_argument("--target-spacing", default="2,2,2", help="Target spacing in mm as x,y,z.")
    parser.add_argument("--ct-clip", default="-1024,3000", help="CT clipping range as min,max.")
    parser.add_argument("--skip-missing-bone", action="store_true", help="Forwarded to prepare_segrap_case.py.")
    parser.add_argument("--include-bone-in-seg-o", action="store_true", help="Forwarded to prepare_segrap_case.py.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue processing after a failed case.")
    return parser.parse_args()


def discover_case_dirs(cases_root: Path, case_glob: str) -> List[Path]:
    case_dirs = sorted(path for path in cases_root.glob(case_glob) if path.is_dir())
    if not case_dirs:
        raise ValueError(f"No case directories found in {cases_root} with glob {case_glob!r}")
    return case_dirs


def build_command(args: argparse.Namespace, case_dir: Path) -> List[str]:
    script = Path(__file__).with_name("prepare_segrap_case.py")
    command = [
        sys.executable,
        str(script),
        "--case-dir",
        str(case_dir),
        "--case-id",
        case_dir.name,
        "--out-root",
        args.out_root,
        "--image-name",
        args.image_name,
        "--target-shape",
        args.target_shape,
        "--target-spacing",
        args.target_spacing,
        f"--ct-clip={args.ct_clip}",
    ]
    if args.bone_structures:
        command.extend(["--bone-structures", args.bone_structures])
    if args.exclude_structures:
        command.extend(["--exclude-structures", args.exclude_structures])
    if args.skip_missing_bone:
        command.append("--skip-missing-bone")
    if args.include_bone_in_seg_o:
        command.append("--include-bone-in-seg-o")
    return command


def main() -> None:
    args = parse_args()
    case_dirs = discover_case_dirs(Path(args.cases_root), args.case_glob)

    failures = []
    for index, case_dir in enumerate(case_dirs, start=1):
        print(f"[{index}/{len(case_dirs)}] Preparing {case_dir.name}")
        result = subprocess.run(build_command(args, case_dir), check=False)
        if result.returncode != 0:
            failures.append(case_dir.name)
            if not args.continue_on_error:
                raise SystemExit(result.returncode)

    if failures:
        raise SystemExit(f"Failed cases: {', '.join(failures)}")

    print(f"Prepared {len(case_dirs)} SegRap cases into {args.out_root}")


if __name__ == "__main__":
    main()
