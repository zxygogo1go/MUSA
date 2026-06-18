"""Batch-create MUSA Stage2 vs MUSA+ Stage3 visual comparisons.

This wrapper runs single-pair MUSA+ inference to save warped CT/segmentation
arrays, then calls `visualize_musa_plus_pair.py` to render PNG diagnostics.
It is intended for paper figures where Stage2 is the MUSA baseline and Stage3
is the proposed adaptive refinement.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Stage2-vs-Stage3 visual comparisons for selected pairs.")
    parser.add_argument("--data-root", required=True, help="Prepared data root containing images, seg_o, seg_b.")
    parser.add_argument("--checkpoint-stage1", required=True, help="Stage1 r2 checkpoint.")
    parser.add_argument("--checkpoint-stage2", required=True, help="Stage2 r1 checkpoint.")
    parser.add_argument("--checkpoint-stage3", required=True, help="MUSA+ Stage3 checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Output directory for pair arrays, figures, and index.md.")
    parser.add_argument("--model-type", default="05dualprnet-v1", help="Registration model type.")
    parser.add_argument("--metadata-path", default=None, help="Metadata path for resolving small-OAR labels.")
    parser.add_argument("--small-oar-labels", default=None, help="Comma-separated explicit small-OAR labels.")
    parser.add_argument("--small-oar-names", default=None, help="Comma-separated small-OAR names.")
    parser.add_argument("--pairs-csv", default=None, help="Explicit moving_id,fixed_id CSV. Used before --compare-csv.")
    parser.add_argument(
        "--compare-csv",
        default=None,
        help="stage2_vs_stage3_by_pair.csv used to select representative pairs.",
    )
    parser.add_argument(
        "--selection",
        default="top-small,worst-large,worst-jac",
        help="Comma-separated selectors: top-small,worst-large,worst-jac,first,all.",
    )
    parser.add_argument("--num-pairs", type=int, default=3, help="Maximum number of selected pairs.")
    parser.add_argument("--top-k", type=int, default=8, help="Labels per local montage.")
    parser.add_argument("--margin", type=int, default=10, help="Voxel crop margin for local montages.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution.")
    parser.add_argument("--gpu", default="0", help="CUDA visible GPU IDs.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip inference when this pair's metrics JSON already exists; still redraw figures.",
    )
    return parser.parse_args()


def read_pairs(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.reader(file_obj)
        for row_idx, row in enumerate(reader, start=1):
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) != 2:
                raise ValueError(f"{path}:{row_idx} must contain moving_id,fixed_id")
            pairs.append((row[0].strip(), row[1].strip()))
    if not pairs:
        raise ValueError(f"No pairs found in {path}")
    return pairs


def read_compare_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def dedupe_pairs(pairs: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    output: List[Tuple[str, str]] = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        output.append(pair)
    return output


def select_pairs_from_compare(rows: Sequence[Dict[str, str]], selection: str, num_pairs: int) -> List[Tuple[str, str]]:
    selectors = [item.strip() for item in selection.split(",") if item.strip()]
    selected: List[Tuple[str, str]] = []
    for selector in selectors:
        if selector == "all":
            ordered = list(rows)
        elif selector == "first":
            ordered = list(rows)
        elif selector == "top-small":
            ordered = sorted(rows, key=lambda row: as_float(row, "small_oar_delta"), reverse=True)
        elif selector == "worst-large":
            ordered = sorted(rows, key=lambda row: as_float(row, "large_oar_worst_delta"))
        elif selector == "worst-jac":
            ordered = sorted(rows, key=lambda row: as_float(row, "stage3_jac_roi_nonpos"), reverse=True)
        else:
            raise ValueError(f"Unsupported selector {selector!r}")
        for row in ordered:
            selected.append((row["moving_id"], row["fixed_id"]))
            break

    pairs = dedupe_pairs(selected)
    if len(pairs) < num_pairs and "all" not in selectors:
        for row in rows:
            pairs.append((row["moving_id"], row["fixed_id"]))
            pairs = dedupe_pairs(pairs)
            if len(pairs) >= num_pairs:
                break
    return pairs[:num_pairs] if num_pairs > 0 else pairs


def pair_prefix(moving_id: str, fixed_id: str) -> str:
    return f"{moving_id}_to_{fixed_id}"


def add_optional(command: List[str], option: str, value: Optional[str]) -> None:
    if value:
        command.extend([option, value])


def run_command(command: Sequence[str]) -> None:
    print("[CMD]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def run_inference(args: argparse.Namespace, pair_dir: Path, moving_id: str, fixed_id: str) -> None:
    prefix = pair_prefix(moving_id, fixed_id)
    metrics_path = pair_dir / f"{prefix}_musa_plus_metrics.json"
    if args.skip_existing and metrics_path.is_file():
        print(f"[INFO] Reusing existing pair outputs: {pair_dir}")
        return

    command = [
        sys.executable,
        str(SCRIPT_DIR / "infer_musa_plus_prepared_pair.py"),
        "--moving-id",
        moving_id,
        "--fixed-id",
        fixed_id,
        "--data-root",
        args.data_root,
        "--model-type",
        args.model_type,
        "--checkpoint-stage1",
        args.checkpoint_stage1,
        "--checkpoint-stage2",
        args.checkpoint_stage2,
        "--checkpoint-stage3",
        args.checkpoint_stage3,
        "--output-dir",
        str(pair_dir),
        "--output-prefix",
        prefix,
        "--gpu",
        args.gpu,
    ]
    add_optional(command, "--metadata-path", args.metadata_path)
    add_optional(command, "--small-oar-labels", args.small_oar_labels)
    add_optional(command, "--small-oar-names", args.small_oar_names)
    if args.cpu:
        command.append("--cpu")
    run_command(command)


def run_small_oar_visualization(args: argparse.Namespace, pair_dir: Path, moving_id: str, fixed_id: str) -> Path:
    output_dir = pair_dir / "viz_musa_vs_ours"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "visualize_musa_plus_pair.py"),
        "--pair-dir",
        str(pair_dir),
        "--data-root",
        args.data_root,
        "--moving-id",
        moving_id,
        "--fixed-id",
        fixed_id,
        "--output-dir",
        str(output_dir),
        "--top-k",
        str(args.top_k),
        "--margin",
        str(args.margin),
        "--dpi",
        str(args.dpi),
    ]
    add_optional(command, "--small-oar-labels", args.small_oar_labels)
    add_optional(command, "--small-oar-names", args.small_oar_names)
    run_command(command)
    return output_dir


def run_headneck_motion_visualization(args: argparse.Namespace, pair_dir: Path, moving_id: str, fixed_id: str) -> Path:
    output_dir = pair_dir / "viz_headneck_motion"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "visualize_musa_plus_headneck_motion.py"),
        "--pair-dir",
        str(pair_dir),
        "--data-root",
        args.data_root,
        "--moving-id",
        moving_id,
        "--fixed-id",
        fixed_id,
        "--output-dir",
        str(output_dir),
        "--dpi",
        str(args.dpi),
    ]
    run_command(command)
    return output_dir


def write_index(output_dir: Path, pair_viz_dirs: Sequence[Tuple[str, str, Path, Path]]) -> None:
    lines = [
        "# MUSA Stage2 vs Proposed Stage3 Visual Comparison",
        "",
        "Color convention in overlay figures:",
        "",
        "- fixed label: green",
        "- original moving label: red",
        "- MUSA Stage2 warped label: cyan",
        "- proposed Stage3 warped label: yellow",
        "",
    ]
    small_figure_names = [
        "01_small_oar_stage2_vs_stage3_overlay.png",
        "02_small_oar_ct_difference.png",
        "03_top_small_oar_gains.png",
        "04_small_oar_worst_deltas.png",
    ]
    motion_figure_names = [
        "01_global_ct_motion_stage1_stage2_stage3.png",
        "02_large_structure_motion_contours.png",
        "03_dvf_magnitude_stage2_stage3_residual.png",
        "04_deformation_grid_and_vectors.png",
    ]
    for moving_id, fixed_id, small_viz_dir, motion_viz_dir in pair_viz_dirs:
        prefix = pair_prefix(moving_id, fixed_id)
        small_rel_dir = small_viz_dir.relative_to(output_dir)
        motion_rel_dir = motion_viz_dir.relative_to(output_dir)
        lines.extend([f"## {prefix}", ""])
        lines.extend(["### Small-OAR Alignment", ""])
        for name in small_figure_names:
            lines.append(f"![{prefix} {name}]({small_rel_dir / name})")
            lines.append("")
        lines.extend(["### Head-Neck Large-Motion Diagnostics", ""])
        for name in motion_figure_names:
            lines.append(f"![{prefix} {name}]({motion_rel_dir / name})")
            lines.append("")
    (output_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs_csv:
        pairs = read_pairs(Path(args.pairs_csv))
        pairs = pairs[: args.num_pairs] if args.num_pairs > 0 else pairs
    elif args.compare_csv:
        rows = read_compare_rows(Path(args.compare_csv))
        pairs = select_pairs_from_compare(rows, args.selection, args.num_pairs)
    else:
        raise ValueError("Provide either --pairs-csv or --compare-csv")

    pair_viz_dirs: List[Tuple[str, str, Path, Path]] = []
    for index, (moving_id, fixed_id) in enumerate(pairs, start=1):
        prefix = pair_prefix(moving_id, fixed_id)
        pair_dir = output_dir / prefix
        pair_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(pairs)}] {moving_id} -> {fixed_id}", flush=True)
        run_inference(args, pair_dir, moving_id, fixed_id)
        small_viz_dir = run_small_oar_visualization(args, pair_dir, moving_id, fixed_id)
        motion_viz_dir = run_headneck_motion_visualization(args, pair_dir, moving_id, fixed_id)
        pair_viz_dirs.append((moving_id, fixed_id, small_viz_dir, motion_viz_dir))

    write_index(output_dir, pair_viz_dirs)
    print(f"[INFO] Wrote visual comparison index: {output_dir / 'index.md'}")


if __name__ == "__main__":
    main()
