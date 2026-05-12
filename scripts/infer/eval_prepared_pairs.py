"""Evaluate DIR-MUSA inference over multiple prepared `.npy` pairs."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import musa
from infer_prepared_pair import (
    RESOLUTION_SHAPES,
    dice_for_labels,
    load_array,
    load_model,
    npy_path,
    run_two_stage,
    save_outputs,
    warp_segmentation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DIR-MUSA over multiple prepared .npy pairs.")
    parser.add_argument("--pairs-csv", required=True, help="CSV with moving_id,fixed_id rows.")
    parser.add_argument("--data-root", default="data", help="Prepared data root containing images, seg_o, and seg_b.")
    parser.add_argument("--model-type", default="01voxelmorph-v1", help="Model type used for both stages.")
    parser.add_argument("--checkpoint-stage1", required=True, help="Stage 1 r2 checkpoint path.")
    parser.add_argument("--checkpoint-stage2", required=True, help="Stage 2 r1 checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for summary files and optional pair outputs.")
    parser.add_argument("--save-pair-outputs", action="store_true", help="Save warped images/DVFs/segmentations per pair.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Evaluate only the first N pairs.")
    parser.add_argument("--gpu", default="0", help="CUDA visible GPU IDs.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    return parser.parse_args()


def read_pairs(path: Path) -> List[Tuple[str, str]]:
    pairs = []
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.reader(file_obj)
        for row_idx, row in enumerate(reader, start=1):
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) != 2:
                raise ValueError(f"{path}:{row_idx} must contain exactly moving_id,fixed_id")
            moving_id, fixed_id = row[0].strip(), row[1].strip()
            if not moving_id or not fixed_id:
                raise ValueError(f"{path}:{row_idx} contains an empty case ID")
            pairs.append((moving_id, fixed_id))
    if not pairs:
        raise ValueError(f"No pairs found in {path}")
    return pairs


def metric_value(metrics: Dict[str, object], key: str, field: str) -> float:
    if key not in metrics:
        return float("nan")
    return float(metrics[key][field])


def evaluate_pair(
    moving_id: str,
    fixed_id: str,
    data_root: Path,
    model_type: str,
    model_stage1: torch.nn.Module,
    model_stage2: torch.nn.Module,
    spatial_transformer_r1: torch.nn.Module,
    composer_r1: torch.nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    moving_np = load_array(npy_path(data_root, "images", moving_id), RESOLUTION_SHAPES["r1"]).astype(np.float32)
    fixed_np = load_array(npy_path(data_root, "images", fixed_id), RESOLUTION_SHAPES["r1"]).astype(np.float32)
    moving_t = musa.utils_basics.numpy2torch(moving_np, device=device, CHECK=True)
    fixed_t = musa.utils_basics.numpy2torch(fixed_np, device=device, CHECK=True)

    with torch.no_grad():
        deformed_t, dvf_t, stage1_deformed_t = run_two_stage(
            moving_t=moving_t,
            fixed_t=fixed_t,
            model_stage1=model_stage1,
            model_stage2=model_stage2,
            model_type=model_type,
            spatial_transformer_r1=spatial_transformer_r1,
            composer_r1=composer_r1,
        )

    deformed_np = musa.utils_basics.torch2numpy(deformed_t, CHECK=True)
    stage1_deformed_np = musa.utils_basics.torch2numpy(stage1_deformed_t, CHECK=True)
    dvf_np = musa.utils_basics.torch2numpy(dvf_t, CHECK=True)

    metrics: Dict[str, object] = {
        "moving_id": moving_id,
        "fixed_id": fixed_id,
        "model_type": musa.utils_model_zoo.normalize_model_type(model_type),
    }

    moving_seg_o = load_array(npy_path(data_root, "seg_o", moving_id), RESOLUTION_SHAPES["r1"]).astype(np.int16)
    fixed_seg_o = load_array(npy_path(data_root, "seg_o", fixed_id), RESOLUTION_SHAPES["r1"]).astype(np.int16)
    warped_seg_o = warp_segmentation(moving_seg_o, dvf_t, spatial_transformer_r1, device)
    metrics["dice_seg_o"] = dice_for_labels(moving_seg_o, fixed_seg_o, warped_seg_o)

    moving_seg_b = load_array(npy_path(data_root, "seg_b", moving_id), RESOLUTION_SHAPES["r1"]).astype(np.int16)
    fixed_seg_b = load_array(npy_path(data_root, "seg_b", fixed_id), RESOLUTION_SHAPES["r1"]).astype(np.int16)
    warped_seg_b = warp_segmentation(moving_seg_b, dvf_t, spatial_transformer_r1, device)
    metrics["dice_seg_b"] = dice_for_labels(moving_seg_b, fixed_seg_b, warped_seg_b)

    return metrics, deformed_np, dvf_np, stage1_deformed_np, warped_seg_o, warped_seg_b


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "moving_id",
        "fixed_id",
        "seg_o_before",
        "seg_o_after",
        "seg_o_delta",
        "seg_b_before",
        "seg_b_after",
        "seg_b_delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    summary = {"num_pairs": len(rows)}
    for prefix in ("seg_o", "seg_b"):
        for field in ("before", "after", "delta"):
            values = np.array([float(row[f"{prefix}_{field}"]) for row in rows], dtype=np.float64)
            summary[f"{prefix}_{field}_mean"] = float(np.nanmean(values))
            summary[f"{prefix}_{field}_std"] = float(np.nanstd(values))
    return summary


def main() -> None:
    args = parse_args()
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    pairs = read_pairs(Path(args.pairs_csv))
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_stage1 = load_model(args.model_type, "r2", args.checkpoint_stage1, device)
    model_stage2 = load_model(args.model_type, "r1", args.checkpoint_stage2, device)
    spatial_transformer_r1 = musa.utils_warp.SpatialTransformer(RESOLUTION_SHAPES["r1"]).to(device)
    composer_r1 = musa.utils_warp.ComposeDVF(RESOLUTION_SHAPES["r1"]).to(device)

    rows = []
    pair_metrics = []
    for index, (moving_id, fixed_id) in enumerate(pairs, start=1):
        print(f"[{index}/{len(pairs)}] {moving_id} -> {fixed_id}", flush=True)
        metrics, deformed_np, dvf_np, stage1_deformed_np, warped_seg_o, warped_seg_b = evaluate_pair(
            moving_id=moving_id,
            fixed_id=fixed_id,
            data_root=data_root,
            model_type=args.model_type,
            model_stage1=model_stage1,
            model_stage2=model_stage2,
            spatial_transformer_r1=spatial_transformer_r1,
            composer_r1=composer_r1,
            device=device,
        )
        pair_metrics.append(metrics)

        row = {
            "moving_id": moving_id,
            "fixed_id": fixed_id,
            "seg_o_before": metric_value(metrics, "dice_seg_o", "mean_before"),
            "seg_o_after": metric_value(metrics, "dice_seg_o", "mean_after"),
            "seg_o_delta": metric_value(metrics, "dice_seg_o", "mean_delta"),
            "seg_b_before": metric_value(metrics, "dice_seg_b", "mean_before"),
            "seg_b_after": metric_value(metrics, "dice_seg_b", "mean_after"),
            "seg_b_delta": metric_value(metrics, "dice_seg_b", "mean_delta"),
        }
        rows.append(row)
        print(
            "[INFO] "
            f"seg_o {row['seg_o_before']:.6f}->{row['seg_o_after']:.6f} ({row['seg_o_delta']:+.6f}); "
            f"seg_b {row['seg_b_before']:.6f}->{row['seg_b_after']:.6f} ({row['seg_b_delta']:+.6f})",
            flush=True,
        )

        if args.save_pair_outputs:
            pair_dir = output_dir / "pairs" / f"{moving_id}_to_{fixed_id}"
            save_outputs(
                pair_dir,
                f"{moving_id}_to_{fixed_id}",
                deformed_np,
                dvf_np,
                stage1_deformed_np,
                warped_seg_o,
                warped_seg_b,
                metrics,
            )

    summary = summarize(rows)
    summary_payload = {
        "summary": summary,
        "checkpoint_stage1": args.checkpoint_stage1,
        "checkpoint_stage2": args.checkpoint_stage2,
        "model_type": musa.utils_model_zoo.normalize_model_type(args.model_type),
        "pairs": pair_metrics,
    }

    write_summary_csv(output_dir / "summary.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"[INFO] Wrote {output_dir / 'summary.csv'}")
    print(f"[INFO] Wrote {output_dir / 'summary.json'}")
    print(
        "[INFO] Summary: "
        f"seg_o {summary['seg_o_before_mean']:.6f}->{summary['seg_o_after_mean']:.6f} "
        f"({summary['seg_o_delta_mean']:+.6f}); "
        f"seg_b {summary['seg_b_before_mean']:.6f}->{summary['seg_b_after_mean']:.6f} "
        f"({summary['seg_b_delta_mean']:+.6f})"
    )


if __name__ == "__main__":
    main()
