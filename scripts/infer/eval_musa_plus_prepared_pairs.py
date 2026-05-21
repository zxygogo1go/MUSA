"""Evaluate MUSA+ Stage-3 inference over multiple prepared `.npy` pairs."""

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
from infer_musa_plus_prepared_pair import (
    RESOLUTION_SHAPES,
    dice_for_labels,
    load_array,
    load_model,
    npy_path,
    parse_int_tuple,
    resolve_option,
    resolve_small_labels,
    run_stage3,
    run_two_stage,
    tensor_from_array,
    warp_segmentation,
)
from musa.registration_models.musa_plus import LocalResidualUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MUSA+ Stage-3 over multiple prepared .npy pairs.")
    parser.add_argument("--pairs-csv", required=True, help="CSV with moving_id,fixed_id rows.")
    parser.add_argument("--data-root", default="data", help="Prepared data root containing images, seg_o, seg_b.")
    parser.add_argument("--model-type", default="05dualprnet-v1", help="Model type used for Stage 1/2.")
    parser.add_argument("--checkpoint-stage1", required=True, help="Stage 1 r2 checkpoint path.")
    parser.add_argument("--checkpoint-stage2", required=True, help="Stage 2 r1 checkpoint path.")
    parser.add_argument("--checkpoint-stage3", required=True, help="MUSA+ Stage-3 checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for summary files.")
    parser.add_argument("--metadata-path", default=None, help="Optional metadata path for resolving small-OAR names.")
    parser.add_argument("--small-oar-labels", default=None, help="Comma-separated small-OAR labels.")
    parser.add_argument(
        "--small-oar-names",
        default=",".join(musa.utils_musa_plus.DEFAULT_SMALL_OAR_NAMES),
        help="Comma-separated SegRap small-OAR names used with metadata.",
    )
    parser.add_argument("--stage3-filters", default=None, help="Stage-3 filters. Default: checkpoint args or 8,16,32.")
    parser.add_argument(
        "--stage3-input-mode",
        default=None,
        choices=musa.utils_musa_plus.STAGE3_INPUT_MODES,
        help="Ablation/information policy. Default: checkpoint args or full.",
    )
    parser.add_argument("--roi-radius-min", type=int, default=None, help="Minimum ROI dilation radius.")
    parser.add_argument("--roi-radius-max", type=int, default=None, help="Maximum ROI dilation radius.")
    parser.add_argument("--roi-smooth-steps", type=int, default=None, help="ROI smoothing steps.")
    parser.add_argument("--residual-scale-min", type=float, default=None, help="Residual scale for easiest pairs.")
    parser.add_argument("--residual-scale-max", type=float, default=None, help="Residual scale for hardest pairs.")
    parser.add_argument("--save-pair-metrics", action="store_true", help="Save one JSON metrics file per pair.")
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


def checkpoint_arg(payload: object, key: str, default):
    if isinstance(payload, dict):
        args = payload.get("args", {})
        if isinstance(args, dict):
            return args.get(key, default)
    return default


def evaluate_pair(
    moving_id: str,
    fixed_id: str,
    data_root: Path,
    model_type: str,
    model_stage1: torch.nn.Module,
    model_stage2: torch.nn.Module,
    model_stage3: torch.nn.Module,
    transformer: torch.nn.Module,
    composer: torch.nn.Module,
    device: torch.device,
    small_oar_labels: List[int],
    roi_radius_min: int,
    roi_radius_max: int,
    roi_smooth_steps: int,
    residual_scale_min: float,
    residual_scale_max: float,
    stage3_input_mode: str,
) -> Dict[str, object]:
    moving_np = load_array(npy_path(data_root, "images", moving_id), RESOLUTION_SHAPES["r1"]).astype(np.float32)
    fixed_np = load_array(npy_path(data_root, "images", fixed_id), RESOLUTION_SHAPES["r1"]).astype(np.float32)
    moving_seg_o = load_array(npy_path(data_root, "seg_o", moving_id), RESOLUTION_SHAPES["r1"]).astype(np.int16)
    fixed_seg_o = load_array(npy_path(data_root, "seg_o", fixed_id), RESOLUTION_SHAPES["r1"]).astype(np.int16)
    moving_seg_b = load_array(npy_path(data_root, "seg_b", moving_id), RESOLUTION_SHAPES["r1"]).astype(np.int16)
    fixed_seg_b = load_array(npy_path(data_root, "seg_b", fixed_id), RESOLUTION_SHAPES["r1"]).astype(np.int16)

    moving_t = musa.utils_basics.numpy2torch(moving_np, device=device, CHECK=True)
    fixed_t = musa.utils_basics.numpy2torch(fixed_np, device=device, CHECK=True)
    moving_seg_o_t = tensor_from_array(moving_seg_o, device, torch.long)
    fixed_seg_o_t = tensor_from_array(fixed_seg_o, device, torch.long)
    moving_seg_b_t = tensor_from_array(moving_seg_b, device, torch.long)
    fixed_seg_b_t = tensor_from_array(fixed_seg_b, device, torch.long)

    with torch.no_grad():
        deformed_stage2_t, dvf_stage2_t, _ = run_two_stage(
            moving_t=moving_t,
            fixed_t=fixed_t,
            model_stage1=model_stage1,
            model_stage2=model_stage2,
            model_type=model_type,
            spatial_transformer_r1=transformer,
            composer_r1=composer,
        )
        _, dvf_final_t, stage3_metrics = run_stage3(
            moving_t=moving_t,
            fixed_t=fixed_t,
            moving_seg_o_t=moving_seg_o_t,
            fixed_seg_o_t=fixed_seg_o_t,
            moving_seg_b_t=moving_seg_b_t,
            fixed_seg_b_t=fixed_seg_b_t,
            deformed_stage2_t=deformed_stage2_t,
            dvf_stage2_t=dvf_stage2_t,
            model_stage3=model_stage3,
            transformer=transformer,
            small_oar_labels=small_oar_labels,
            roi_radius_min=roi_radius_min,
            roi_radius_max=roi_radius_max,
            roi_smooth_steps=roi_smooth_steps,
            residual_scale_min=residual_scale_min,
            residual_scale_max=residual_scale_max,
            stage3_input_mode=stage3_input_mode,
        )

    warped_seg_o_stage2 = warp_segmentation(moving_seg_o, dvf_stage2_t, transformer, device)
    warped_seg_b_stage2 = warp_segmentation(moving_seg_b, dvf_stage2_t, transformer, device)
    warped_seg_o_final = warp_segmentation(moving_seg_o, dvf_final_t, transformer, device)
    warped_seg_b_final = warp_segmentation(moving_seg_b, dvf_final_t, transformer, device)

    oar_present = np.union1d(np.unique(moving_seg_o), np.unique(fixed_seg_o))
    oar_present = np.union1d(oar_present, np.unique(warped_seg_o_stage2))
    oar_present = np.union1d(oar_present, np.unique(warped_seg_o_final)).astype(np.int64)
    small_present, large_present = musa.utils_musa_plus.split_present_labels(oar_present, small_oar_labels)
    bone_present = np.union1d(np.unique(moving_seg_b), np.unique(fixed_seg_b))
    bone_present = np.union1d(bone_present, np.unique(warped_seg_b_stage2))
    bone_present = np.union1d(bone_present, np.unique(warped_seg_b_final)).astype(np.int64)
    bone_present = [int(label) for label in bone_present if int(label) > 0]

    return {
        "moving_id": moving_id,
        "fixed_id": fixed_id,
        "stage3": stage3_metrics,
        "dice_seg_o_stage2": dice_for_labels(moving_seg_o, fixed_seg_o, warped_seg_o_stage2),
        "dice_seg_b_stage2": dice_for_labels(moving_seg_b, fixed_seg_b, warped_seg_b_stage2),
        "dice_seg_o_musa_plus": dice_for_labels(moving_seg_o, fixed_seg_o, warped_seg_o_final),
        "dice_seg_b_musa_plus": dice_for_labels(moving_seg_b, fixed_seg_b, warped_seg_b_final),
        "small_oar_per_label": musa.utils_musa_plus.label_dice_table(
            moving_seg_o,
            fixed_seg_o,
            warped_seg_o_stage2,
            warped_seg_o_final,
            small_present,
        ),
        "large_oar_per_label": musa.utils_musa_plus.label_dice_table(
            moving_seg_o,
            fixed_seg_o,
            warped_seg_o_stage2,
            warped_seg_o_final,
            large_present,
        ),
        "bone_per_label": musa.utils_musa_plus.label_dice_table(
            moving_seg_b,
            fixed_seg_b,
            warped_seg_b_stage2,
            warped_seg_b_final,
            bone_present,
        ),
    }


def row_from_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    stage3 = metrics["stage3"]
    small = metrics["small_oar_per_label"]
    large = metrics["large_oar_per_label"]
    bone = metrics["bone_per_label"]
    return {
        "moving_id": metrics["moving_id"],
        "fixed_id": metrics["fixed_id"],
        "difficulty": stage3["difficulty"],
        "roi_radius": stage3["roi_radius"],
        "residual_scale": stage3["residual_scale"],
        "small_soft_stage2": stage3["small_stage2_dice"],
        "small_soft_final": stage3["small_final_dice"],
        "small_soft_delta": stage3["small_delta"],
        "small_label_stage2": small["mean_stage2"],
        "small_label_final": small["mean_final"],
        "small_label_delta": small["mean_delta"],
        "small_label_worst_delta": small["worst_delta"],
        "large_label_delta": large["mean_delta"],
        "large_label_worst_delta": large["worst_delta"],
        "large_drop_gt_0_02": large["num_drop_gt_0_02"],
        "large_drop_gt_0_05": large["num_drop_gt_0_05"],
        "bone_label_delta": bone["mean_delta"],
        "bone_label_worst_delta": bone["worst_delta"],
        "bone_drop_gt_0_02": bone["num_drop_gt_0_02"],
        "bone_drop_gt_0_05": bone["num_drop_gt_0_05"],
        "residual_roi_p95": stage3["residual_magnitude"]["roi_p95"],
        "residual_roi_max": stage3["residual_magnitude"]["roi_max"],
        "final_jac_nonpos": stage3["final_jacobian"]["global_nonpos_ratio"],
        "final_jac_roi_nonpos": stage3["final_jacobian"]["roi_nonpos_ratio"],
        "final_jac_roi_min": stage3["final_jacobian"]["roi_min"],
    }


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    summary = {"num_pairs": len(rows)}
    numeric_keys = [key for key in rows[0] if key not in ("moving_id", "fixed_id")]
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values))
    return summary


def summarize_by_difficulty(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Summarize easy/medium/hard pair buckets using the adaptive difficulty score."""

    if not rows:
        return {}
    values = np.asarray([float(row["difficulty"]) for row in rows], dtype=np.float64)
    q_easy, q_hard = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    buckets = {
        "easy": [row for row in rows if float(row["difficulty"]) <= q_easy],
        "medium": [row for row in rows if q_easy < float(row["difficulty"]) <= q_hard],
        "hard": [row for row in rows if float(row["difficulty"]) > q_hard],
    }
    return {
        name: {
            "difficulty_min": float(min(float(row["difficulty"]) for row in bucket)),
            "difficulty_max": float(max(float(row["difficulty"]) for row in bucket)),
            **summarize(bucket),
        }
        for name, bucket in buckets.items()
        if bucket
    }


def main() -> None:
    args = parse_args()
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage3_payload = torch.load(args.checkpoint_stage3, map_location=device)
    filters_value = args.stage3_filters or checkpoint_arg(stage3_payload, "filters", "8,16,32")
    model_stage3 = LocalResidualUNet(in_channels=7, out_channels=3, filters=parse_int_tuple(filters_value))
    model_stage3.load_state_dict(musa.utils_musa_plus.checkpoint_to_state_dict(stage3_payload))
    model_stage3.to(device)
    model_stage3.eval()

    small_oar_labels = resolve_small_labels(args, stage3_payload, data_root)
    roi_radius_min = int(resolve_option(args, stage3_payload, "roi_radius_min", 3))
    roi_radius_max = int(resolve_option(args, stage3_payload, "roi_radius_max", 8))
    roi_smooth_steps = int(resolve_option(args, stage3_payload, "roi_smooth_steps", 2))
    residual_scale_min = float(resolve_option(args, stage3_payload, "residual_scale_min", 0.25))
    residual_scale_max = float(resolve_option(args, stage3_payload, "residual_scale_max", 1.0))
    stage3_input_mode = str(resolve_option(args, stage3_payload, "stage3_input_mode", "full"))

    pairs = read_pairs(Path(args.pairs_csv))
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]

    model_stage1 = load_model(args.model_type, "r2", args.checkpoint_stage1, device)
    model_stage2 = load_model(args.model_type, "r1", args.checkpoint_stage2, device)
    transformer = musa.utils_warp.SpatialTransformer(RESOLUTION_SHAPES["r1"]).to(device)
    composer = musa.utils_warp.ComposeDVF(RESOLUTION_SHAPES["r1"]).to(device)

    rows = []
    all_metrics = []
    for index, (moving_id, fixed_id) in enumerate(pairs, start=1):
        print(f"[{index}/{len(pairs)}] {moving_id} -> {fixed_id}", flush=True)
        metrics = evaluate_pair(
            moving_id=moving_id,
            fixed_id=fixed_id,
            data_root=data_root,
            model_type=args.model_type,
            model_stage1=model_stage1,
            model_stage2=model_stage2,
            model_stage3=model_stage3,
            transformer=transformer,
            composer=composer,
            device=device,
            small_oar_labels=small_oar_labels,
            roi_radius_min=roi_radius_min,
            roi_radius_max=roi_radius_max,
            roi_smooth_steps=roi_smooth_steps,
            residual_scale_min=residual_scale_min,
            residual_scale_max=residual_scale_max,
            stage3_input_mode=stage3_input_mode,
        )
        row = row_from_metrics(metrics)
        rows.append(row)
        all_metrics.append(metrics)
        print(
            "[INFO] "
            f"difficulty {row['difficulty']:.3f}, roi {row['roi_radius']:.1f}; "
            f"small-label {row['small_label_stage2']:.4f}->{row['small_label_final']:.4f} "
            f"({row['small_label_delta']:+.4f}); "
            f"large worst {row['large_label_worst_delta']:+.4f}; "
            f"bone worst {row['bone_label_worst_delta']:+.4f}; "
            f"jac roi<=0 {row['final_jac_roi_nonpos']:.3e}",
            flush=True,
        )
        if args.save_pair_metrics:
            pair_path = output_dir / "pairs" / f"{moving_id}_to_{fixed_id}_musa_plus_metrics.json"
            pair_path.parent.mkdir(parents=True, exist_ok=True)
            pair_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary = summarize(rows)
    difficulty_buckets = summarize_by_difficulty(rows)
    payload = {
        "summary": summary,
        "difficulty_buckets": difficulty_buckets,
        "checkpoint_stage1": args.checkpoint_stage1,
        "checkpoint_stage2": args.checkpoint_stage2,
        "checkpoint_stage3": args.checkpoint_stage3,
        "stage3_input_mode": stage3_input_mode,
        "small_oar_labels": small_oar_labels,
        "rows": rows,
    }
    write_summary_csv(output_dir / "musa_plus_summary.csv", rows)
    (output_dir / "musa_plus_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "musa_plus_pair_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"[INFO] Summary CSV: {output_dir / 'musa_plus_summary.csv'}")
    print(f"[INFO] Summary JSON: {output_dir / 'musa_plus_summary.json'}")


if __name__ == "__main__":
    main()
