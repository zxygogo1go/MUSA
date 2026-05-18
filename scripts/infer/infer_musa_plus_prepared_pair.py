"""Run MUSA+ Stage-3 inference on one prepared `.npy` image/segmentation pair."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    warp_segmentation,
)
from musa.registration_models.musa_plus import LocalResidualUNet


def parse_int_tuple(value: str) -> Tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MUSA+ Stage-3 inference on prepared .npy pairs.")
    parser.add_argument("--moving-id", required=True, help="Moving case ID, e.g. segrap_0000.")
    parser.add_argument("--fixed-id", required=True, help="Fixed case ID, e.g. segrap_0001.")
    parser.add_argument("--data-root", default="data", help="Prepared data root containing images, seg_o, seg_b.")
    parser.add_argument("--model-type", default="05dualprnet-v1", help="Model type used for Stage 1/2.")
    parser.add_argument("--checkpoint-stage1", required=True, help="Stage 1 r2 checkpoint path.")
    parser.add_argument("--checkpoint-stage2", required=True, help="Stage 2 r1 checkpoint path.")
    parser.add_argument("--checkpoint-stage3", required=True, help="MUSA+ Stage-3 checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for .npy outputs and metrics.")
    parser.add_argument("--output-prefix", default=None, help="Output prefix. Default: <moving-id>_to_<fixed-id>.")
    parser.add_argument("--metadata-path", default=None, help="Optional metadata path for resolving small-OAR names.")
    parser.add_argument("--small-oar-labels", default=None, help="Comma-separated small-OAR labels.")
    parser.add_argument(
        "--small-oar-names",
        default=",".join(musa.utils_musa_plus.DEFAULT_SMALL_OAR_NAMES),
        help="Comma-separated SegRap small-OAR names used with metadata.",
    )
    parser.add_argument("--stage3-filters", default=None, help="Stage-3 filters. Default: checkpoint args or 8,16,32.")
    parser.add_argument("--roi-radius-min", type=int, default=None, help="Minimum ROI dilation radius.")
    parser.add_argument("--roi-radius-max", type=int, default=None, help="Maximum ROI dilation radius.")
    parser.add_argument("--roi-smooth-steps", type=int, default=None, help="ROI smoothing steps.")
    parser.add_argument("--residual-scale-min", type=float, default=None, help="Residual scale for easiest pairs.")
    parser.add_argument("--residual-scale-max", type=float, default=None, help="Residual scale for hardest pairs.")
    parser.add_argument("--gpu", default="0", help="CUDA visible GPU IDs.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    return parser.parse_args()


def checkpoint_arg(payload: object, key: str, default):
    if isinstance(payload, dict):
        args = payload.get("args", {})
        if isinstance(args, dict):
            return args.get(key, default)
    return default


def resolve_option(args: argparse.Namespace, payload: object, attr: str, default):
    value = getattr(args, attr)
    if value is not None:
        return value
    return checkpoint_arg(payload, attr, default)


def tensor_from_array(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(array[np.newaxis, np.newaxis, ...]).to(device=device, dtype=dtype)


def resolve_small_labels(args: argparse.Namespace, payload: object, data_root: Path) -> List[int]:
    metadata_path = args.metadata_path
    if metadata_path is None:
        candidate = data_root / "metadata"
        metadata_path = str(candidate) if candidate.exists() else None
    labels = musa.utils_musa_plus.resolve_small_oar_labels(
        small_oar_labels=args.small_oar_labels,
        small_oar_names=args.small_oar_names,
        metadata_path=metadata_path,
    )
    if labels:
        return labels
    if isinstance(payload, dict) and payload.get("small_oar_labels"):
        return [int(label) for label in payload["small_oar_labels"]]
    raise ValueError("Could not resolve small-OAR labels from args, metadata, or Stage-3 checkpoint.")


def run_stage3(
    moving_t: torch.Tensor,
    fixed_t: torch.Tensor,
    moving_seg_o_t: torch.Tensor,
    fixed_seg_o_t: torch.Tensor,
    moving_seg_b_t: torch.Tensor,
    fixed_seg_b_t: torch.Tensor,
    deformed_stage2_t: torch.Tensor,
    dvf_stage2_t: torch.Tensor,
    model_stage3: torch.nn.Module,
    transformer: torch.nn.Module,
    small_oar_labels: Sequence[int],
    roi_radius_min: int,
    roi_radius_max: int,
    roi_smooth_steps: int,
    residual_scale_min: float,
    residual_scale_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    moving_oar = musa.utils_musa_plus.seg_to_foreground_mask(moving_seg_o_t)
    fixed_oar = musa.utils_musa_plus.seg_to_foreground_mask(fixed_seg_o_t)
    moving_small = musa.utils_musa_plus.seg_to_label_mask(moving_seg_o_t, small_oar_labels)
    fixed_small = musa.utils_musa_plus.seg_to_label_mask(fixed_seg_o_t, small_oar_labels)
    moving_bone = (moving_seg_b_t > 0).float()
    fixed_bone = (fixed_seg_b_t > 0).float()

    difficulty = musa.utils_musa_plus.estimate_pair_difficulty(
        moving=moving_t,
        fixed=fixed_t,
        moving_oar_mask=moving_oar,
        fixed_oar_mask=fixed_oar,
        moving_bone_mask=moving_bone,
        fixed_bone_mask=fixed_bone,
    )
    warped_small_stage2 = transformer(moving_small, dvf_stage2_t, mode="bilinear").clamp(0.0, 1.0)
    warped_bone_stage2 = transformer(moving_bone, dvf_stage2_t, mode="bilinear").clamp(0.0, 1.0)

    roi_source = torch.maximum(fixed_small, warped_small_stage2)
    roi_radius = musa.utils_musa_plus.difficulty_to_radius(difficulty, roi_radius_min, roi_radius_max)
    roi_gate = musa.utils_musa_plus.build_roi_gate(roi_source, radius=roi_radius, smooth_steps=roi_smooth_steps)
    stage3_inputs = musa.utils_musa_plus.make_stage3_inputs(
        fixed=fixed_t,
        deformed_stage2=deformed_stage2_t,
        fixed_small_mask=fixed_small,
        warped_small_mask_stage2=warped_small_stage2,
        dvf_stage2=dvf_stage2_t,
        fixed_bone_mask=fixed_bone,
        warped_bone_mask_stage2=warped_bone_stage2,
    )

    raw_local_dvf = model_stage3(stage3_inputs)
    residual_scale = musa.utils_musa_plus.difficulty_to_value(
        difficulty,
        residual_scale_min,
        residual_scale_max,
    ).view(-1, 1, 1, 1, 1)
    gated_local_dvf = raw_local_dvf * residual_scale * roi_gate
    dvf_final_t = dvf_stage2_t + gated_local_dvf
    deformed_final_t = transformer(moving_t, dvf_final_t, mode="bilinear")
    warped_small_final = transformer(moving_small, dvf_final_t, mode="bilinear").clamp(0.0, 1.0)

    small_stage2_dice = musa.utils_musa_plus.binary_dice_per_batch(warped_small_stage2, fixed_small).mean()
    small_final_dice = musa.utils_musa_plus.binary_dice_per_batch(warped_small_final, fixed_small).mean()
    residual_mag = torch.sqrt(gated_local_dvf.pow(2).sum(dim=1) + 1e-6).mean()
    metrics = {
        "difficulty": float(difficulty.mean().detach().cpu()),
        "roi_radius": float(roi_radius),
        "residual_scale": float(residual_scale.mean().detach().cpu()),
        "small_stage2_dice": float(small_stage2_dice.detach().cpu()),
        "small_final_dice": float(small_final_dice.detach().cpu()),
        "small_delta": float((small_final_dice - small_stage2_dice).detach().cpu()),
        "residual_mag_mean": float(residual_mag.detach().cpu()),
    }
    return deformed_final_t, dvf_final_t, metrics


def save_musa_plus_outputs(
    output_dir: Path,
    prefix: str,
    deformed_final_np: np.ndarray,
    dvf_final_np: np.ndarray,
    deformed_stage2_np: np.ndarray,
    dvf_stage2_np: np.ndarray,
    stage1_deformed_np: np.ndarray,
    warped_seg_o_final: np.ndarray,
    warped_seg_b_final: np.ndarray,
    warped_seg_o_stage2: np.ndarray,
    warped_seg_b_stage2: np.ndarray,
    metrics: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{prefix}_musa_plus_deformed_img.npy", deformed_final_np.astype(np.float32))
    np.save(output_dir / f"{prefix}_musa_plus_dvf_chwd.npy", dvf_final_np.astype(np.float32))
    np.save(output_dir / f"{prefix}_musa_plus_dvf_hwdc.npy", np.moveaxis(dvf_final_np, 0, -1).astype(np.float32))
    np.save(output_dir / f"{prefix}_stage2_deformed_img.npy", deformed_stage2_np.astype(np.float32))
    np.save(output_dir / f"{prefix}_stage2_dvf_chwd.npy", dvf_stage2_np.astype(np.float32))
    np.save(output_dir / f"{prefix}_stage2_dvf_hwdc.npy", np.moveaxis(dvf_stage2_np, 0, -1).astype(np.float32))
    np.save(output_dir / f"{prefix}_stage1_deformed_img.npy", stage1_deformed_np.astype(np.float32))
    np.save(output_dir / f"{prefix}_musa_plus_deformed_seg_o.npy", warped_seg_o_final.astype(np.int16))
    np.save(output_dir / f"{prefix}_musa_plus_deformed_seg_b.npy", warped_seg_b_final.astype(np.int16))
    np.save(output_dir / f"{prefix}_stage2_deformed_seg_o.npy", warped_seg_o_stage2.astype(np.int16))
    np.save(output_dir / f"{prefix}_stage2_deformed_seg_b.npy", warped_seg_b_stage2.astype(np.int16))
    (output_dir / f"{prefix}_musa_plus_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    prefix = args.output_prefix or f"{args.moving_id}_to_{args.fixed_id}"

    stage3_payload = torch.load(args.checkpoint_stage3, map_location=device)
    filters_value = args.stage3_filters or checkpoint_arg(stage3_payload, "filters", "8,16,32")
    filters = parse_int_tuple(filters_value)
    model_stage3 = LocalResidualUNet(in_channels=7, out_channels=3, filters=filters)
    model_stage3.load_state_dict(musa.utils_musa_plus.checkpoint_to_state_dict(stage3_payload))
    model_stage3.to(device)
    model_stage3.eval()

    small_oar_labels = resolve_small_labels(args, stage3_payload, data_root)
    roi_radius_min = int(resolve_option(args, stage3_payload, "roi_radius_min", 3))
    roi_radius_max = int(resolve_option(args, stage3_payload, "roi_radius_max", 8))
    roi_smooth_steps = int(resolve_option(args, stage3_payload, "roi_smooth_steps", 2))
    residual_scale_min = float(resolve_option(args, stage3_payload, "residual_scale_min", 0.25))
    residual_scale_max = float(resolve_option(args, stage3_payload, "residual_scale_max", 1.0))

    moving_np = load_array(npy_path(data_root, "images", args.moving_id), RESOLUTION_SHAPES["r1"]).astype(np.float32)
    fixed_np = load_array(npy_path(data_root, "images", args.fixed_id), RESOLUTION_SHAPES["r1"]).astype(np.float32)
    moving_seg_o = load_array(npy_path(data_root, "seg_o", args.moving_id), RESOLUTION_SHAPES["r1"]).astype(np.int16)
    fixed_seg_o = load_array(npy_path(data_root, "seg_o", args.fixed_id), RESOLUTION_SHAPES["r1"]).astype(np.int16)
    moving_seg_b = load_array(npy_path(data_root, "seg_b", args.moving_id), RESOLUTION_SHAPES["r1"]).astype(np.int16)
    fixed_seg_b = load_array(npy_path(data_root, "seg_b", args.fixed_id), RESOLUTION_SHAPES["r1"]).astype(np.int16)

    moving_t = musa.utils_basics.numpy2torch(moving_np, device=device, CHECK=True)
    fixed_t = musa.utils_basics.numpy2torch(fixed_np, device=device, CHECK=True)
    moving_seg_o_t = tensor_from_array(moving_seg_o, device, torch.long)
    fixed_seg_o_t = tensor_from_array(fixed_seg_o, device, torch.long)
    moving_seg_b_t = tensor_from_array(moving_seg_b, device, torch.long)
    fixed_seg_b_t = tensor_from_array(fixed_seg_b, device, torch.long)

    model_stage1 = load_model(args.model_type, "r2", args.checkpoint_stage1, device)
    model_stage2 = load_model(args.model_type, "r1", args.checkpoint_stage2, device)
    transformer = musa.utils_warp.SpatialTransformer(RESOLUTION_SHAPES["r1"]).to(device)
    composer = musa.utils_warp.ComposeDVF(RESOLUTION_SHAPES["r1"]).to(device)

    with torch.no_grad():
        deformed_stage2_t, dvf_stage2_t, stage1_deformed_t = run_two_stage(
            moving_t=moving_t,
            fixed_t=fixed_t,
            model_stage1=model_stage1,
            model_stage2=model_stage2,
            model_type=args.model_type,
            spatial_transformer_r1=transformer,
            composer_r1=composer,
        )
        deformed_final_t, dvf_final_t, stage3_metrics = run_stage3(
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
        )

    deformed_final_np = musa.utils_basics.torch2numpy(deformed_final_t, CHECK=True)
    dvf_final_np = musa.utils_basics.torch2numpy(dvf_final_t, CHECK=True)
    deformed_stage2_np = musa.utils_basics.torch2numpy(deformed_stage2_t, CHECK=True)
    dvf_stage2_np = musa.utils_basics.torch2numpy(dvf_stage2_t, CHECK=True)
    stage1_deformed_np = musa.utils_basics.torch2numpy(stage1_deformed_t, CHECK=True)

    warped_seg_o_stage2 = warp_segmentation(moving_seg_o, dvf_stage2_t, transformer, device)
    warped_seg_b_stage2 = warp_segmentation(moving_seg_b, dvf_stage2_t, transformer, device)
    warped_seg_o_final = warp_segmentation(moving_seg_o, dvf_final_t, transformer, device)
    warped_seg_b_final = warp_segmentation(moving_seg_b, dvf_final_t, transformer, device)

    metrics: Dict[str, object] = {
        "moving_id": args.moving_id,
        "fixed_id": args.fixed_id,
        "model_type": musa.utils_model_zoo.normalize_model_type(args.model_type),
        "checkpoint_stage1": args.checkpoint_stage1,
        "checkpoint_stage2": args.checkpoint_stage2,
        "checkpoint_stage3": args.checkpoint_stage3,
        "small_oar_labels": [int(label) for label in small_oar_labels],
        "stage3": stage3_metrics,
        "dice_seg_o_stage2": dice_for_labels(moving_seg_o, fixed_seg_o, warped_seg_o_stage2),
        "dice_seg_b_stage2": dice_for_labels(moving_seg_b, fixed_seg_b, warped_seg_b_stage2),
        "dice_seg_o_musa_plus": dice_for_labels(moving_seg_o, fixed_seg_o, warped_seg_o_final),
        "dice_seg_b_musa_plus": dice_for_labels(moving_seg_b, fixed_seg_b, warped_seg_b_final),
    }

    save_musa_plus_outputs(
        output_dir,
        prefix,
        deformed_final_np,
        dvf_final_np,
        deformed_stage2_np,
        dvf_stage2_np,
        stage1_deformed_np,
        warped_seg_o_final,
        warped_seg_b_final,
        warped_seg_o_stage2,
        warped_seg_b_stage2,
        metrics,
    )

    print(f"[INFO] Saved MUSA+ outputs to {output_dir}")
    print(
        "[INFO] small-OAR Dice "
        f"{stage3_metrics['small_stage2_dice']:.6f} -> {stage3_metrics['small_final_dice']:.6f} "
        f"({stage3_metrics['small_delta']:+.6f})"
    )
    print(f"[INFO] Metrics: {output_dir / f'{prefix}_musa_plus_metrics.json'}")


if __name__ == "__main__":
    main()
