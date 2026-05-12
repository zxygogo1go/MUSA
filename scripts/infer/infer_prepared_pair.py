"""Run DIR-MUSA inference on prepared `.npy` image/segmentation pairs."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import musa


RESOLUTION_SHAPES = {
    "r1": (160, 160, 192),
    "r2": (80, 80, 96),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-stage DIR-MUSA inference on prepared .npy pairs.")
    parser.add_argument("--moving-id", required=True, help="Moving case ID, e.g. segrap_0000.")
    parser.add_argument("--fixed-id", required=True, help="Fixed case ID, e.g. segrap_0001.")
    parser.add_argument("--data-root", default="data", help="Prepared data root containing images, seg_o, and seg_b.")
    parser.add_argument("--model-type", default="01voxelmorph-v1", help="Model type used for both stages.")
    parser.add_argument("--checkpoint-stage1", required=True, help="Stage 1 r2 checkpoint path.")
    parser.add_argument("--checkpoint-stage2", required=True, help="Stage 2 r1 checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for .npy outputs and metrics.")
    parser.add_argument("--output-prefix", default=None, help="Output prefix. Default: <moving-id>_to_<fixed-id>.")
    parser.add_argument("--gpu", default="0", help="CUDA visible GPU IDs.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--debug", action="store_true", help="Print tensor/array overviews.")
    return parser.parse_args()


def checkpoint_to_state_dict(payload: object) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported checkpoint payload type: {type(payload)}")


def load_model(model_type: str, model_resolution: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = musa.utils_model_zoo.get_model_v1(
        inshape=RESOLUTION_SHAPES[model_resolution],
        model_type=model_type,
        model_resolution=model_resolution,
    )
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint_to_state_dict(payload))
    model.to(device)
    model.eval()
    return model


def npy_path(data_root: Path, folder: str, case_id: str) -> Path:
    return data_root / folder / f"{case_id}.npy"


def load_array(path: Path, expected_shape: Tuple[int, int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    array = np.load(path)
    if tuple(array.shape) != tuple(expected_shape):
        raise ValueError(f"{path} shape {array.shape} != {expected_shape}")
    return array


def run_two_stage(
    moving_t: torch.Tensor,
    fixed_t: torch.Tensor,
    model_stage1: torch.nn.Module,
    model_stage2: torch.nn.Module,
    model_type: str,
    spatial_transformer_r1: torch.nn.Module,
    composer_r1: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_type = musa.utils_model_zoo.normalize_model_type(model_type)
    flag_pad = model_type.startswith("04transmorph-v1")

    moving_r2 = musa.utils_warp.vol_downsamplex2(moving_t)
    fixed_r2 = musa.utils_warp.vol_downsamplex2(fixed_t)

    inputs_stage1 = (moving_r2, fixed_r2)
    if not flag_pad:
        _, dvf_r2_stage1 = musa.utils_model_zoo.model_register_v1(inputs_stage1, model_stage1, model_type)
    else:
        pad_size = (16, 16, 24, 24, 24, 24)
        padded_stage1 = [F.pad(d, pad=pad_size) for d in inputs_stage1]
        _, dvf_r2_stage1 = musa.utils_model_zoo.model_register_v1(padded_stage1, model_stage1, model_type)
        dvf_r2_stage1 = dvf_r2_stage1[..., 24:24 + 80, 24:24 + 80, 16:16 + 96]

    dvf_r1_stage1 = musa.utils_warp.dvf_upsample(dvf_r2_stage1)
    deformed_r1_stage1 = spatial_transformer_r1(moving_t, dvf_r1_stage1, mode="bilinear")

    inputs_stage2 = (deformed_r1_stage1, fixed_t)
    _, dvf_r1_stage2 = musa.utils_model_zoo.model_register_v1(inputs_stage2, model_stage2, model_type)

    dvf_composed = composer_r1(dvf_r1_stage1, dvf_r1_stage2)
    deformed_final = spatial_transformer_r1(moving_t, dvf_composed, mode="bilinear")
    return deformed_final, dvf_composed, deformed_r1_stage1


def warp_segmentation(seg_np: np.ndarray, dvf_t: torch.Tensor, transformer: torch.nn.Module, device: torch.device) -> np.ndarray:
    seg_t = musa.utils_basics.numpy2torch(seg_np.astype(np.float32), device=device, CHECK=True)
    warped_t = transformer(seg_t, dvf_t, mode="nearest")
    warped_np = musa.utils_basics.torch2numpy(warped_t, CHECK=True)
    return np.rint(warped_np).astype(np.int16)


def dice_for_labels(moving: np.ndarray, fixed: np.ndarray, warped: np.ndarray, include_background: bool = False) -> Dict[str, object]:
    labels = np.union1d(np.unique(moving), np.unique(fixed))
    labels = np.union1d(labels, np.unique(warped)).astype(np.int64)
    labels = np.sort(labels)
    if not include_background:
        labels = labels[labels != 0]

    per_label = {}
    before_values = []
    after_values = []
    for label in labels:
        moving_mask = moving == label
        fixed_mask = fixed == label
        warped_mask = warped == label
        before = (2.0 * np.logical_and(moving_mask, fixed_mask).sum()) / (moving_mask.sum() + fixed_mask.sum() + 1e-5)
        after = (2.0 * np.logical_and(warped_mask, fixed_mask).sum()) / (warped_mask.sum() + fixed_mask.sum() + 1e-5)
        per_label[str(int(label))] = {"before": float(before), "after": float(after), "delta": float(after - before)}
        before_values.append(before)
        after_values.append(after)

    mean_before = float(np.mean(before_values)) if before_values else 0.0
    mean_after = float(np.mean(after_values)) if after_values else 0.0
    return {
        "labels": [int(label) for label in labels],
        "mean_before": mean_before,
        "mean_after": mean_after,
        "mean_delta": float(mean_after - mean_before),
        "per_label": per_label,
    }


def save_outputs(
    output_dir: Path,
    prefix: str,
    deformed_np: np.ndarray,
    dvf_np: np.ndarray,
    stage1_deformed_np: np.ndarray,
    warped_seg_o: Optional[np.ndarray],
    warped_seg_b: Optional[np.ndarray],
    metrics: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{prefix}_deformed_img.npy", deformed_np.astype(np.float32))
    np.save(output_dir / f"{prefix}_stage1_deformed_img.npy", stage1_deformed_np.astype(np.float32))
    np.save(output_dir / f"{prefix}_dvf_chwd.npy", dvf_np.astype(np.float32))
    np.save(output_dir / f"{prefix}_dvf_hwdc.npy", np.moveaxis(dvf_np, 0, -1).astype(np.float32))
    if warped_seg_o is not None:
        np.save(output_dir / f"{prefix}_deformed_seg_o.npy", warped_seg_o.astype(np.int16))
    if warped_seg_b is not None:
        np.save(output_dir / f"{prefix}_deformed_seg_b.npy", warped_seg_b.astype(np.int16))
    (output_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    prefix = args.output_prefix or f"{args.moving_id}_to_{args.fixed_id}"

    moving_np = load_array(npy_path(data_root, "images", args.moving_id), RESOLUTION_SHAPES["r1"]).astype(np.float32)
    fixed_np = load_array(npy_path(data_root, "images", args.fixed_id), RESOLUTION_SHAPES["r1"]).astype(np.float32)

    moving_t = musa.utils_basics.numpy2torch(moving_np, device=device, CHECK=True)
    fixed_t = musa.utils_basics.numpy2torch(fixed_np, device=device, CHECK=True)

    if args.debug:
        musa.utils_basics.numpy_overview(moving_np, "moving_img")
        musa.utils_basics.numpy_overview(fixed_np, "fixed_img")

    model_stage1 = load_model(args.model_type, "r2", args.checkpoint_stage1, device)
    model_stage2 = load_model(args.model_type, "r1", args.checkpoint_stage2, device)
    spatial_transformer_r1 = musa.utils_warp.SpatialTransformer(RESOLUTION_SHAPES["r1"]).to(device)
    composer_r1 = musa.utils_warp.ComposeDVF(RESOLUTION_SHAPES["r1"]).to(device)

    with torch.no_grad():
        deformed_t, dvf_t, stage1_deformed_t = run_two_stage(
            moving_t=moving_t,
            fixed_t=fixed_t,
            model_stage1=model_stage1,
            model_stage2=model_stage2,
            model_type=args.model_type,
            spatial_transformer_r1=spatial_transformer_r1,
            composer_r1=composer_r1,
        )

    deformed_np = musa.utils_basics.torch2numpy(deformed_t, CHECK=True)
    stage1_deformed_np = musa.utils_basics.torch2numpy(stage1_deformed_t, CHECK=True)
    dvf_np = musa.utils_basics.torch2numpy(dvf_t, CHECK=True)

    metrics: Dict[str, object] = {
        "moving_id": args.moving_id,
        "fixed_id": args.fixed_id,
        "model_type": musa.utils_model_zoo.normalize_model_type(args.model_type),
        "checkpoint_stage1": args.checkpoint_stage1,
        "checkpoint_stage2": args.checkpoint_stage2,
    }

    warped_seg_o = None
    warped_seg_b = None
    seg_o_moving_path = npy_path(data_root, "seg_o", args.moving_id)
    seg_o_fixed_path = npy_path(data_root, "seg_o", args.fixed_id)
    seg_b_moving_path = npy_path(data_root, "seg_b", args.moving_id)
    seg_b_fixed_path = npy_path(data_root, "seg_b", args.fixed_id)

    if seg_o_moving_path.is_file() and seg_o_fixed_path.is_file():
        moving_seg_o = load_array(seg_o_moving_path, RESOLUTION_SHAPES["r1"]).astype(np.int16)
        fixed_seg_o = load_array(seg_o_fixed_path, RESOLUTION_SHAPES["r1"]).astype(np.int16)
        warped_seg_o = warp_segmentation(moving_seg_o, dvf_t, spatial_transformer_r1, device)
        metrics["dice_seg_o"] = dice_for_labels(moving_seg_o, fixed_seg_o, warped_seg_o)

    if seg_b_moving_path.is_file() and seg_b_fixed_path.is_file():
        moving_seg_b = load_array(seg_b_moving_path, RESOLUTION_SHAPES["r1"]).astype(np.int16)
        fixed_seg_b = load_array(seg_b_fixed_path, RESOLUTION_SHAPES["r1"]).astype(np.int16)
        warped_seg_b = warp_segmentation(moving_seg_b, dvf_t, spatial_transformer_r1, device)
        metrics["dice_seg_b"] = dice_for_labels(moving_seg_b, fixed_seg_b, warped_seg_b)

    save_outputs(output_dir, prefix, deformed_np, dvf_np, stage1_deformed_np, warped_seg_o, warped_seg_b, metrics)

    print(f"[INFO] Saved outputs to {output_dir}")
    if "dice_seg_o" in metrics:
        dice_o = metrics["dice_seg_o"]
        print(f"[INFO] seg_o mean Dice: {dice_o['mean_before']:.6f} -> {dice_o['mean_after']:.6f} ({dice_o['mean_delta']:+.6f})")
    if "dice_seg_b" in metrics:
        dice_b = metrics["dice_seg_b"]
        print(f"[INFO] seg_b mean Dice: {dice_b['mean_before']:.6f} -> {dice_b['mean_after']:.6f} ({dice_b['mean_delta']:+.6f})")
    print(f"[INFO] Metrics: {output_dir / f'{prefix}_metrics.json'}")


if __name__ == "__main__":
    main()
