"""Train MUSA+ Stage-3 small-OAR local residual refinement.

Phase-1 prototype from `docs/musa_plus_research_plan.md`:

- freeze trained Stage-1/Stage-2 DIR-MUSA registration models;
- train a lightweight ROI residual U-Net after the composed Stage-2 DVF;
- use rule-based pair difficulty, ROI gating, and anatomy-conditioned maps.
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import musa
from musa.registration_models.musa_plus import LocalResidualUNet


RESOLUTION_SHAPES = {
    "r1": (160, 160, 192),
    "r2": (80, 80, 96),
}


def parse_int_tuple(value: str) -> Tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MUSA+ Stage-3 local refinement on top of frozen two-stage DIR-MUSA."
    )
    parser.add_argument("--trn-list", required=True, help="Training list of case IDs without suffix.")
    parser.add_argument("--val-list", required=True, help="Validation list of case IDs without suffix.")
    parser.add_argument("--vol-path", required=True, help="Path to prepared image .npy files.")
    parser.add_argument("--seg-path-o", required=True, help="Path to prepared multi-label OAR .npy files.")
    parser.add_argument("--seg-path-b", required=True, help="Path to prepared binary bone .npy files.")
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="Optional metadata directory/file from prepare_segrap_case.py for resolving small-OAR names.",
    )
    parser.add_argument(
        "--small-oar-labels",
        default=None,
        help="Comma-separated integer labels for small OARs. Overrides --small-oar-names/metadata.",
    )
    parser.add_argument(
        "--small-oar-names",
        default=",".join(musa.utils_musa_plus.DEFAULT_SMALL_OAR_NAMES),
        help="Comma-separated SegRap structure names used when resolving labels from metadata.",
    )
    parser.add_argument(
        "--model-type",
        default="05dualprnet-v1",
        choices=[
            "01voxelmorph-vf",
            "02resunet-vf",
            "03lkunet-vf-lk09",
            "04transmorph-vf",
            "05dualprnet-vf",
            "01voxelmorph-v1",
            "02resunet-v1",
            "03lkunet-v1-lk09",
            "04transmorph-v1",
            "05dualprnet-v1",
        ],
        help="Frozen Stage-1/Stage-2 registration model type.",
    )
    parser.add_argument("--model-load-stage1", required=True, help="Frozen Stage-1 r2 checkpoint.")
    parser.add_argument("--model-load-stage2", required=True, help="Frozen Stage-2 r1 checkpoint.")
    parser.add_argument("--checkpoint-path", default=None, help="Resume Stage-3 training checkpoint.")
    parser.add_argument("--out-dir", required=True, help="Output directory for Stage-3 checkpoints and logs.")

    parser.add_argument("--lr", type=float, default=1e-4, help="Stage-3 learning rate.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size. Batch size 1 is recommended.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--steps-per-epoch", type=int, default=100, help="Training steps per epoch.")
    parser.add_argument("--epoch-save", type=int, default=10, help="Checkpoint save frequency.")
    parser.add_argument("--epoch-val", type=int, default=10, help="Validation frequency.")
    parser.add_argument("--num-workers", type=int, default=0, help="Training DataLoader workers.")
    parser.add_argument("--gpu", default="0", help="CUDA visible GPU IDs.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training/debugging.")
    parser.add_argument("--cudnn", choices=["det", "ben", "default"], default="ben", help="CUDNN mode.")

    parser.add_argument("--filters", default="8,16,32", help="Stage-3 U-Net filters, e.g. 8,16,32.")
    parser.add_argument(
        "--stage3-input-mode",
        default="full",
        choices=musa.utils_musa_plus.STAGE3_INPUT_MODES,
        help=(
            "Stage-3 information policy: full uses fixed small/bone masks; "
            "no-fixed-small zeros fixed small mask and uses warped-moving ROI; "
            "no-fixed-seg also zeros fixed bone and uses CT-only difficulty."
        ),
    )
    parser.add_argument("--roi-radius-min", type=int, default=3, help="Minimum small-OAR ROI dilation radius.")
    parser.add_argument("--roi-radius-max", type=int, default=8, help="Maximum small-OAR ROI dilation radius.")
    parser.add_argument("--roi-smooth-steps", type=int, default=2, help="Average-pool smoothing passes for ROI gate.")
    parser.add_argument("--residual-scale-min", type=float, default=0.25, help="Residual scale for easiest pairs.")
    parser.add_argument("--residual-scale-max", type=float, default=1.00, help="Residual scale for hardest pairs.")

    parser.add_argument("--lambda-local-img", type=float, default=1.0, help="Local image MSE weight.")
    parser.add_argument("--lambda-small-min", type=float, default=1.0, help="Small-OAR Dice weight for easiest pairs.")
    parser.add_argument("--lambda-small-max", type=float, default=3.0, help="Small-OAR Dice weight for hardest pairs.")
    parser.add_argument("--lambda-smooth", type=float, default=0.10, help="Base weighted smoothness weight.")
    parser.add_argument("--lambda-smooth-extra", type=float, default=0.20, help="Extra smoothness weight at difficulty=1.")
    parser.add_argument("--lambda-mag", type=float, default=0.01, help="Weighted residual magnitude penalty.")
    parser.add_argument("--lambda-jacobian", type=float, default=0.0, help="Low-Jacobian/folding penalty weight.")
    parser.add_argument("--jacobian-margin", type=float, default=0.05, help="Minimum desired Jacobian determinant margin.")
    parser.add_argument("--jacobian-roi-weight", type=float, default=5.0, help="Extra Jacobian penalty weight inside ROI.")
    parser.add_argument("--lambda-preserve-large", type=float, default=0.50, help="Large-OAR preservation weight.")
    parser.add_argument("--lambda-preserve-bone", type=float, default=0.50, help="Bone preservation weight.")
    parser.add_argument(
        "--best-policy",
        default="noharm",
        choices=["noharm", "small-final"],
        help="Which validation policy writes best_stage3.pth. noharm penalizes folding and large/bone degradation.",
    )
    parser.add_argument("--best-jacobian-penalty", type=float, default=5.0, help="Penalty for ROI Jacobian <= 0 ratio.")
    parser.add_argument("--best-large-drop-penalty", type=float, default=2.0, help="Penalty for negative large-OAR delta.")
    parser.add_argument("--best-bone-drop-penalty", type=float, default=2.0, help="Penalty for negative bone delta.")
    parser.add_argument("--best-residual-p95-penalty", type=float, default=0.0, help="Optional penalty for ROI residual p95.")
    return parser.parse_args()


class Stage3PairDataset(Dataset):
    """Load image/OAR/bone pairs for Stage-3 training or validation."""

    def __init__(
        self,
        case_ids: Sequence[str],
        path_vol: str,
        path_seg_o: str,
        path_seg_b: str,
        mode: str,
        ftype: str = ".npy",
    ) -> None:
        self.case_ids = list(case_ids)
        self.path_vol = path_vol
        self.path_seg_o = path_seg_o
        self.path_seg_b = path_seg_b
        self.ftype = ftype
        if mode == "train":
            self.index_pair = list(itertools.product(self.case_ids, repeat=2))
        elif mode == "val":
            n = len(self.case_ids) // 2
            self.index_pair = [(self.case_ids[i], self.case_ids[i + n]) for i in range(n)]
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        if not self.index_pair:
            raise ValueError(f"No {mode} pairs could be built from {len(self.case_ids)} case IDs")

    def __len__(self) -> int:
        return len(self.index_pair)

    def _load_pair(self, idx: int, folder: str) -> Tuple[np.ndarray, np.ndarray]:
        moving_id, fixed_id = self.index_pair[idx]
        moving = musa.utils_dataloader.load_vol(moving_id + self.ftype, folder)
        fixed = musa.utils_dataloader.load_vol(fixed_id + self.ftype, folder)
        return moving, fixed

    def __getitem__(self, idx: int):
        moving_id, fixed_id = self.index_pair[idx]
        moving, fixed = self._load_pair(idx, self.path_vol)
        moving_seg_o, fixed_seg_o = self._load_pair(idx, self.path_seg_o)
        moving_seg_b, fixed_seg_b = self._load_pair(idx, self.path_seg_b)

        return (
            torch.from_numpy(moving).float(),
            torch.from_numpy(fixed).float(),
            torch.from_numpy(moving_seg_o).long(),
            torch.from_numpy(fixed_seg_o).long(),
            torch.from_numpy(moving_seg_b).long(),
            torch.from_numpy(fixed_seg_b).long(),
            moving_id,
            fixed_id,
        )


def configure_runtime(args: argparse.Namespace) -> torch.device:
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.cudnn == "det":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif args.cudnn == "ben":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))


def infer_metadata_path(args: argparse.Namespace) -> str:
    if args.metadata_path:
        return args.metadata_path
    candidate = Path(args.vol_path).resolve().parent / "metadata"
    return str(candidate) if candidate.exists() else ""


def load_registration_model(
    model_type: str,
    resolution: str,
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    model = musa.utils_model_zoo.get_model_v1(
        inshape=RESOLUTION_SHAPES[resolution],
        model_type=model_type,
        model_resolution=resolution,
    )
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(musa.utils_musa_plus.checkpoint_to_state_dict(payload))
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def run_two_stage_frozen(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    model_stage1: torch.nn.Module,
    model_stage2: torch.nn.Module,
    model_type: str,
    spatial_transformer_r1: torch.nn.Module,
    composer_r1: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model_type = musa.utils_model_zoo.normalize_model_type(model_type)
    flag_pad = model_type.startswith("04transmorph-v1")

    with torch.no_grad():
        moving_r2 = musa.utils_warp.vol_downsamplex2(moving)
        fixed_r2 = musa.utils_warp.vol_downsamplex2(fixed)
        inputs_stage1 = (moving_r2, fixed_r2)
        if not flag_pad:
            _, dvf_r2_stage1 = musa.utils_model_zoo.model_register_v1(inputs_stage1, model_stage1, model_type)
        else:
            pad_size = (16, 16, 24, 24, 24, 24)
            padded_stage1 = [torch.nn.functional.pad(d, pad=pad_size) for d in inputs_stage1]
            _, dvf_r2_stage1 = musa.utils_model_zoo.model_register_v1(padded_stage1, model_stage1, model_type)
            dvf_r2_stage1 = dvf_r2_stage1[..., 24:24 + 80, 24:24 + 80, 16:16 + 96]

        dvf_r1_stage1 = musa.utils_warp.dvf_upsample(dvf_r2_stage1)
        deformed_stage1 = spatial_transformer_r1(moving, dvf_r1_stage1, mode="bilinear")
        _, dvf_r1_stage2 = musa.utils_model_zoo.model_register_v1((deformed_stage1, fixed), model_stage2, model_type)
        dvf_stage2 = composer_r1(dvf_r1_stage1, dvf_r1_stage2)
        deformed_stage2 = spatial_transformer_r1(moving, dvf_stage2, mode="bilinear")
    return deformed_stage2.detach(), dvf_stage2.detach()


def batch_to_device(batch, device: torch.device):
    tensors = [item.to(device) for item in batch[:6]]
    moving_ids = batch[6]
    fixed_ids = batch[7]
    return (*tensors, moving_ids, fixed_ids)


def stage3_forward(
    batch,
    model_stage1: torch.nn.Module,
    model_stage2: torch.nn.Module,
    model_stage3: torch.nn.Module,
    model_type: str,
    spatial_transformer_r1: torch.nn.Module,
    composer_r1: torch.nn.Module,
    small_oar_labels: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
    compute_jacobian: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    moving, fixed, moving_seg_o, fixed_seg_o, moving_seg_b, fixed_seg_b, _, _ = batch_to_device(batch, device)

    moving_oar = musa.utils_musa_plus.seg_to_foreground_mask(moving_seg_o)
    fixed_oar = musa.utils_musa_plus.seg_to_foreground_mask(fixed_seg_o)
    moving_small = musa.utils_musa_plus.seg_to_label_mask(moving_seg_o, small_oar_labels)
    fixed_small = musa.utils_musa_plus.seg_to_label_mask(fixed_seg_o, small_oar_labels)
    moving_large = (moving_oar - moving_small).clamp(0.0, 1.0)
    fixed_large = (fixed_oar - fixed_small).clamp(0.0, 1.0)
    moving_bone = (moving_seg_b > 0).float()
    fixed_bone = (fixed_seg_b > 0).float()

    deformed_stage2, dvf_stage2 = run_two_stage_frozen(
        moving=moving,
        fixed=fixed,
        model_stage1=model_stage1,
        model_stage2=model_stage2,
        model_type=model_type,
        spatial_transformer_r1=spatial_transformer_r1,
        composer_r1=composer_r1,
    )

    warped_small_stage2 = spatial_transformer_r1(moving_small, dvf_stage2, mode="bilinear").clamp(0.0, 1.0)
    warped_large_stage2 = spatial_transformer_r1(moving_large, dvf_stage2, mode="bilinear").clamp(0.0, 1.0)
    warped_bone_stage2 = spatial_transformer_r1(moving_bone, dvf_stage2, mode="bilinear").clamp(0.0, 1.0)

    conditioning = musa.utils_musa_plus.stage3_conditioning_masks(
        input_mode=args.stage3_input_mode,
        fixed_small_mask=fixed_small,
        warped_small_mask_stage2=warped_small_stage2,
        fixed_bone_mask=fixed_bone,
    )
    if args.stage3_input_mode == "no-fixed-seg":
        difficulty = musa.utils_musa_plus.estimate_pair_difficulty_ct_only(
            moving=moving,
            fixed=fixed,
            deformed_stage2=deformed_stage2,
            dvf_stage2=dvf_stage2,
        )
    else:
        stage2_small_roi = torch.maximum(fixed_small, warped_small_stage2.detach())
        difficulty = musa.utils_musa_plus.estimate_stage2_pair_difficulty(
            fixed=fixed,
            deformed_stage2=deformed_stage2,
            dvf_stage2=dvf_stage2,
            warped_small_mask_stage2=warped_small_stage2,
            fixed_small_mask=fixed_small,
            warped_bone_mask_stage2=warped_bone_stage2,
            fixed_bone_mask=fixed_bone,
            image_mask=stage2_small_roi,
        )
    roi_radius = musa.utils_musa_plus.difficulty_to_radius_per_batch(
        difficulty,
        radius_min=args.roi_radius_min,
        radius_max=args.roi_radius_max,
    )
    roi_gate = musa.utils_musa_plus.build_roi_gate_per_batch(
        conditioning["roi_source"],
        radii=roi_radius,
        smooth_steps=args.roi_smooth_steps,
    )

    stage3_inputs = musa.utils_musa_plus.make_stage3_inputs(
        fixed=fixed,
        deformed_stage2=deformed_stage2,
        fixed_small_mask=conditioning["fixed_small_feature"],
        warped_small_mask_stage2=warped_small_stage2,
        dvf_stage2=dvf_stage2,
        fixed_bone_mask=conditioning["fixed_bone_feature"],
        warped_bone_mask_stage2=warped_bone_stage2,
    )
    raw_local_dvf = model_stage3(stage3_inputs)
    residual_scale = musa.utils_musa_plus.difficulty_to_value(
        difficulty,
        args.residual_scale_min,
        args.residual_scale_max,
    ).view(-1, 1, 1, 1, 1)
    local_dvf = raw_local_dvf * residual_scale
    gated_local_dvf = local_dvf * roi_gate
    dvf_final = dvf_stage2 + gated_local_dvf

    deformed_final = spatial_transformer_r1(moving, dvf_final, mode="bilinear")
    warped_small_final = spatial_transformer_r1(moving_small, dvf_final, mode="bilinear").clamp(0.0, 1.0)
    warped_large_final = spatial_transformer_r1(moving_large, dvf_final, mode="bilinear").clamp(0.0, 1.0)
    warped_bone_final = spatial_transformer_r1(moving_bone, dvf_final, mode="bilinear").clamp(0.0, 1.0)

    anatomy_maps = musa.utils_musa_plus.build_anatomy_maps(
        fixed_bone_mask=conditioning["anatomy_bone"],
        roi_gate=roi_gate,
        difficulty=difficulty,
    )

    lambda_small = musa.utils_musa_plus.difficulty_to_value(
        difficulty,
        args.lambda_small_min,
        args.lambda_small_max,
    )
    lambda_smooth = args.lambda_smooth + args.lambda_smooth_extra * difficulty

    loss_local_img_per_pair = musa.utils_musa_plus.masked_mse_loss_per_batch(deformed_final, fixed, roi_gate)
    loss_small_per_pair = musa.utils_musa_plus.binary_dice_loss_per_batch(warped_small_final, fixed_small)
    loss_smooth_per_pair = musa.utils_musa_plus.weighted_gradient_loss_per_batch(gated_local_dvf, anatomy_maps["smooth"])
    loss_mag_per_pair = musa.utils_musa_plus.weighted_magnitude_loss_per_batch(local_dvf, anatomy_maps["magnitude"])
    loss_jacobian_per_pair = musa.utils_musa_plus.jacobian_hinge_loss_per_batch(
        dvf_final,
        roi_gate=roi_gate,
        margin=args.jacobian_margin,
        roi_weight=args.jacobian_roi_weight,
    )
    loss_preserve_large_per_pair = musa.utils_musa_plus.binary_dice_loss_per_batch(
        warped_large_final,
        warped_large_stage2.detach(),
    )
    loss_preserve_bone_per_pair = musa.utils_musa_plus.binary_dice_loss_per_batch(
        warped_bone_final,
        warped_bone_stage2.detach(),
    )

    loss_per_pair = (
        args.lambda_local_img * loss_local_img_per_pair
        + lambda_small * loss_small_per_pair
        + lambda_smooth * loss_smooth_per_pair
        + args.lambda_mag * loss_mag_per_pair
        + args.lambda_jacobian * loss_jacobian_per_pair
        + args.lambda_preserve_large * loss_preserve_large_per_pair
        + args.lambda_preserve_bone * loss_preserve_bone_per_pair
    )
    loss = loss_per_pair.mean()

    loss_local_img = loss_local_img_per_pair.mean()
    loss_small = loss_small_per_pair.mean()
    loss_smooth = loss_smooth_per_pair.mean()
    loss_mag = loss_mag_per_pair.mean()
    loss_jacobian = loss_jacobian_per_pair.mean()
    loss_preserve_large = loss_preserve_large_per_pair.mean()
    loss_preserve_bone = loss_preserve_bone_per_pair.mean()

    with torch.no_grad():
        small_stage2_dice = musa.utils_musa_plus.binary_dice_per_batch(warped_small_stage2, fixed_small).mean()
        small_final_dice = musa.utils_musa_plus.binary_dice_per_batch(warped_small_final, fixed_small).mean()
        large_stage2_dice = musa.utils_musa_plus.binary_dice_per_batch(warped_large_stage2, fixed_large).mean()
        large_final_dice = musa.utils_musa_plus.binary_dice_per_batch(warped_large_final, fixed_large).mean()
        bone_stage2_dice = musa.utils_musa_plus.binary_dice_per_batch(warped_bone_stage2, fixed_bone).mean()
        bone_final_dice = musa.utils_musa_plus.binary_dice_per_batch(warped_bone_final, fixed_bone).mean()
        residual_stats = musa.utils_musa_plus.magnitude_stats(gated_local_dvf, roi_gate)
        if compute_jacobian:
            stage2_jac = musa.utils_musa_plus.jacobian_stats(dvf_stage2, roi_gate)
            final_jac = musa.utils_musa_plus.jacobian_stats(dvf_final, roi_gate)
        else:
            stage2_jac = {
                "global_nonpos_ratio": 0.0,
                "roi_nonpos_ratio": 0.0,
                "roi_min": 0.0,
            }
            final_jac = {
                "global_nonpos_ratio": 0.0,
                "roi_nonpos_ratio": 0.0,
                "roi_min": 0.0,
            }

    metrics = {
        "loss": float(loss.detach().cpu()),
        "loss_local_img": float(loss_local_img.detach().cpu()),
        "loss_small": float(loss_small.detach().cpu()),
        "loss_smooth": float(loss_smooth.detach().cpu()),
        "loss_mag": float(loss_mag.detach().cpu()),
        "loss_jacobian": float(loss_jacobian.detach().cpu()),
        "loss_preserve_large": float(loss_preserve_large.detach().cpu()),
        "loss_preserve_bone": float(loss_preserve_bone.detach().cpu()),
        "difficulty": float(difficulty.mean().detach().cpu()),
        "roi_radius": float(roi_radius.float().mean().detach().cpu()),
        "roi_radius_min": float(roi_radius.float().min().detach().cpu()),
        "roi_radius_max": float(roi_radius.float().max().detach().cpu()),
        "residual_scale": float(residual_scale.mean().detach().cpu()),
        "lambda_small": float(lambda_small.mean().detach().cpu()),
        "lambda_smooth": float(lambda_smooth.mean().detach().cpu()),
        "small_stage2_dice": float(small_stage2_dice.detach().cpu()),
        "small_final_dice": float(small_final_dice.detach().cpu()),
        "small_delta": float((small_final_dice - small_stage2_dice).detach().cpu()),
        "large_stage2_dice": float(large_stage2_dice.detach().cpu()),
        "large_final_dice": float(large_final_dice.detach().cpu()),
        "large_delta": float((large_final_dice - large_stage2_dice).detach().cpu()),
        "bone_stage2_dice": float(bone_stage2_dice.detach().cpu()),
        "bone_final_dice": float(bone_final_dice.detach().cpu()),
        "bone_delta": float((bone_final_dice - bone_stage2_dice).detach().cpu()),
        "residual_mag_mean": residual_stats["global_mean"],
        "residual_mag_p95": residual_stats["global_p95"],
        "residual_mag_max": residual_stats["global_max"],
        "residual_mag_roi_mean": residual_stats["roi_mean"],
        "residual_mag_roi_p95": residual_stats["roi_p95"],
        "residual_mag_roi_max": residual_stats["roi_max"],
        "stage2_jac_nonpos": stage2_jac["global_nonpos_ratio"],
        "stage2_jac_roi_nonpos": stage2_jac["roi_nonpos_ratio"],
        "stage2_jac_roi_min": stage2_jac["roi_min"],
        "final_jac_nonpos": final_jac["global_nonpos_ratio"],
        "final_jac_roi_nonpos": final_jac["roi_nonpos_ratio"],
        "final_jac_roi_min": final_jac["roi_min"],
    }
    return loss, metrics


def mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def format_metrics(prefix: str, epoch: int, epochs: int, metrics: Dict[str, float], seconds: float) -> str:
    return (
        f"{prefix} Epoch {epoch}/{epochs} - "
        f"{seconds:.2f}s - "
        f"loss {metrics['loss']:.4e} - "
        f"small {metrics['small_stage2_dice']:.4f}->{metrics['small_final_dice']:.4f} "
        f"({metrics['small_delta']:+.4f}) - "
        f"large {metrics['large_stage2_dice']:.4f}->{metrics['large_final_dice']:.4f} "
        f"({metrics['large_delta']:+.4f}) - "
        f"bone {metrics['bone_stage2_dice']:.4f}->{metrics['bone_final_dice']:.4f} "
        f"({metrics['bone_delta']:+.4f}) - "
        f"jac_roi<=0 {metrics.get('final_jac_roi_nonpos', 0.0):.3e} - "
        f"difficulty {metrics['difficulty']:.3f} - roi {metrics['roi_radius']:.1f}"
    )


def noharm_score(metrics: Dict[str, float], args: argparse.Namespace) -> float:
    """Validation score that rewards small-OAR improvement and penalizes harm."""

    large_drop = max(0.0, -float(metrics["large_delta"]))
    bone_drop = max(0.0, -float(metrics["bone_delta"]))
    jac_roi = max(0.0, float(metrics.get("final_jac_roi_nonpos", 0.0)))
    residual_p95 = max(0.0, float(metrics.get("residual_mag_roi_p95", 0.0)))
    return (
        float(metrics["small_delta"])
        - args.best_large_drop_penalty * large_drop
        - args.best_bone_drop_penalty * bone_drop
        - args.best_jacobian_penalty * jac_roi
        - args.best_residual_p95_penalty * residual_p95
    )


def selection_score(metrics: Dict[str, float], args: argparse.Namespace) -> float:
    if args.best_policy == "small-final":
        return float(metrics["small_final_dice"])
    return noharm_score(metrics, args)


def save_checkpoint(
    path: Path,
    epoch: int,
    model_stage3: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    small_oar_labels: Sequence[int],
    best_val_small_dice: float,
    best_val_noharm_score: float,
    best_val_selection_score: float,
    history: Dict[str, List[Dict[str, float]]],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_stage3.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "small_oar_labels": list(small_oar_labels),
        "best_val_small_dice": best_val_small_dice,
        "best_val_noharm_score": best_val_noharm_score,
        "best_val_selection_score": best_val_selection_score,
        "best_policy": args.best_policy,
        "history": history,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()
    device = configure_runtime(args)
    model_type = musa.utils_model_zoo.normalize_model_type(args.model_type)

    metadata_path = infer_metadata_path(args)
    small_oar_labels = musa.utils_musa_plus.resolve_small_oar_labels(
        small_oar_labels=args.small_oar_labels,
        small_oar_names=args.small_oar_names,
        metadata_path=metadata_path or None,
    )
    if not small_oar_labels:
        raise ValueError(
            "Could not resolve small-OAR labels. Provide --small-oar-labels or --metadata-path "
            "with prepare_segrap_case.py metadata."
        )

    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print("[INFO] MUSA+ Stage-3 training")
    print(f"[INFO] device={device}, model_type={model_type}")
    print(f"[INFO] output={out_dir}")
    print(f"[INFO] metadata_path={metadata_path or '<none>'}")
    print(f"[INFO] small_oar_labels={small_oar_labels}")
    print(f"[INFO] stage3_input_mode={args.stage3_input_mode}")

    trn_files = musa.utils_dataloader.read_file_list(args.trn_list)
    val_files = musa.utils_dataloader.read_file_list(args.val_list)
    trn_dataset = Stage3PairDataset(trn_files, args.vol_path, args.seg_path_o, args.seg_path_b, mode="train")
    val_dataset = Stage3PairDataset(val_files, args.vol_path, args.seg_path_o, args.seg_path_b, mode="val")
    trn_loader = DataLoader(
        trn_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    spatial_transformer_r1 = musa.utils_warp.SpatialTransformer(RESOLUTION_SHAPES["r1"]).to(device)
    composer_r1 = musa.utils_warp.ComposeDVF(RESOLUTION_SHAPES["r1"]).to(device)
    model_stage1 = load_registration_model(model_type, "r2", args.model_load_stage1, device)
    model_stage2 = load_registration_model(model_type, "r1", args.model_load_stage2, device)

    filters = parse_int_tuple(args.filters)
    model_stage3 = LocalResidualUNet(in_channels=7, out_channels=3, filters=filters).to(device)
    optimizer = torch.optim.Adam(model_stage3.parameters(), lr=args.lr)
    epoch_start = 0
    best_val_small_dice = 0.0
    best_val_noharm_score = float("-inf")
    best_val_selection_score = float("-inf")
    history: Dict[str, List[Dict[str, float]]] = {"train": [], "val": []}

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model_stage3.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_start = int(checkpoint["epoch"]) + 1
        best_val_small_dice = float(checkpoint.get("best_val_small_dice", 0.0))
        best_val_noharm_score = float(checkpoint.get("best_val_noharm_score", float("-inf")))
        best_val_selection_score = float(checkpoint.get("best_val_selection_score", float("-inf")))
        history = checkpoint.get("history", history)
        print(f"[INFO] Resumed Stage-3 checkpoint {args.checkpoint_path} at epoch {epoch_start}")

    for epoch in range(epoch_start, args.epochs):
        model_stage3.train()
        train_start = time.time()
        train_rows: List[Dict[str, float]] = []
        for step, batch in enumerate(trn_loader):
            if step >= args.steps_per_epoch:
                break
            optimizer.zero_grad()
            loss, metrics = stage3_forward(
                batch=batch,
                model_stage1=model_stage1,
                model_stage2=model_stage2,
                model_stage3=model_stage3,
                model_type=model_type,
                spatial_transformer_r1=spatial_transformer_r1,
                composer_r1=composer_r1,
                small_oar_labels=small_oar_labels,
                args=args,
                device=device,
            )
            loss.backward()
            optimizer.step()
            train_rows.append(metrics)

        train_metrics = mean_metrics(train_rows)
        history["train"].append(train_metrics)
        print(format_metrics("Train", epoch + 1, args.epochs, train_metrics, time.time() - train_start), flush=True)

        if (epoch + 1) % args.epoch_val == 0:
            model_stage3.eval()
            val_start = time.time()
            val_rows: List[Dict[str, float]] = []
            with torch.no_grad():
                for batch in val_loader:
                    _, metrics = stage3_forward(
                        batch=batch,
                        model_stage1=model_stage1,
                        model_stage2=model_stage2,
                        model_stage3=model_stage3,
                        model_type=model_type,
                        spatial_transformer_r1=spatial_transformer_r1,
                        composer_r1=composer_r1,
                        small_oar_labels=small_oar_labels,
                        args=args,
                        device=device,
                        compute_jacobian=True,
                    )
                    val_rows.append(metrics)
            val_metrics = mean_metrics(val_rows)
            history["val"].append(val_metrics)
            print(format_metrics("Val", epoch + 1, args.epochs, val_metrics, time.time() - val_start), flush=True)
            val_noharm_score = noharm_score(val_metrics, args)
            val_selection_score = selection_score(val_metrics, args)
            print(
                f"[INFO] Val selection: policy={args.best_policy}, "
                f"score={val_selection_score:.6f}, noharm={val_noharm_score:.6f}",
                flush=True,
            )

            if val_metrics["small_final_dice"] > best_val_small_dice:
                best_val_small_dice = val_metrics["small_final_dice"]
                save_checkpoint(
                    out_dir / "best_stage3_small.pth",
                    epoch,
                    model_stage3,
                    optimizer,
                    args,
                    small_oar_labels,
                    best_val_small_dice,
                    best_val_noharm_score,
                    best_val_selection_score,
                    history,
                )
                print(f"[INFO] Saved best small-Dice Stage-3 checkpoint with small Dice {best_val_small_dice:.4f}")

            if val_noharm_score > best_val_noharm_score:
                best_val_noharm_score = val_noharm_score
                save_checkpoint(
                    out_dir / "best_stage3_noharm.pth",
                    epoch,
                    model_stage3,
                    optimizer,
                    args,
                    small_oar_labels,
                    best_val_small_dice,
                    best_val_noharm_score,
                    best_val_selection_score,
                    history,
                )
                print(f"[INFO] Saved best no-harm Stage-3 checkpoint with score {best_val_noharm_score:.6f}")

            if val_selection_score > best_val_selection_score:
                best_val_selection_score = val_selection_score
                save_checkpoint(
                    out_dir / "best_stage3.pth",
                    epoch,
                    model_stage3,
                    optimizer,
                    args,
                    small_oar_labels,
                    best_val_small_dice,
                    best_val_noharm_score,
                    best_val_selection_score,
                    history,
                )
                print(f"[INFO] Saved selected best Stage-3 checkpoint with score {best_val_selection_score:.6f}")

        if (epoch + 1) % args.epoch_save == 0:
            checkpoint_path = checkpoint_dir / f"{epoch + 1:04d}.pth"
            save_checkpoint(
                checkpoint_path,
                epoch,
                model_stage3,
                optimizer,
                args,
                small_oar_labels,
                best_val_small_dice,
                best_val_noharm_score,
                best_val_selection_score,
                history,
            )
            print(f"[INFO] Checkpoint saved to {checkpoint_path}")

    save_checkpoint(
        out_dir / "final_stage3.pth",
        args.epochs - 1,
        model_stage3,
        optimizer,
        args,
        small_oar_labels,
        best_val_small_dice,
        best_val_noharm_score,
        best_val_selection_score,
        history,
    )
    print(f"[INFO] Final Stage-3 checkpoint saved to {out_dir / 'final_stage3.pth'}")


if __name__ == "__main__":
    main()
