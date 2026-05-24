"""Utilities for the MUSA+ Stage-3 local refinement prototype."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_SMALL_OAR_NAMES: Tuple[str, ...] = (
    "OpticNerve_L",
    "OpticNerve_R",
    "Cochlea_L",
    "Cochlea_R",
    "Lens_L",
    "Lens_R",
    "Pituitary",
    "Chiasm",
    "IAC_L",
    "IAC_R",
    "MiddleEar_L",
    "MiddleEar_R",
    "TympanicCavity_L",
    "TympanicCavity_R",
    "VestibulSemi_L",
    "VestibulSemi_R",
)

STAGE3_INPUT_MODES: Tuple[str, ...] = (
    "full",
    "no-fixed-small",
    "no-fixed-seg",
)


def parse_label_list(value: Optional[str]) -> List[int]:
    """Parse a comma-separated label list into sorted positive integers."""

    if value is None or value.strip() == "":
        return []
    labels = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        label = int(text)
        if label <= 0:
            raise ValueError(f"Small-OAR labels must be positive, got {label}")
        labels.append(label)
    return sorted(set(labels))


def parse_name_list(value: Optional[str]) -> List[str]:
    """Parse a comma-separated name list."""

    if value is None or value.strip() == "":
        return list(DEFAULT_SMALL_OAR_NAMES)
    return [part.strip() for part in value.split(",") if part.strip()]


def _metadata_files(metadata_path: Path) -> List[Path]:
    if metadata_path.is_file():
        return [metadata_path]
    if metadata_path.is_dir():
        return sorted(metadata_path.glob("*.json"))
    return []


def resolve_small_oar_labels(
    small_oar_labels: Optional[str] = None,
    small_oar_names: Optional[str] = None,
    metadata_path: Optional[str] = None,
) -> List[int]:
    """Resolve small-OAR labels from explicit labels or SegRap metadata.

    Explicit labels take precedence. When labels are omitted, this reads
    `metadata/*.json` files emitted by `prepare_segrap_case.py` and maps known
    small-OAR structure names to their integer labels.
    """

    labels = parse_label_list(small_oar_labels)
    if labels:
        return labels

    if metadata_path is None:
        return []

    names = set(parse_name_list(small_oar_names))
    resolved = set()
    for path in _metadata_files(Path(metadata_path)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        label_map: Dict[str, int] = payload.get("label_map", {})
        for name in names:
            if name in label_map:
                resolved.add(int(label_map[name]))
    return sorted(label for label in resolved if label > 0)


def seg_to_label_mask(seg: torch.Tensor, labels: Sequence[int]) -> torch.Tensor:
    """Convert an integer segmentation to a binary mask for selected labels."""

    if seg.ndim == 4:
        seg = seg.unsqueeze(1)
    if seg.ndim != 5 or seg.shape[1] != 1:
        raise ValueError(f"Expected segmentation shape (B,1,H,W,D) or (B,H,W,D), got {tuple(seg.shape)}")

    mask = torch.zeros_like(seg, dtype=torch.bool)
    seg_long = seg.long()
    for label in labels:
        mask = torch.logical_or(mask, seg_long == int(label))
    return mask.float()


def seg_to_foreground_mask(seg: torch.Tensor) -> torch.Tensor:
    """Return all non-background labels as a binary mask."""

    if seg.ndim == 4:
        seg = seg.unsqueeze(1)
    if seg.ndim != 5 or seg.shape[1] != 1:
        raise ValueError(f"Expected segmentation shape (B,1,H,W,D) or (B,H,W,D), got {tuple(seg.shape)}")
    return (seg > 0).float()


def binary_dice_per_batch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Soft binary Dice score for each batch item."""

    pred = pred.float()
    target = target.float()
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    denominator = pred.sum(dim=dims) + target.sum(dim=dims)
    empty = denominator <= eps
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return torch.where(empty, torch.ones_like(dice), dice)


def binary_dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Mean soft binary Dice loss."""

    return 1.0 - binary_dice_per_batch(pred, target, eps=eps).mean()


def binary_dice_loss_per_batch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Soft binary Dice loss for each batch item."""

    return 1.0 - binary_dice_per_batch(pred, target, eps=eps)


def estimate_pair_difficulty(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    moving_oar_mask: torch.Tensor,
    fixed_oar_mask: torch.Tensor,
    moving_bone_mask: torch.Tensor,
    fixed_bone_mask: torch.Tensor,
    weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> torch.Tensor:
    """Rule-based pair difficulty score in [0, 1].

    The first Phase-1 version follows the research plan:

    `w1 * (1 - initial_bone_dice) + w2 * (1 - initial_oar_dice) + w3 * image_mse`.
    """

    bone_dice = binary_dice_per_batch(moving_bone_mask, fixed_bone_mask)
    oar_dice = binary_dice_per_batch(moving_oar_mask, fixed_oar_mask)
    image_mse = (moving.float() - fixed.float()).pow(2).flatten(1).mean(dim=1)
    image_mse = image_mse.clamp(0.0, 1.0)

    w_bone, w_oar, w_img = weights
    difficulty = w_bone * (1.0 - bone_dice) + w_oar * (1.0 - oar_dice) + w_img * image_mse
    return difficulty.clamp(0.0, 1.0)


def estimate_pair_difficulty_ct_only(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    deformed_stage2: Optional[torch.Tensor] = None,
    dvf_stage2: Optional[torch.Tensor] = None,
    weights: Tuple[float, float, float] = (0.35, 0.45, 0.20),
) -> torch.Tensor:
    """Rule-based difficulty that does not use fixed/moving segmentations."""

    initial_mse = (moving.float() - fixed.float()).pow(2).flatten(1).mean(dim=1).clamp(0.0, 1.0)
    if deformed_stage2 is None:
        stage2_mse = initial_mse
    else:
        stage2_mse = (deformed_stage2.float() - fixed.float()).pow(2).flatten(1).mean(dim=1).clamp(0.0, 1.0)
    if dvf_stage2 is None:
        flow_score = torch.zeros_like(initial_mse)
    else:
        flow_mag = torch.sqrt(dvf_stage2.float().pow(2).sum(dim=1)).flatten(1)
        flow_score = (torch.quantile(flow_mag, q=0.95, dim=1) / 20.0).clamp(0.0, 1.0)

    w_initial, w_stage2, w_flow = weights
    difficulty = w_initial * initial_mse + w_stage2 * stage2_mse + w_flow * flow_score
    return difficulty.clamp(0.0, 1.0)


def estimate_stage2_pair_difficulty(
    fixed: torch.Tensor,
    deformed_stage2: torch.Tensor,
    dvf_stage2: torch.Tensor,
    warped_small_mask_stage2: torch.Tensor,
    fixed_small_mask: torch.Tensor,
    warped_bone_mask_stage2: torch.Tensor,
    fixed_bone_mask: torch.Tensor,
    image_mask: Optional[torch.Tensor] = None,
    weights: Tuple[float, float, float, float] = (0.45, 0.20, 0.25, 0.10),
    flow_scale: float = 20.0,
) -> torch.Tensor:
    """Pair difficulty after the frozen Stage-2 registration.

    This score focuses Stage-3 on the residual problem it actually has to
    solve: poor small-OAR overlap after Stage 2, residual bone mismatch,
    local image disagreement, and unusually large Stage-2 flow.
    """

    small_dice = binary_dice_per_batch(warped_small_mask_stage2, fixed_small_mask)
    bone_dice = binary_dice_per_batch(warped_bone_mask_stage2, fixed_bone_mask)
    if image_mask is None:
        image_residual = (deformed_stage2.float() - fixed.float()).pow(2).flatten(1).mean(dim=1)
    else:
        image_residual = masked_mse_loss_per_batch(deformed_stage2, fixed, image_mask)
    image_residual = image_residual.clamp(0.0, 1.0)

    flow_mag = torch.sqrt(dvf_stage2.float().pow(2).sum(dim=1)).flatten(1)
    flow_score = (torch.quantile(flow_mag, q=0.95, dim=1) / max(flow_scale, 1e-6)).clamp(0.0, 1.0)

    w_small, w_bone, w_img, w_flow = weights
    difficulty = (
        w_small * (1.0 - small_dice)
        + w_bone * (1.0 - bone_dice)
        + w_img * image_residual
        + w_flow * flow_score
    )
    return difficulty.clamp(0.0, 1.0)


def estimate_pair_difficulty_by_mode(
    input_mode: str,
    moving: torch.Tensor,
    fixed: torch.Tensor,
    moving_oar_mask: torch.Tensor,
    fixed_oar_mask: torch.Tensor,
    moving_bone_mask: torch.Tensor,
    fixed_bone_mask: torch.Tensor,
    deformed_stage2: Optional[torch.Tensor] = None,
    dvf_stage2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Estimate pair difficulty using the selected Stage-3 information policy."""

    if input_mode not in STAGE3_INPUT_MODES:
        raise ValueError(f"Unsupported Stage-3 input mode {input_mode!r}; choices={STAGE3_INPUT_MODES}")
    if input_mode == "no-fixed-seg":
        return estimate_pair_difficulty_ct_only(
            moving=moving,
            fixed=fixed,
            deformed_stage2=deformed_stage2,
            dvf_stage2=dvf_stage2,
        )
    return estimate_pair_difficulty(
        moving=moving,
        fixed=fixed,
        moving_oar_mask=moving_oar_mask,
        fixed_oar_mask=fixed_oar_mask,
        moving_bone_mask=moving_bone_mask,
        fixed_bone_mask=fixed_bone_mask,
    )


def difficulty_to_value(difficulty: torch.Tensor, value_min: float, value_max: float) -> torch.Tensor:
    """Linearly map difficulty in [0, 1] to `[value_min, value_max]`."""

    return value_min + difficulty.float().clamp(0.0, 1.0) * (value_max - value_min)


def difficulty_to_radius_per_batch(difficulty: torch.Tensor, radius_min: int, radius_max: int) -> torch.Tensor:
    """Map each batch item's difficulty to an integer ROI dilation radius."""

    if radius_min < 0 or radius_max < radius_min:
        raise ValueError(f"Invalid radius range: {radius_min}..{radius_max}")
    values = difficulty_to_value(difficulty.flatten(), float(radius_min), float(radius_max))
    return torch.round(values).long().clamp(min=radius_min, max=radius_max)


def difficulty_to_radius(difficulty: torch.Tensor, radius_min: int, radius_max: int) -> int:
    """Map mean batch difficulty to an integer ROI dilation radius."""

    if radius_min < 0 or radius_max < radius_min:
        raise ValueError(f"Invalid radius range: {radius_min}..{radius_max}")
    value = difficulty_to_value(difficulty.mean(), float(radius_min), float(radius_max))
    return int(torch.round(value).item())


def build_roi_gate_per_batch(mask: torch.Tensor, radii: torch.Tensor, smooth_steps: int = 2) -> torch.Tensor:
    """Build a dilated smooth ROI gate with a possibly different radius per pair."""

    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask shape (B,1,H,W,D), got {tuple(mask.shape)}")
    radii = torch.as_tensor(radii, device=mask.device).flatten().long()
    if radii.numel() == 1:
        radii = radii.expand(mask.shape[0])
    if radii.numel() != mask.shape[0]:
        raise ValueError(f"Expected {mask.shape[0]} radii, got {radii.numel()}")
    gates = [
        build_roi_gate(mask[index : index + 1], radius=int(radii[index].item()), smooth_steps=smooth_steps)
        for index in range(mask.shape[0])
    ]
    return torch.cat(gates, dim=0)


def build_roi_gate(mask: torch.Tensor, radius: int, smooth_steps: int = 2) -> torch.Tensor:
    """Build a dilated smooth ROI gate from a binary small-OAR mask."""

    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask shape (B,1,H,W,D), got {tuple(mask.shape)}")
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")

    hard_mask = (mask > 0).float()
    if radius == 0:
        gate = hard_mask
    else:
        kernel_size = 2 * radius + 1
        gate = F.max_pool3d(hard_mask, kernel_size=kernel_size, stride=1, padding=radius)

    for _ in range(max(0, int(smooth_steps))):
        gate = F.avg_pool3d(gate, kernel_size=3, stride=1, padding=1)
        gate = torch.maximum(gate, hard_mask)
    return gate.clamp(0.0, 1.0)


def stage3_conditioning_masks(
    input_mode: str,
    fixed_small_mask: torch.Tensor,
    warped_small_mask_stage2: torch.Tensor,
    fixed_bone_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return Stage-3 feature/ROI/anatomy masks for leakage ablations."""

    if input_mode not in STAGE3_INPUT_MODES:
        raise ValueError(f"Unsupported Stage-3 input mode {input_mode!r}; choices={STAGE3_INPUT_MODES}")

    zeros_small = torch.zeros_like(fixed_small_mask)
    zeros_bone = torch.zeros_like(fixed_bone_mask)
    if input_mode == "full":
        fixed_small_feature = fixed_small_mask
        fixed_bone_feature = fixed_bone_mask
        roi_source = torch.maximum(fixed_small_mask, warped_small_mask_stage2.detach())
        anatomy_bone = fixed_bone_mask
    elif input_mode == "no-fixed-small":
        fixed_small_feature = zeros_small
        fixed_bone_feature = fixed_bone_mask
        roi_source = warped_small_mask_stage2.detach()
        anatomy_bone = fixed_bone_mask
    else:
        fixed_small_feature = zeros_small
        fixed_bone_feature = zeros_bone
        roi_source = warped_small_mask_stage2.detach()
        anatomy_bone = zeros_bone

    return {
        "fixed_small_feature": fixed_small_feature,
        "fixed_bone_feature": fixed_bone_feature,
        "roi_source": roi_source,
        "anatomy_bone": anatomy_bone,
    }


def normalize_dvf_magnitude(dvf: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return a per-case normalized DVF magnitude channel."""

    magnitude = torch.sqrt(dvf.float().pow(2).sum(dim=1, keepdim=True) + eps)
    flat = magnitude.flatten(1)
    scale = torch.quantile(flat, q=0.95, dim=1).clamp_min(eps)
    scale = scale.view(-1, 1, 1, 1, 1)
    return (magnitude / scale).clamp(0.0, 1.0)


def build_anatomy_maps(
    fixed_bone_mask: torch.Tensor,
    roi_gate: torch.Tensor,
    difficulty: torch.Tensor,
    smooth_base: float = 1.0,
    smooth_bone: float = 4.0,
    smooth_boundary: float = 2.0,
    smooth_difficulty: float = 1.0,
    mag_inside: float = 0.2,
    mag_outside: float = 6.0,
    mag_bone: float = 4.0,
) -> Dict[str, torch.Tensor]:
    """Create rule-based anatomy-conditioned regularization maps."""

    if fixed_bone_mask.shape != roi_gate.shape:
        raise ValueError(f"fixed_bone_mask shape {tuple(fixed_bone_mask.shape)} != roi_gate {tuple(roi_gate.shape)}")

    fixed_bone_mask = fixed_bone_mask.float().clamp(0.0, 1.0)
    roi_gate = roi_gate.float().clamp(0.0, 1.0)
    outside_roi = (1.0 - roi_gate).clamp(0.0, 1.0)
    pooled = F.avg_pool3d(roi_gate, kernel_size=3, stride=1, padding=1)
    boundary = (roi_gate - pooled).abs().clamp(0.0, 1.0)
    diff = difficulty.float().view(-1, 1, 1, 1, 1).clamp(0.0, 1.0)

    smooth_bone_weight = smooth_bone * (1.0 + 0.5 * diff)
    smooth_boundary_weight = smooth_boundary * (1.0 + diff)
    mag_inside_weight = mag_inside * (1.0 - 0.5 * diff)
    mag_outside_weight = mag_outside * (1.0 + 0.5 * diff)
    mag_bone_weight = mag_bone * (1.0 + 0.5 * diff)

    r_smooth = (
        smooth_base
        + smooth_bone_weight * fixed_bone_mask
        + smooth_boundary_weight * boundary
        + smooth_difficulty * diff * roi_gate
    )
    r_mag = mag_inside_weight * roi_gate + mag_outside_weight * outside_roi + mag_bone_weight * fixed_bone_mask
    r_refine = roi_gate * (1.0 + diff)
    return {"smooth": r_smooth, "magnitude": r_mag, "refine": r_refine}


def masked_mse_loss_per_batch(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Masked MSE for each batch item."""

    weight = mask.float()
    while weight.ndim < pred.ndim:
        weight = weight.unsqueeze(1)
    numerator = ((pred - target).pow(2) * weight).flatten(1).sum(dim=1)
    denominator = weight.flatten(1).sum(dim=1) * pred.shape[1] + eps
    return numerator / denominator


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """MSE inside a soft mask."""

    return masked_mse_loss_per_batch(pred, target, mask, eps=eps).mean()


def weighted_gradient_loss_per_batch(dvf: torch.Tensor, weight_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Weighted first-order smoothness loss for each batch item."""

    weight_map = weight_map.float()
    dx = dvf[:, :, 1:, :, :] - dvf[:, :, :-1, :, :]
    dy = dvf[:, :, :, 1:, :] - dvf[:, :, :, :-1, :]
    dz = dvf[:, :, :, :, 1:] - dvf[:, :, :, :, :-1]
    wx = 0.5 * (weight_map[:, :, 1:, :, :] + weight_map[:, :, :-1, :, :])
    wy = 0.5 * (weight_map[:, :, :, 1:, :] + weight_map[:, :, :, :-1, :])
    wz = 0.5 * (weight_map[:, :, :, :, 1:] + weight_map[:, :, :, :, :-1])

    channels = dvf.shape[1]
    reduce_dims = (1, 2, 3, 4)
    loss_x = (dx.pow(2) * wx).sum(dim=reduce_dims) / (wx.sum(dim=reduce_dims) * channels + eps)
    loss_y = (dy.pow(2) * wy).sum(dim=reduce_dims) / (wy.sum(dim=reduce_dims) * channels + eps)
    loss_z = (dz.pow(2) * wz).sum(dim=reduce_dims) / (wz.sum(dim=reduce_dims) * channels + eps)
    return (loss_x + loss_y + loss_z) / 3.0


def weighted_gradient_loss(dvf: torch.Tensor, weight_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Weighted first-order smoothness loss for a DVF."""

    return weighted_gradient_loss_per_batch(dvf, weight_map, eps=eps).mean()


def weighted_magnitude_loss_per_batch(dvf: torch.Tensor, weight_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Weighted residual DVF magnitude loss for each batch item."""

    channels = dvf.shape[1]
    weight_map = weight_map.float()
    reduce_dims = (1, 2, 3, 4)
    return (dvf.pow(2) * weight_map).sum(dim=reduce_dims) / (weight_map.sum(dim=reduce_dims) * channels + eps)


def weighted_magnitude_loss(dvf: torch.Tensor, weight_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Weighted residual DVF magnitude loss."""

    return weighted_magnitude_loss_per_batch(dvf, weight_map, eps=eps).mean()


def _summary_from_values(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean().cpu()),
        "p95": float(torch.quantile(values, q=0.95).cpu()),
        "max": float(values.max().cpu()),
    }


def magnitude_stats(
    dvf: torch.Tensor,
    roi_gate: Optional[torch.Tensor] = None,
    roi_threshold: float = 1e-4,
) -> Dict[str, float]:
    """DVF magnitude stats globally and, optionally, inside the ROI gate."""

    magnitude = torch.sqrt(dvf.float().pow(2).sum(dim=1, keepdim=True) + 1e-6)
    stats = {f"global_{key}": value for key, value in _summary_from_values(magnitude).items()}
    if roi_gate is not None:
        roi_mask = roi_gate.float() > roi_threshold
        roi_values = magnitude[roi_mask.expand_as(magnitude)]
        stats.update({f"roi_{key}": value for key, value in _summary_from_values(roi_values).items()})
    return stats


def jacobian_determinant(dvf: torch.Tensor) -> torch.Tensor:
    """Compute forward-difference Jacobian determinant of `id + dvf`."""

    if dvf.ndim != 5 or dvf.shape[1] != 3:
        raise ValueError(f"Expected DVF shape (B,3,H,W,D), got {tuple(dvf.shape)}")
    base = dvf[:, :, :-1, :-1, :-1]
    du_dx = dvf[:, :, 1:, :-1, :-1] - base
    du_dy = dvf[:, :, :-1, 1:, :-1] - base
    du_dz = dvf[:, :, :-1, :-1, 1:] - base

    j00 = 1.0 + du_dx[:, 0]
    j01 = du_dy[:, 0]
    j02 = du_dz[:, 0]
    j10 = du_dx[:, 1]
    j11 = 1.0 + du_dy[:, 1]
    j12 = du_dz[:, 1]
    j20 = du_dx[:, 2]
    j21 = du_dy[:, 2]
    j22 = 1.0 + du_dz[:, 2]

    return (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )


def jacobian_hinge_loss_per_batch(
    dvf: torch.Tensor,
    roi_gate: Optional[torch.Tensor] = None,
    margin: float = 0.05,
    roi_weight: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize low or non-positive Jacobian determinants for each batch item."""

    jac = jacobian_determinant(dvf)
    penalty = F.relu(float(margin) - jac).pow(2).unsqueeze(1)
    global_loss = penalty.flatten(1).mean(dim=1)
    if roi_gate is None:
        return global_loss

    roi_inner = (roi_gate[:, :, :-1, :-1, :-1] > 1e-4).float()
    roi_numerator = (penalty * roi_inner).flatten(1).sum(dim=1)
    roi_denominator = roi_inner.flatten(1).sum(dim=1).clamp_min(eps)
    roi_loss = roi_numerator / roi_denominator
    return global_loss + float(roi_weight) * roi_loss


def jacobian_hinge_loss(
    dvf: torch.Tensor,
    roi_gate: Optional[torch.Tensor] = None,
    margin: float = 0.05,
    roi_weight: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean low-Jacobian hinge loss."""

    return jacobian_hinge_loss_per_batch(
        dvf=dvf,
        roi_gate=roi_gate,
        margin=margin,
        roi_weight=roi_weight,
        eps=eps,
    ).mean()


def jacobian_stats(
    dvf: torch.Tensor,
    roi_gate: Optional[torch.Tensor] = None,
    roi_threshold: float = 1e-4,
) -> Dict[str, float]:
    """Summarize folding and low-Jacobian behavior globally and inside ROI."""

    jac = jacobian_determinant(dvf).detach().float()
    flat = jac.flatten()
    stats = {
        "global_min": float(flat.min().cpu()),
        "global_p01": float(torch.quantile(flat, q=0.01).cpu()),
        "global_p05": float(torch.quantile(flat, q=0.05).cpu()),
        "global_nonpos_ratio": float((flat <= 0).float().mean().cpu()),
    }
    if roi_gate is not None:
        roi_inner = roi_gate[:, :, :-1, :-1, :-1] > roi_threshold
        roi_values = jac[roi_inner[:, 0]]
        if roi_values.numel() == 0:
            stats.update(
                {
                    "roi_min": 0.0,
                    "roi_p01": 0.0,
                    "roi_p05": 0.0,
                    "roi_nonpos_ratio": 0.0,
                }
            )
        else:
            stats.update(
                {
                    "roi_min": float(roi_values.min().cpu()),
                    "roi_p01": float(torch.quantile(roi_values, q=0.01).cpu()),
                    "roi_p05": float(torch.quantile(roi_values, q=0.05).cpu()),
                    "roi_nonpos_ratio": float((roi_values <= 0).float().mean().cpu()),
                }
            )
    return stats


def label_dice(before: np.ndarray, fixed: np.ndarray, after: np.ndarray, label: int) -> Dict[str, float]:
    """Before/after Dice for one integer label."""

    before_mask = before == label
    fixed_mask = fixed == label
    after_mask = after == label
    before_dice = (2.0 * np.logical_and(before_mask, fixed_mask).sum()) / (
        before_mask.sum() + fixed_mask.sum() + 1e-5
    )
    after_dice = (2.0 * np.logical_and(after_mask, fixed_mask).sum()) / (
        after_mask.sum() + fixed_mask.sum() + 1e-5
    )
    return {
        "before": float(before_dice),
        "after": float(after_dice),
        "delta": float(after_dice - before_dice),
    }


def label_dice_table(
    moving: np.ndarray,
    fixed: np.ndarray,
    stage2: np.ndarray,
    final: np.ndarray,
    labels: Sequence[int],
) -> Dict[str, object]:
    """Per-label Dice table comparing unregistered, Stage-2, and final masks."""

    per_label = {}
    stage2_values = []
    final_values = []
    deltas = []
    for label in sorted(set(int(v) for v in labels if int(v) > 0)):
        before_stage2 = label_dice(moving, fixed, stage2, label)
        before_final = label_dice(moving, fixed, final, label)
        entry = {
            "before": before_stage2["before"],
            "stage2": before_stage2["after"],
            "final": before_final["after"],
            "delta_final_vs_stage2": before_final["after"] - before_stage2["after"],
        }
        per_label[str(label)] = {key: float(value) for key, value in entry.items()}
        stage2_values.append(entry["stage2"])
        final_values.append(entry["final"])
        deltas.append(entry["delta_final_vs_stage2"])

    if not per_label:
        return {
            "labels": [],
            "mean_stage2": 0.0,
            "mean_final": 0.0,
            "mean_delta": 0.0,
            "median_delta": 0.0,
            "worst_delta": 0.0,
            "num_drop_gt_0_02": 0,
            "num_drop_gt_0_05": 0,
            "per_label": {},
        }

    stage2_arr = np.asarray(stage2_values, dtype=np.float64)
    final_arr = np.asarray(final_values, dtype=np.float64)
    delta_arr = np.asarray(deltas, dtype=np.float64)
    return {
        "labels": [int(label) for label in sorted(set(int(v) for v in labels if int(v) > 0))],
        "mean_stage2": float(np.mean(stage2_arr)),
        "mean_final": float(np.mean(final_arr)),
        "mean_delta": float(np.mean(delta_arr)),
        "median_delta": float(np.median(delta_arr)),
        "worst_delta": float(np.min(delta_arr)),
        "num_drop_gt_0_02": int(np.sum(delta_arr < -0.02)),
        "num_drop_gt_0_05": int(np.sum(delta_arr < -0.05)),
        "per_label": per_label,
    }


def split_present_labels(labels: Sequence[int], small_labels: Sequence[int]) -> Tuple[List[int], List[int]]:
    """Split non-background labels into small-OAR and large/other labels."""

    small = sorted(set(int(label) for label in small_labels if int(label) > 0))
    small_set = set(small)
    large = sorted(set(int(label) for label in labels if int(label) > 0 and int(label) not in small_set))
    return small, large


def make_stage3_inputs(
    fixed: torch.Tensor,
    deformed_stage2: torch.Tensor,
    fixed_small_mask: torch.Tensor,
    warped_small_mask_stage2: torch.Tensor,
    dvf_stage2: torch.Tensor,
    fixed_bone_mask: torch.Tensor,
    warped_bone_mask_stage2: torch.Tensor,
) -> torch.Tensor:
    """Assemble the Phase-1 Stage-3 feature tensor."""

    dvf_magnitude = normalize_dvf_magnitude(dvf_stage2)
    return torch.cat(
        (
            fixed.float(),
            deformed_stage2.float(),
            fixed_small_mask.float(),
            warped_small_mask_stage2.float(),
            dvf_magnitude.float(),
            fixed_bone_mask.float(),
            warped_bone_mask_stage2.float(),
        ),
        dim=1,
    )


def checkpoint_to_state_dict(payload: object) -> Dict[str, torch.Tensor]:
    """Extract a PyTorch state dict from common checkpoint payloads."""

    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported checkpoint payload type: {type(payload)}")
