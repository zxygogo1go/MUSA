"""Counterfactual Registrar Spectroscopy helpers for MUSA+ Stage-3 training."""

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import utils_musa_plus
from . import utils_warp

try:
    from scipy.ndimage import distance_transform_edt
except ModuleNotFoundError:  # pragma: no cover - scipy is listed as a project dependency.
    distance_transform_edt = None


CRS_PROBE_MODES: Tuple[str, ...] = (
    "translation",
    "rotation",
    "anisotropic_scale",
    "shear",
    "bending",
    "low_frequency_bspline",
)


@dataclass(frozen=True)
class ProbeSpec:
    """Requested synthetic deformation bin for one counterfactual sample."""

    target_label: int
    mode: str
    amplitude_mm: float
    support_radius_mm: float


@dataclass
class ProbeMetadata:
    """Serializable metadata used for CRS logging and adaptive sampling."""

    target_label: int
    mode: str
    amplitude_mm: float
    amplitude_bin: str
    support_radius_mm: float
    support_radius_bin: str
    volume_voxels: int
    volume_bin: str
    rejected_attempts: int = 0
    accepted_jacobian_min: float = 0.0
    bbox_min: Tuple[int, int, int] = (0, 0, 0)
    bbox_max: Tuple[int, int, int] = (0, 0, 0)


def _ensure_5d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor.unsqueeze(1)
    if tensor.ndim != 5:
        raise ValueError(f"Expected a 4D or 5D tensor, got shape {tuple(tensor.shape)}")
    return tensor


def _normalize_mode(mode: str) -> str:
    text = mode.strip().lower().replace("-", "_")
    if text in {"bspline", "b_spline", "low_frequency_b_spline"}:
        return "low_frequency_bspline"
    if text not in CRS_PROBE_MODES:
        raise ValueError(f"Unsupported CRS probe mode {mode!r}; choices={CRS_PROBE_MODES}")
    return text


def _as_tuple3(values: Sequence[float]) -> Tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"Expected three spacing values, got {values}")
    output = tuple(float(value) for value in values)
    if any(value <= 0 for value in output):
        raise ValueError(f"Spacing values must be positive, got {output}")
    return output


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _meshgrid_voxels(
    spatial_shape: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    axes = [torch.arange(size, device=device, dtype=dtype) for size in spatial_shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0).unsqueeze(0)


def integrate_stationary_velocity(
    velocity: torch.Tensor,
    scaling_steps: int = 4,
    composer: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """Integrate a stationary velocity field with scaling-and-squaring.

    The field uses the same voxel-unit, fixed-grid sampling convention as
    :class:`musa.utils_warp.SpatialTransformer`.
    """

    if velocity.ndim != 5 or velocity.shape[1] != 3:
        raise ValueError(f"Expected velocity shape (B,3,D,H,W), got {tuple(velocity.shape)}")
    if scaling_steps < 0:
        raise ValueError(f"scaling_steps must be non-negative, got {scaling_steps}")

    if composer is None:
        composer = utils_warp.ComposeDVF(tuple(velocity.shape[2:])).to(velocity.device)

    flow = velocity / float(2**scaling_steps)
    for _ in range(scaling_steps):
        flow = composer(flow, flow)
    return flow


def masked_epe_per_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    outside_weight: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Endpoint error averaged per batch item inside a soft ROI."""

    weight = mask.float().clamp(0.0, 1.0)
    if outside_weight > 0:
        weight = outside_weight + (1.0 - outside_weight) * weight
    while weight.ndim < pred.ndim:
        weight = weight.unsqueeze(1)
    error = torch.sqrt((pred.float() - target.float()).pow(2).sum(dim=1, keepdim=True) + eps)
    numerator = (error * weight).flatten(1).sum(dim=1)
    denominator = weight.flatten(1).sum(dim=1).clamp_min(eps)
    return numerator / denominator


def masked_charbonnier_per_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    outside_weight: float = 0.0,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Masked Charbonnier loss for vector fields, reduced per batch item."""

    weight = mask.float().clamp(0.0, 1.0)
    if outside_weight > 0:
        weight = outside_weight + (1.0 - outside_weight) * weight
    while weight.ndim < pred.ndim:
        weight = weight.unsqueeze(1)
    loss = torch.sqrt((pred.float() - target.float()).pow(2) + eps**2)
    numerator = (loss * weight).flatten(1).sum(dim=1)
    denominator = (weight.flatten(1).sum(dim=1) * pred.shape[1]).clamp_min(1e-6)
    return numerator / denominator


class AnatomicalCounterfactualProbeGenerator:
    """Generate local anatomical counterfactual pairs around small OARs."""

    def __init__(
        self,
        small_oar_labels: Sequence[int],
        spacing_mm: Sequence[float] = (2.0, 2.0, 2.0),
        probe_modes: Sequence[str] = CRS_PROBE_MODES,
        amplitude_mm: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        support_radius_mm: Sequence[float] = (8.0, 12.0, 16.0, 24.0),
        scaling_steps: int = 4,
        folding_jacobian_min: float = 0.02,
        max_attempts: int = 4,
        appearance_augmentation: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        labels = sorted(set(int(label) for label in small_oar_labels if int(label) > 0))
        if not labels:
            raise ValueError("CRS requires at least one positive small-OAR label")
        self.small_oar_labels = labels
        self.spacing_mm = _as_tuple3(spacing_mm)
        self.probe_modes = tuple(_normalize_mode(mode) for mode in probe_modes)
        self.amplitude_mm = tuple(float(value) for value in amplitude_mm)
        self.support_radius_mm = tuple(float(value) for value in support_radius_mm)
        self.scaling_steps = int(scaling_steps)
        self.folding_jacobian_min = float(folding_jacobian_min)
        self.max_attempts = int(max_attempts)
        self.appearance_augmentation = bool(appearance_augmentation)
        self.rng = np.random.default_rng(seed)

        if not self.probe_modes:
            raise ValueError("CRS requires at least one probe mode")
        if not self.amplitude_mm or any(value <= 0 for value in self.amplitude_mm):
            raise ValueError(f"Invalid CRS amplitudes: {self.amplitude_mm}")
        if not self.support_radius_mm or any(value <= 0 for value in self.support_radius_mm):
            raise ValueError(f"Invalid CRS support radii: {self.support_radius_mm}")

    def sample_specs(self, batch_size: int) -> List[ProbeSpec]:
        specs = []
        for _ in range(batch_size):
            specs.append(
                ProbeSpec(
                    target_label=int(self.rng.choice(self.small_oar_labels)),
                    mode=str(self.rng.choice(self.probe_modes)),
                    amplitude_mm=float(self.rng.choice(self.amplitude_mm)),
                    support_radius_mm=float(self.rng.choice(self.support_radius_mm)),
                )
            )
        return specs

    def generate(
        self,
        fixed_ct: torch.Tensor,
        fixed_seg_o: torch.Tensor,
        fixed_seg_b: torch.Tensor,
        probe_specs: Optional[Sequence[ProbeSpec]] = None,
    ) -> Dict[str, object]:
        fixed_ct = _ensure_5d(fixed_ct).float()
        fixed_seg_o = _ensure_5d(fixed_seg_o).long()
        fixed_seg_b = _ensure_5d(fixed_seg_b).long()
        if fixed_ct.shape[0] != fixed_seg_o.shape[0] or fixed_ct.shape[0] != fixed_seg_b.shape[0]:
            raise ValueError("fixed_ct, fixed_seg_o, and fixed_seg_b must have matching batch sizes")
        if fixed_ct.shape[2:] != fixed_seg_o.shape[2:] or fixed_ct.shape[2:] != fixed_seg_b.shape[2:]:
            raise ValueError("fixed_ct and segmentation tensors must have matching spatial shapes")

        batch_size = fixed_ct.shape[0]
        if probe_specs is None:
            probe_specs = self.sample_specs(batch_size)
        if len(probe_specs) != batch_size:
            raise ValueError(f"Expected {batch_size} probe specs, got {len(probe_specs)}")

        transformer = utils_warp.SpatialTransformer(tuple(fixed_ct.shape[2:])).to(fixed_ct.device)
        composer = utils_warp.ComposeDVF(tuple(fixed_ct.shape[2:])).to(fixed_ct.device)

        moving_ct_rows = []
        moving_seg_o_rows = []
        moving_seg_b_rows = []
        known_dvf_rows = []
        support_rows = []
        metadata_rows = []

        for index, spec in enumerate(probe_specs):
            sample = self._generate_one(
                fixed_ct=fixed_ct[index : index + 1],
                fixed_seg_o=fixed_seg_o[index : index + 1],
                fixed_seg_b=fixed_seg_b[index : index + 1],
                spec=spec,
                transformer=transformer,
                composer=composer,
            )
            moving_ct_rows.append(sample["moving_ct"])
            moving_seg_o_rows.append(sample["moving_seg_o"])
            moving_seg_b_rows.append(sample["moving_seg_b"])
            known_dvf_rows.append(sample["known_gt_dvf"])
            support_rows.append(sample["support"])
            metadata_rows.append(sample["metadata"])

        return {
            "counterfactual_moving_ct": torch.cat(moving_ct_rows, dim=0),
            "counterfactual_moving_seg_o": torch.cat(moving_seg_o_rows, dim=0),
            "counterfactual_moving_seg_b": torch.cat(moving_seg_b_rows, dim=0),
            "known_gt_dvf": torch.cat(known_dvf_rows, dim=0),
            "support": torch.cat(support_rows, dim=0),
            "metadata": metadata_rows,
        }

    def _generate_one(
        self,
        fixed_ct: torch.Tensor,
        fixed_seg_o: torch.Tensor,
        fixed_seg_b: torch.Tensor,
        spec: ProbeSpec,
        transformer: torch.nn.Module,
        composer: torch.nn.Module,
    ) -> Dict[str, object]:
        label = self._resolve_present_label(fixed_seg_o, int(spec.target_label))
        target_mask = (fixed_seg_o == label).float()
        support, centroid_vox, volume_voxels, bbox_min, bbox_max = self._support_from_mask(
            target_mask=target_mask,
            support_radius_mm=float(spec.support_radius_mm),
        )
        metadata = ProbeMetadata(
            target_label=int(label),
            mode=_normalize_mode(spec.mode),
            amplitude_mm=float(spec.amplitude_mm),
            amplitude_bin=self._amplitude_bin(float(spec.amplitude_mm)),
            support_radius_mm=float(spec.support_radius_mm),
            support_radius_bin=self._radius_bin(float(spec.support_radius_mm)),
            volume_voxels=int(volume_voxels),
            volume_bin=self._volume_bin(int(volume_voxels)),
            bbox_min=tuple(int(value) for value in bbox_min),
            bbox_max=tuple(int(value) for value in bbox_max),
        )

        velocity = self._build_velocity(
            mode=metadata.mode,
            amplitude_mm=metadata.amplitude_mm,
            support_radius_mm=metadata.support_radius_mm,
            support=support,
            centroid_vox=centroid_vox,
            spatial_shape=fixed_ct.shape[2:],
            device=fixed_ct.device,
            dtype=fixed_ct.dtype,
        )
        known_gt_dvf, accepted_velocity, jac_min, rejected = self._integrate_with_rejection(velocity, composer)
        inverse_dvf = integrate_stationary_velocity(-accepted_velocity, self.scaling_steps, composer=composer)
        metadata.rejected_attempts = rejected
        metadata.accepted_jacobian_min = jac_min

        moving_ct = transformer(fixed_ct, inverse_dvf, mode="bilinear").clamp(0.0, 1.0)
        if self.appearance_augmentation:
            moving_ct = self._appearance_augment(moving_ct)
        moving_seg_o = transformer(fixed_seg_o.float(), inverse_dvf, mode="nearest").round().long()
        moving_seg_b = transformer(fixed_seg_b.float(), inverse_dvf, mode="nearest").round().long()

        return {
            "moving_ct": moving_ct,
            "moving_seg_o": moving_seg_o,
            "moving_seg_b": moving_seg_b,
            "known_gt_dvf": known_gt_dvf.detach(),
            "support": support.detach(),
            "metadata": metadata,
        }

    def _resolve_present_label(self, fixed_seg_o: torch.Tensor, requested_label: int) -> int:
        present = set(int(value) for value in torch.unique(fixed_seg_o.detach()).cpu().tolist())
        if requested_label in present:
            return requested_label
        candidates = [label for label in self.small_oar_labels if label in present]
        if candidates:
            return int(self.rng.choice(candidates))
        return requested_label

    def _support_from_mask(
        self,
        target_mask: torch.Tensor,
        support_radius_mm: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, Tuple[int, int, int], Tuple[int, int, int]]:
        mask = target_mask.detach() > 0.5
        coords = torch.nonzero(mask[0, 0], as_tuple=False)
        spatial_shape = tuple(target_mask.shape[2:])
        if coords.numel() == 0:
            center = torch.tensor(
                [(size - 1) / 2.0 for size in spatial_shape],
                device=target_mask.device,
                dtype=target_mask.dtype,
            )
            bbox_min = tuple(int(max(0, math.floor(value - 1))) for value in center.detach().cpu().tolist())
            bbox_max = tuple(int(min(spatial_shape[i] - 1, math.ceil(float(center[i]) + 1))) for i in range(3))
            support = self._centroid_support(spatial_shape, center, support_radius_mm, target_mask.device, target_mask.dtype)
            return support, center, 0, bbox_min, bbox_max

        coords_float = coords.to(device=target_mask.device, dtype=target_mask.dtype)
        center = coords_float.mean(dim=0)
        bbox_min_t = coords.min(dim=0).values
        bbox_max_t = coords.max(dim=0).values
        bbox_min = tuple(int(value) for value in bbox_min_t.detach().cpu().tolist())
        bbox_max = tuple(int(value) for value in bbox_max_t.detach().cpu().tolist())
        volume_voxels = int(coords.shape[0])

        if distance_transform_edt is None:
            support = self._centroid_support(spatial_shape, center, support_radius_mm, target_mask.device, target_mask.dtype)
        else:
            mask_np = mask[0, 0].detach().cpu().numpy().astype(bool)
            distance_mm = distance_transform_edt(~mask_np, sampling=self.spacing_mm)
            clipped = np.clip(distance_mm / max(support_radius_mm, 1e-6), 0.0, 1.0)
            support_np = 0.5 * (1.0 + np.cos(np.pi * clipped))
            support_np[distance_mm > support_radius_mm] = 0.0
            support_np[mask_np] = 1.0
            support = torch.from_numpy(support_np).to(device=target_mask.device, dtype=target_mask.dtype)
            support = support.view(1, 1, *spatial_shape)
        return support.clamp(0.0, 1.0), center, volume_voxels, bbox_min, bbox_max

    def _centroid_support(
        self,
        spatial_shape: Sequence[int],
        center_vox: torch.Tensor,
        support_radius_mm: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        grid = _meshgrid_voxels(spatial_shape, device, dtype)
        spacing = torch.tensor(self.spacing_mm, device=device, dtype=dtype).view(1, 3, 1, 1, 1)
        rel_mm = (grid - center_vox.view(1, 3, 1, 1, 1)) * spacing
        distance = torch.sqrt(rel_mm.pow(2).sum(dim=1, keepdim=True) + 1e-6)
        clipped = (distance / max(float(support_radius_mm), 1e-6)).clamp(0.0, 1.0)
        support = 0.5 * (1.0 + torch.cos(math.pi * clipped))
        return torch.where(distance <= float(support_radius_mm), support, torch.zeros_like(support))

    def _build_velocity(
        self,
        mode: str,
        amplitude_mm: float,
        support_radius_mm: float,
        support: torch.Tensor,
        centroid_vox: torch.Tensor,
        spatial_shape: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        grid = _meshgrid_voxels(spatial_shape, device, dtype)
        spacing = torch.tensor(self.spacing_mm, device=device, dtype=dtype).view(1, 3, 1, 1, 1)
        rel_mm = (grid - centroid_vox.view(1, 3, 1, 1, 1)) * spacing
        support = support.to(device=device, dtype=dtype)
        mode = _normalize_mode(mode)

        if mode == "translation":
            direction = torch.tensor(_random_unit_vector(self.rng), device=device, dtype=dtype).view(1, 3, 1, 1, 1)
            displacement_mm = direction * float(amplitude_mm)
        elif mode == "rotation":
            axis = torch.tensor(_random_unit_vector(self.rng), device=device, dtype=dtype).view(1, 3, 1, 1, 1)
            theta = float(amplitude_mm) / max(float(support_radius_mm), 1e-6)
            displacement_mm = theta * torch.cat(
                (
                    axis[:, 1:2] * rel_mm[:, 2:3] - axis[:, 2:3] * rel_mm[:, 1:2],
                    axis[:, 2:3] * rel_mm[:, 0:1] - axis[:, 0:1] * rel_mm[:, 2:3],
                    axis[:, 0:1] * rel_mm[:, 1:2] - axis[:, 1:2] * rel_mm[:, 0:1],
                ),
                dim=1,
            )
        elif mode == "anisotropic_scale":
            displacement_mm = torch.zeros_like(rel_mm)
            axis = int(self.rng.integers(0, 3))
            sign = float(self.rng.choice([-1.0, 1.0]))
            scale = sign * float(amplitude_mm) / max(float(support_radius_mm), 1e-6)
            displacement_mm[:, axis : axis + 1] = scale * rel_mm[:, axis : axis + 1]
        elif mode == "shear":
            displacement_mm = torch.zeros_like(rel_mm)
            source_axis, target_axis = self.rng.choice(3, size=2, replace=False)
            sign = float(self.rng.choice([-1.0, 1.0]))
            shear = sign * float(amplitude_mm) / max(float(support_radius_mm), 1e-6)
            displacement_mm[:, target_axis : target_axis + 1] = shear * rel_mm[:, source_axis : source_axis + 1]
        elif mode == "bending":
            displacement_mm = torch.zeros_like(rel_mm)
            source_axis, target_axis = self.rng.choice(3, size=2, replace=False)
            sign = float(self.rng.choice([-1.0, 1.0]))
            phase = math.pi * rel_mm[:, source_axis : source_axis + 1] / max(float(support_radius_mm), 1e-6)
            displacement_mm[:, target_axis : target_axis + 1] = sign * float(amplitude_mm) * torch.sin(phase)
        elif mode == "low_frequency_bspline":
            coarse_shape = tuple(max(2, min(6, math.ceil(size / 32))) for size in spatial_shape)
            coarse = torch.from_numpy(self.rng.normal(size=(1, 3, *coarse_shape)).astype(np.float32)).to(
                device=device,
                dtype=dtype,
            )
            displacement_mm = F.interpolate(coarse, size=tuple(spatial_shape), mode="trilinear", align_corners=True)
            for _ in range(2):
                displacement_mm = F.avg_pool3d(displacement_mm, kernel_size=3, stride=1, padding=1)
            magnitude = torch.sqrt(displacement_mm.pow(2).sum(dim=1, keepdim=True) + 1e-6)
            roi_values = magnitude[support.expand_as(magnitude) > 1e-4]
            scale = torch.quantile(roi_values, q=0.95).clamp_min(1e-6) if roi_values.numel() else magnitude.max()
            displacement_mm = displacement_mm * (float(amplitude_mm) / scale.clamp_min(1e-6))
        else:  # pragma: no cover - guarded by _normalize_mode.
            raise ValueError(f"Unsupported CRS probe mode {mode!r}")

        displacement_mm = displacement_mm * support
        return displacement_mm / spacing

    def _integrate_with_rejection(
        self,
        velocity: torch.Tensor,
        composer: torch.nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
        rejected = 0
        candidate_velocity = velocity
        for attempt in range(max(1, self.max_attempts)):
            flow = integrate_stationary_velocity(candidate_velocity, self.scaling_steps, composer=composer)
            with torch.no_grad():
                jac = utils_musa_plus.jacobian_stats(flow)
            jac_min = float(jac["global_min"])
            if jac_min > self.folding_jacobian_min and float(jac["global_nonpos_ratio"]) == 0.0:
                return flow, candidate_velocity, jac_min, rejected
            rejected += 1
            candidate_velocity = candidate_velocity * 0.5
        flow = integrate_stationary_velocity(candidate_velocity, self.scaling_steps, composer=composer)
        with torch.no_grad():
            jac = utils_musa_plus.jacobian_stats(flow)
        return flow, candidate_velocity, float(jac["global_min"]), rejected

    def _appearance_augment(self, image: torch.Tensor) -> torch.Tensor:
        output = image
        scale = float(self.rng.uniform(0.95, 1.05))
        shift = float(self.rng.uniform(-0.03, 0.03))
        gamma = float(self.rng.uniform(0.85, 1.15))
        output = (output * scale + shift).clamp(0.0, 1.0)
        output = output.clamp_min(1e-6).pow(gamma)
        noise_std = float(self.rng.uniform(0.0, 0.015))
        if noise_std > 0:
            noise = torch.from_numpy(self.rng.normal(0.0, noise_std, size=tuple(output.shape)).astype(np.float32)).to(
                device=output.device,
                dtype=output.dtype,
            )
            output = output + noise
        if float(self.rng.uniform()) < 0.35:
            output = F.avg_pool3d(output, kernel_size=3, stride=1, padding=1)
        coarse = torch.from_numpy(self.rng.normal(size=(1, 1, 4, 4, 4)).astype(np.float32)).to(
            device=image.device,
            dtype=image.dtype,
        )
        bias = F.interpolate(coarse, size=image.shape[2:], mode="trilinear", align_corners=True)
        bias = bias / bias.abs().flatten(1).amax(dim=1).view(-1, 1, 1, 1, 1).clamp_min(1e-6)
        output = output * (1.0 + 0.05 * bias)
        return output.clamp(0.0, 1.0)

    @staticmethod
    def _amplitude_bin(value: float) -> str:
        return f"{float(value):.2f}mm"

    @staticmethod
    def _radius_bin(value: float) -> str:
        return f"{float(value):.2f}mm"

    @staticmethod
    def _volume_bin(volume_voxels: int) -> str:
        if volume_voxels < 64:
            return "tiny"
        if volume_voxels < 512:
            return "small"
        if volume_voxels < 4096:
            return "medium"
        return "large"


class ResidualTargetBuilder:
    """Build exact CRS residual targets for the current additive Stage-3 convention."""

    @staticmethod
    def build_additive(known_gt_dvf: torch.Tensor, stage2_dvf: torch.Tensor) -> torch.Tensor:
        if known_gt_dvf.shape != stage2_dvf.shape:
            raise ValueError(f"DVF shape mismatch: {tuple(known_gt_dvf.shape)} != {tuple(stage2_dvf.shape)}")
        return known_gt_dvf.detach() - stage2_dvf.detach()

    @staticmethod
    def reconstruct_additive(stage2_dvf: torch.Tensor, effective_residual: torch.Tensor) -> torch.Tensor:
        if stage2_dvf.shape != effective_residual.shape:
            raise ValueError(f"DVF shape mismatch: {tuple(stage2_dvf.shape)} != {tuple(effective_residual.shape)}")
        return stage2_dvf + effective_residual


class RegistrarResponseAnalyzer:
    """Collect CRS response metrics and write JSON/CSV/TensorBoard summaries."""

    def __init__(self, output_dir: Optional[Path] = None, writer: object = None) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.writer = writer
        self.rows: List[Dict[str, float]] = []

    def compute_rows(
        self,
        metadata: Sequence[ProbeMetadata],
        stage2_dvf: torch.Tensor,
        final_dvf: torch.Tensor,
        known_gt_dvf: torch.Tensor,
        roi_mask: torch.Tensor,
        eps: float = 1e-6,
    ) -> List[Dict[str, float]]:
        stage2_error = masked_epe_per_batch(stage2_dvf, known_gt_dvf, roi_mask)
        final_error = masked_epe_per_batch(final_dvf, known_gt_dvf, roi_mask)
        gt_norm = masked_epe_per_batch(known_gt_dvf, torch.zeros_like(known_gt_dvf), roi_mask)
        rows: List[Dict[str, float]] = []
        for index, meta in enumerate(metadata):
            s2_error = float(stage2_error[index].detach().cpu())
            s3_error = float(final_error[index].detach().cpu())
            norm = float(gt_norm[index].detach().cpu())
            blind_ratio = s2_error / (norm + eps)
            stage3_ratio = s3_error / (s2_error + eps)
            row = {
                **asdict(meta),
                "stage2_residual_error": s2_error,
                "stage2_blind_ratio": blind_ratio,
                "stage2_recovery_gain": 1.0 - blind_ratio,
                "stage3_final_error": s3_error,
                "stage3_error_ratio": stage3_ratio,
                "residual_recovery_gain": 1.0 - stage3_ratio,
            }
            rows.append(row)
        return rows

    def update(self, rows: Iterable[Dict[str, float]]) -> None:
        self.rows.extend(dict(row) for row in rows)

    def summary_by_bin(self) -> List[Dict[str, float]]:
        groups: Dict[Tuple[object, ...], List[Dict[str, float]]] = {}
        key_fields = ("target_label", "mode", "amplitude_bin", "support_radius_bin", "volume_bin")
        metric_fields = (
            "stage2_residual_error",
            "stage2_blind_ratio",
            "stage2_recovery_gain",
            "stage3_final_error",
            "stage3_error_ratio",
            "residual_recovery_gain",
        )
        for row in self.rows:
            key = tuple(row[field] for field in key_fields)
            groups.setdefault(key, []).append(row)

        summary = []
        for key, rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
            entry = {field: value for field, value in zip(key_fields, key)}
            entry["count"] = len(rows)
            for field in metric_fields:
                entry[field] = float(np.mean([float(row[field]) for row in rows]))
            summary.append(entry)
        return summary

    def save(self) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows_path = self.output_dir / "crs_response_rows.csv"
        if self.rows:
            fieldnames = list(self.rows[0].keys())
            with rows_path.open("w", newline="", encoding="utf-8") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)

        summary = self.summary_by_bin()
        summary_path = self.output_dir / "crs_response_summary.csv"
        if summary:
            with summary_path.open("w", newline="", encoding="utf-8") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=list(summary[0].keys()))
                writer.writeheader()
                writer.writerows(summary)

        payload = {"num_rows": len(self.rows), "summary": summary}
        (self.output_dir / "crs_response_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def log_tensorboard(self, step: int) -> None:
        if self.writer is None or not self.rows:
            return
        latest = self.rows[-1]
        for key in (
            "stage2_residual_error",
            "stage2_blind_ratio",
            "stage2_recovery_gain",
            "stage3_final_error",
            "stage3_error_ratio",
            "residual_recovery_gain",
        ):
            self.writer.add_scalar(f"crs/latest/{key}", float(latest[key]), step)
        summary = self.summary_by_bin()
        if not summary:
            return
        for key in ("stage2_recovery_gain", "residual_recovery_gain", "stage2_blind_ratio", "stage3_error_ratio"):
            self.writer.add_scalar(f"crs/mean/{key}", float(np.mean([row[key] for row in summary])), step)
        for row in summary:
            tag = (
                f"label_{row['target_label']}/{row['mode']}/"
                f"amp_{row['amplitude_bin']}/radius_{row['support_radius_bin']}/vol_{row['volume_bin']}"
            )
            self.writer.add_scalar(f"crs_bins/{tag}/stage2_recovery_gain", row["stage2_recovery_gain"], step)
            self.writer.add_scalar(f"crs_bins/{tag}/residual_recovery_gain", row["residual_recovery_gain"], step)


class BlindSpectrumSampler:
    """Adaptive sampler that favors Stage-2 blind and Stage-3 hard CRS bins."""

    def __init__(
        self,
        uniform_exploration: float = 0.20,
        temperature: float = 1.0,
        ema_decay: float = 0.90,
        seed: Optional[int] = None,
    ) -> None:
        self.uniform_exploration = float(np.clip(max(0.20, uniform_exploration), 0.0, 1.0))
        self.temperature = max(float(temperature), 1e-6)
        self.ema_decay = float(np.clip(ema_decay, 0.0, 0.999))
        self.rng = np.random.default_rng(seed)
        self.ema_blind_score: Dict[Tuple[int, str, str, str], float] = {}
        self.ema_stage3_error: Dict[Tuple[int, str, str, str], float] = {}

    @staticmethod
    def _key_from_values(label: int, mode: str, amplitude_mm: float, support_radius_mm: float) -> Tuple[int, str, str, str]:
        return (
            int(label),
            _normalize_mode(mode),
            AnatomicalCounterfactualProbeGenerator._amplitude_bin(float(amplitude_mm)),
            AnatomicalCounterfactualProbeGenerator._radius_bin(float(support_radius_mm)),
        )

    @staticmethod
    def _key_from_row(row: Dict[str, float]) -> Tuple[int, str, str, str]:
        return (
            int(row["target_label"]),
            _normalize_mode(str(row["mode"])),
            str(row["amplitude_bin"]),
            str(row["support_radius_bin"]),
        )

    def sample_specs(
        self,
        batch_size: int,
        labels: Sequence[int],
        modes: Sequence[str],
        amplitudes_mm: Sequence[float],
        support_radii_mm: Sequence[float],
        epoch: int,
        warmup_epochs: int,
    ) -> List[ProbeSpec]:
        candidates = [
            ProbeSpec(int(label), _normalize_mode(mode), float(amplitude), float(radius))
            for label in labels
            for mode in modes
            for amplitude in amplitudes_mm
            for radius in support_radii_mm
        ]
        if not candidates:
            raise ValueError("CRS sampler has no candidate bins")

        use_uniform = epoch < warmup_epochs or not self.ema_blind_score
        if use_uniform:
            indices = self.rng.integers(0, len(candidates), size=batch_size)
            return [candidates[int(index)] for index in indices]

        scores = []
        for candidate in candidates:
            key = self._key_from_values(
                candidate.target_label,
                candidate.mode,
                candidate.amplitude_mm,
                candidate.support_radius_mm,
            )
            score = self.ema_blind_score.get(key, 0.0) * self.ema_stage3_error.get(key, 0.0)
            scores.append(score)
        score_tensor = np.asarray(scores, dtype=np.float64)
        score_tensor = score_tensor - np.max(score_tensor)
        adaptive = np.exp(score_tensor / self.temperature)
        adaptive = adaptive / np.sum(adaptive)
        uniform = np.ones_like(adaptive) / len(adaptive)
        probability = self.uniform_exploration * uniform + (1.0 - self.uniform_exploration) * adaptive
        indices = self.rng.choice(len(candidates), size=batch_size, replace=True, p=probability)
        return [candidates[int(index)] for index in indices]

    def update_from_rows(self, rows: Iterable[Dict[str, float]]) -> None:
        for row in rows:
            key = self._key_from_row(row)
            blind = float(row["stage2_blind_ratio"])
            stage3_error = float(row["stage3_error_ratio"])
            if key not in self.ema_blind_score:
                self.ema_blind_score[key] = blind
                self.ema_stage3_error[key] = stage3_error
            else:
                decay = self.ema_decay
                self.ema_blind_score[key] = decay * self.ema_blind_score[key] + (1.0 - decay) * blind
                self.ema_stage3_error[key] = decay * self.ema_stage3_error[key] + (1.0 - decay) * stage3_error

    def state_dict(self) -> Dict[str, object]:
        return {
            "uniform_exploration": self.uniform_exploration,
            "temperature": self.temperature,
            "ema_decay": self.ema_decay,
            "ema_blind_score": {json.dumps(key): value for key, value in self.ema_blind_score.items()},
            "ema_stage3_error": {json.dumps(key): value for key, value in self.ema_stage3_error.items()},
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        restored_exploration = max(0.20, float(state.get("uniform_exploration", self.uniform_exploration)))
        self.uniform_exploration = float(np.clip(restored_exploration, 0.0, 1.0))
        self.temperature = max(float(state.get("temperature", self.temperature)), 1e-6)
        self.ema_decay = float(state.get("ema_decay", self.ema_decay))
        self.ema_blind_score = {
            tuple(json.loads(key)): float(value) for key, value in dict(state.get("ema_blind_score", {})).items()
        }
        self.ema_stage3_error = {
            tuple(json.loads(key)): float(value) for key, value in dict(state.get("ema_stage3_error", {})).items()
        }


class CounterfactualProbeCache:
    """Small optional tensor cache for CRS batches and frozen registrar outputs."""

    def __init__(self, cache_dir: Path, signature: Dict[str, str], seed: Optional[int] = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.signature = dict(signature)
        self.rng = np.random.default_rng(seed)

    def sample(self, device: torch.device) -> Optional[Dict[str, object]]:
        paths = sorted(self.cache_dir.glob("*.pt"))
        if not paths:
            return None
        order = self.rng.permutation(len(paths))
        for index in order[: min(len(paths), 8)]:
            payload = torch.load(paths[int(index)], map_location=device)
            if payload.get("signature") == self.signature:
                return payload["record"]
        return None

    def save(self, record: Dict[str, object]) -> Path:
        cpu_record = {}
        for key, value in record.items():
            if torch.is_tensor(value):
                cpu_record[key] = value.detach().cpu()
            elif isinstance(value, list) and value and isinstance(value[0], ProbeMetadata):
                cpu_record[key] = [asdict(item) for item in value]
            else:
                cpu_record[key] = value
        path = self.cache_dir / f"crs_{time.time_ns()}.pt"
        torch.save({"signature": self.signature, "record": cpu_record}, path)
        return path


def metadata_from_cache(rows: Sequence[Dict[str, object]]) -> List[ProbeMetadata]:
    return [ProbeMetadata(**row) for row in rows]
