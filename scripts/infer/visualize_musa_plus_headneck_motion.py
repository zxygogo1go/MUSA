"""Visualize global head-neck motion for MUSA Stage2 vs MUSA+ Stage3.

The small-OAR visualizer focuses on local organ alignment. This script instead
renders whole head-neck motion diagnostics: CT posture changes, foreground/bone
contours, displacement magnitude, vector fields, and deformed grids.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create head-neck motion diagnostics for Stage2 vs Stage3.")
    parser.add_argument("--pair-dir", required=True, help="Directory from infer_musa_plus_prepared_pair.py.")
    parser.add_argument("--data-root", default="data", help="Prepared data root with images, seg_o, seg_b.")
    parser.add_argument("--moving-id", required=True, help="Moving case ID.")
    parser.add_argument("--fixed-id", required=True, help="Fixed case ID.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: <pair-dir>/viz_headneck_motion.")
    parser.add_argument("--prefix", default=None, help="Output prefix. Default: <moving-id>_to_<fixed-id>.")
    parser.add_argument("--grid-step", type=int, default=12, help="Voxel spacing for deformed grid lines.")
    parser.add_argument("--quiver-stride", type=int, default=12, help="Stride for displacement arrows.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution.")
    return parser.parse_args()


def load_npy(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def npy_path(data_root: Path, folder: str, case_id: str) -> Path:
    return data_root / folder / f"{case_id}.npy"


def pair_file(pair_dir: Path, prefix: str, suffix: str) -> Path:
    path = pair_dir / f"{prefix}_{suffix}"
    if path.is_file():
        return path
    candidates = sorted(pair_dir.glob(f"*_{suffix}"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not resolve {suffix} in {pair_dir}; expected {path}")


def optional_pair_file(pair_dir: Path, prefix: str, suffix: str) -> Optional[Path]:
    try:
        return pair_file(pair_dir, prefix, suffix)
    except FileNotFoundError:
        return None


def slice2d(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return array[index, :, :].T
    if axis == 1:
        return array[:, index, :].T
    if axis == 2:
        return array[:, :, index].T
    raise ValueError(f"Invalid axis: {axis}")


def center_slices(mask: np.ndarray, shape: Sequence[int]) -> Tuple[int, int, int]:
    if np.any(mask):
        coords = np.argwhere(mask)
        center = np.rint(coords.mean(axis=0)).astype(int)
        return tuple(int(np.clip(center[axis], 0, shape[axis] - 1)) for axis in range(3))
    return tuple(int(size // 2) for size in shape)


def show_image(ax, image: np.ndarray, title: str, cmap: str = "gray", vmin: float = 0.0, vmax: float = 1.0) -> None:
    ax.imshow(image, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def contour(ax, mask2d: np.ndarray, color: str, linewidth: float = 0.9, linestyle: str = "solid") -> None:
    if np.any(mask2d):
        ax.contour(mask2d > 0, levels=[0.5], colors=[color], linewidths=linewidth, linestyles=linestyle)


def foreground(seg_o: np.ndarray, seg_b: Optional[np.ndarray] = None) -> np.ndarray:
    mask = seg_o > 0
    if seg_b is not None:
        mask = np.logical_or(mask, seg_b > 0)
    return mask


def load_metrics(pair_dir: Path, prefix: str) -> Dict[str, object]:
    path = pair_file(pair_dir, prefix, "musa_plus_metrics.json")
    return json.loads(path.read_text(encoding="utf-8"))


def dvf_magnitude(dvf_chwd: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(dvf_chwd.astype(np.float32) ** 2, axis=0))


def plane_components(dvf_chwd: np.ndarray, axis: int, index: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return in-plane displacement components matching `slice2d` display coordinates."""

    if axis == 0:
        return dvf_chwd[1, index, :, :].T, dvf_chwd[2, index, :, :].T
    if axis == 1:
        return dvf_chwd[0, :, index, :].T, dvf_chwd[2, :, index, :].T
    if axis == 2:
        return dvf_chwd[0, :, :, index].T, dvf_chwd[1, :, :, index].T
    raise ValueError(f"Invalid axis: {axis}")


def plot_global_ct_motion(
    path: Path,
    moving: np.ndarray,
    fixed: np.ndarray,
    stage1: Optional[np.ndarray],
    stage2: np.ndarray,
    stage3: np.ndarray,
    focus: np.ndarray,
    dpi: int,
) -> None:
    indices = center_slices(focus, fixed.shape)
    axis_names = ["Sagittal", "Coronal", "Axial"]
    panels = [
        ("Fixed", fixed),
        ("Moving", moving),
        ("Stage1", stage1),
        ("MUSA Stage2", stage2),
        ("Our Stage3", stage3),
    ]
    fig, axs = plt.subplots(3, 7, figsize=(18, 9.5), constrained_layout=True)
    for row_idx, (axis, index) in enumerate(enumerate(indices)):
        fixed_sl = slice2d(fixed, axis, index)
        moving_sl = slice2d(moving, axis, index)
        stage2_sl = slice2d(stage2, axis, index)
        stage3_sl = slice2d(stage3, axis, index)
        for col_idx, (title, image) in enumerate(panels):
            ax = axs[row_idx, col_idx]
            if image is None:
                ax.axis("off")
                continue
            show_image(ax, slice2d(image, axis, index), f"{axis_names[axis]} {title}")
        diff_panels = [
            ("|Moving-Fixed|", np.abs(moving_sl - fixed_sl)),
            ("|Stage2-Fixed|", np.abs(stage2_sl - fixed_sl)),
            ("|Stage3-Fixed|", np.abs(stage3_sl - fixed_sl)),
        ]
        for offset, (title, image) in enumerate(diff_panels, start=5):
            show_image(axs[row_idx, offset], image, f"{axis_names[axis]} {title}", cmap="magma", vmax=0.45)
    fig.suptitle("Global CT posture/alignment: MUSA Stage2 vs proposed Stage3", fontsize=13)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_large_structure_contours(
    path: Path,
    fixed_img: np.ndarray,
    moving_fg: np.ndarray,
    fixed_fg: np.ndarray,
    stage2_fg: np.ndarray,
    stage3_fg: np.ndarray,
    moving_bone: np.ndarray,
    fixed_bone: np.ndarray,
    stage2_bone: np.ndarray,
    stage3_bone: np.ndarray,
    focus: np.ndarray,
    dpi: int,
) -> None:
    indices = center_slices(focus, fixed_img.shape)
    axis_names = ["Sagittal", "Coronal", "Axial"]
    rows = [
        ("All OAR/bone foreground", moving_fg, fixed_fg, stage2_fg, stage3_fg),
        ("Bone / mandible foreground", moving_bone, fixed_bone, stage2_bone, stage3_bone),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for row_idx, (row_title, moving_mask, fixed_mask, stage2_mask, stage3_mask) in enumerate(rows):
        for col_idx, (axis, index) in enumerate(enumerate(indices)):
            ax = axs[row_idx, col_idx]
            show_image(ax, slice2d(fixed_img, axis, index), f"{axis_names[axis]} {row_title}")
            contour(ax, slice2d(fixed_mask, axis, index), "lime", linewidth=0.9)
            contour(ax, slice2d(moving_mask, axis, index), "red", linewidth=0.7)
            contour(ax, slice2d(stage2_mask, axis, index), "cyan", linewidth=0.8)
            contour(ax, slice2d(stage3_mask, axis, index), "yellow", linewidth=0.8)
    fig.suptitle("Large head-neck motion contours: fixed=green, moving=red, Stage2=cyan, Stage3=yellow", fontsize=12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_dvf_magnitude(
    path: Path,
    fixed_img: np.ndarray,
    stage2_dvf: np.ndarray,
    stage3_dvf: np.ndarray,
    focus: np.ndarray,
    dpi: int,
) -> None:
    residual = stage3_dvf - stage2_dvf
    mags = [
        ("Stage2 DVF magnitude", dvf_magnitude(stage2_dvf)),
        ("Stage3 DVF magnitude", dvf_magnitude(stage3_dvf)),
        ("Stage3 residual magnitude", dvf_magnitude(residual)),
    ]
    indices = center_slices(focus, fixed_img.shape)
    axis_names = ["Sagittal", "Coronal", "Axial"]
    vmax = max(float(np.percentile(mag, 99)) for _, mag in mags[:2])
    residual_vmax = max(float(np.percentile(mags[2][1], 99)), 1e-6)
    fig, axs = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for row_idx, (title, mag) in enumerate(mags):
        for col_idx, (axis, index) in enumerate(enumerate(indices)):
            ax = axs[row_idx, col_idx]
            show_image(ax, slice2d(fixed_img, axis, index), f"{axis_names[axis]} {title}", vmax=1.0)
            limit = residual_vmax if row_idx == 2 else vmax
            image = ax.imshow(slice2d(mag, axis, index), cmap="viridis", origin="lower", alpha=0.72, vmin=0.0, vmax=limit)
            fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle("Displacement magnitude over fixed CT", fontsize=12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_quiver_and_grid(
    path: Path,
    fixed_img: np.ndarray,
    stage2_dvf: np.ndarray,
    stage3_dvf: np.ndarray,
    focus: np.ndarray,
    grid_step: int,
    quiver_stride: int,
    dpi: int,
) -> None:
    indices = center_slices(focus, fixed_img.shape)
    axis_names = ["Sagittal", "Coronal", "Axial"]
    fields = [("MUSA Stage2", stage2_dvf, "cyan"), ("Our Stage3", stage3_dvf, "yellow")]
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    for row_idx, (field_title, dvf, color) in enumerate(fields):
        for col_idx, (axis, index) in enumerate(enumerate(indices)):
            ax = axs[row_idx, col_idx]
            base = slice2d(fixed_img, axis, index)
            show_image(ax, base, f"{axis_names[axis]} {field_title}: arrows + deformed grid")
            u, v = plane_components(dvf, axis, index)
            rows, cols = u.shape

            for x in range(0, cols, max(1, grid_step)):
                y_values = np.arange(rows)
                x_values = np.full(rows, x)
                ax.plot(x_values, y_values, color="white", linewidth=0.25, alpha=0.25)
                ax.plot(x_values + u[:, x], y_values + v[:, x], color=color, linewidth=0.45, alpha=0.7)
            for y in range(0, rows, max(1, grid_step)):
                x_values = np.arange(cols)
                y_values = np.full(cols, y)
                ax.plot(x_values, y_values, color="white", linewidth=0.25, alpha=0.25)
                ax.plot(x_values + u[y, :], y_values + v[y, :], color=color, linewidth=0.45, alpha=0.7)

            yy, xx = np.mgrid[0:rows:max(1, quiver_stride), 0:cols:max(1, quiver_stride)]
            ax.quiver(
                xx,
                yy,
                u[yy, xx],
                v[yy, xx],
                color="red",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.0022,
                alpha=0.65,
            )
    fig.suptitle("Large-motion deformation field: white=regular grid, colored=deformed grid", fontsize=12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    if not np.any(mask):
        return None
    return np.argwhere(mask).mean(axis=0)


def centroid_distance(mask: np.ndarray, fixed_mask: np.ndarray) -> float:
    source = centroid(mask)
    target = centroid(fixed_mask)
    if source is None or target is None:
        return float("nan")
    return float(np.linalg.norm(source - target))


def dice(mask: np.ndarray, fixed_mask: np.ndarray, eps: float = 1e-5) -> float:
    denom = float(mask.sum() + fixed_mask.sum())
    if denom <= eps:
        return 1.0
    return float((2.0 * np.logical_and(mask, fixed_mask).sum() + eps) / (denom + eps))


def dvf_stats(name: str, dvf: np.ndarray) -> Dict[str, Union[float, str]]:
    mag = dvf_magnitude(dvf)
    return {
        "structure": name,
        "centroid_before": "",
        "centroid_stage2": "",
        "centroid_stage3": "",
        "dice_before": "",
        "dice_stage2": "",
        "dice_stage3": "",
        "dvf_mean": float(np.mean(mag)),
        "dvf_p95": float(np.percentile(mag, 95)),
        "dvf_max": float(np.max(mag)),
    }


def write_motion_stats(
    path: Path,
    moving_fg: np.ndarray,
    fixed_fg: np.ndarray,
    stage2_fg: np.ndarray,
    stage3_fg: np.ndarray,
    moving_bone: np.ndarray,
    fixed_bone: np.ndarray,
    stage2_bone: np.ndarray,
    stage3_bone: np.ndarray,
    stage2_dvf: np.ndarray,
    stage3_dvf: np.ndarray,
) -> None:
    rows: List[Dict[str, Union[float, str]]] = []
    for name, moving_mask, fixed_mask, stage2_mask, stage3_mask in [
        ("all_oar_bone_foreground", moving_fg, fixed_fg, stage2_fg, stage3_fg),
        ("bone_mandible_foreground", moving_bone, fixed_bone, stage2_bone, stage3_bone),
    ]:
        rows.append(
            {
                "structure": name,
                "centroid_before": centroid_distance(moving_mask, fixed_mask),
                "centroid_stage2": centroid_distance(stage2_mask, fixed_mask),
                "centroid_stage3": centroid_distance(stage3_mask, fixed_mask),
                "dice_before": dice(moving_mask, fixed_mask),
                "dice_stage2": dice(stage2_mask, fixed_mask),
                "dice_stage3": dice(stage3_mask, fixed_mask),
                "dvf_mean": "",
                "dvf_p95": "",
                "dvf_max": "",
            }
        )
    rows.append(dvf_stats("stage2_dvf_global", stage2_dvf))
    rows.append(dvf_stats("stage3_dvf_global", stage3_dvf))
    rows.append(dvf_stats("stage3_residual_dvf_global", stage3_dvf - stage2_dvf))

    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "structure",
                "centroid_before",
                "centroid_stage2",
                "centroid_stage3",
                "dice_before",
                "dice_stage2",
                "dice_stage3",
                "dvf_mean",
                "dvf_p95",
                "dvf_max",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    pair_dir = Path(args.pair_dir)
    data_root = Path(args.data_root)
    prefix = args.prefix or f"{args.moving_id}_to_{args.fixed_id}"
    output_dir = Path(args.output_dir) if args.output_dir else pair_dir / "viz_headneck_motion"
    output_dir.mkdir(parents=True, exist_ok=True)

    moving = load_npy(npy_path(data_root, "images", args.moving_id))
    fixed = load_npy(npy_path(data_root, "images", args.fixed_id))
    moving_seg_o = load_npy(npy_path(data_root, "seg_o", args.moving_id))
    fixed_seg_o = load_npy(npy_path(data_root, "seg_o", args.fixed_id))
    moving_seg_b = load_npy(npy_path(data_root, "seg_b", args.moving_id))
    fixed_seg_b = load_npy(npy_path(data_root, "seg_b", args.fixed_id))

    stage1_path = optional_pair_file(pair_dir, prefix, "stage1_deformed_img.npy")
    stage1 = load_npy(stage1_path) if stage1_path is not None else None
    stage2 = load_npy(pair_file(pair_dir, prefix, "stage2_deformed_img.npy"))
    stage3 = load_npy(pair_file(pair_dir, prefix, "musa_plus_deformed_img.npy"))
    stage2_seg_o = load_npy(pair_file(pair_dir, prefix, "stage2_deformed_seg_o.npy"))
    stage3_seg_o = load_npy(pair_file(pair_dir, prefix, "musa_plus_deformed_seg_o.npy"))
    stage2_seg_b = load_npy(pair_file(pair_dir, prefix, "stage2_deformed_seg_b.npy"))
    stage3_seg_b = load_npy(pair_file(pair_dir, prefix, "musa_plus_deformed_seg_b.npy"))
    stage2_dvf = load_npy(pair_file(pair_dir, prefix, "stage2_dvf_chwd.npy"))
    stage3_dvf = load_npy(pair_file(pair_dir, prefix, "musa_plus_dvf_chwd.npy"))

    moving_fg = foreground(moving_seg_o, moving_seg_b)
    fixed_fg = foreground(fixed_seg_o, fixed_seg_b)
    stage2_fg = foreground(stage2_seg_o, stage2_seg_b)
    stage3_fg = foreground(stage3_seg_o, stage3_seg_b)

    plot_global_ct_motion(
        output_dir / "01_global_ct_motion_stage1_stage2_stage3.png",
        moving,
        fixed,
        stage1,
        stage2,
        stage3,
        fixed_fg,
        args.dpi,
    )
    plot_large_structure_contours(
        output_dir / "02_large_structure_motion_contours.png",
        fixed,
        moving_fg,
        fixed_fg,
        stage2_fg,
        stage3_fg,
        moving_seg_b > 0,
        fixed_seg_b > 0,
        stage2_seg_b > 0,
        stage3_seg_b > 0,
        fixed_fg,
        args.dpi,
    )
    plot_dvf_magnitude(
        output_dir / "03_dvf_magnitude_stage2_stage3_residual.png",
        fixed,
        stage2_dvf,
        stage3_dvf,
        fixed_fg,
        args.dpi,
    )
    plot_quiver_and_grid(
        output_dir / "04_deformation_grid_and_vectors.png",
        fixed,
        stage2_dvf,
        stage3_dvf,
        fixed_fg,
        args.grid_step,
        args.quiver_stride,
        args.dpi,
    )
    write_motion_stats(
        output_dir / "headneck_motion_stats.csv",
        moving_fg,
        fixed_fg,
        stage2_fg,
        stage3_fg,
        moving_seg_b > 0,
        fixed_seg_b > 0,
        stage2_seg_b > 0,
        stage3_seg_b > 0,
        stage2_dvf,
        stage3_dvf,
    )

    report = {
        "moving_id": args.moving_id,
        "fixed_id": args.fixed_id,
        "figures": [
            "01_global_ct_motion_stage1_stage2_stage3.png",
            "02_large_structure_motion_contours.png",
            "03_dvf_magnitude_stage2_stage3_residual.png",
            "04_deformation_grid_and_vectors.png",
        ],
        "stats": "headneck_motion_stats.csv",
    }
    metrics_path = optional_pair_file(pair_dir, prefix, "musa_plus_metrics.json")
    if metrics_path is not None:
        report["metrics_json"] = str(metrics_path)
    (output_dir / "headneck_motion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote head-neck motion visual diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
