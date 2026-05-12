"""Visual diagnostics for a prepared DIR-MUSA inference pair."""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual diagnostics for one prepared inference pair.")
    parser.add_argument("--pair-dir", required=True, help="Directory containing infer_prepared_pair outputs.")
    parser.add_argument("--data-root", default="data", help="Prepared data root containing images, seg_o, seg_b, metadata.")
    parser.add_argument("--moving-id", required=True, help="Moving case ID.")
    parser.add_argument("--fixed-id", required=True, help="Fixed case ID.")
    parser.add_argument("--output-dir", default=None, help="Output directory for figures. Default: <pair-dir>/viz.")
    parser.add_argument("--prefix", default=None, help="Output file prefix. Default: <moving-id>_to_<fixed-id>.")
    parser.add_argument("--low-dice-threshold", type=float, default=0.5, help="Threshold for low OAR Dice table.")
    parser.add_argument("--worst-k", type=int, default=12, help="Number of worst OARs to show.")
    parser.add_argument("--jacobian-stride", type=int, default=2, help="Subsampling stride for Jacobian diagnostics.")
    parser.add_argument("--dpi", type=int, default=160, help="PNG resolution.")
    return parser.parse_args()


def load_npy(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def case_path(data_root: Path, folder: str, case_id: str) -> Path:
    return data_root / folder / f"{case_id}.npy"


def find_pair_file(pair_dir: Path, prefix: str, suffix: str) -> Path:
    path = pair_dir / f"{prefix}_{suffix}"
    if path.is_file():
        return path
    candidates = sorted(pair_dir.glob(f"*_{suffix}"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not resolve {suffix} in {pair_dir}; expected {path}")


def central_slices(seg: Optional[np.ndarray], shape: Sequence[int]) -> Tuple[int, int, int]:
    if seg is not None and np.any(seg > 0):
        coords = np.argwhere(seg > 0)
        center = np.rint(coords.mean(axis=0)).astype(int)
        return tuple(int(np.clip(center[i], 0, shape[i] - 1)) for i in range(3))
    return tuple(dim // 2 for dim in shape)


def slice2d(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return array[index, :, :].T
    if axis == 1:
        return array[:, index, :].T
    if axis == 2:
        return array[:, :, index].T
    raise ValueError(f"Invalid axis: {axis}")


def show_image(ax, array2d: np.ndarray, title: str, vmin: float = 0.0, vmax: float = 1.0, cmap: str = "gray") -> None:
    ax.imshow(array2d, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def overlay_contours(ax, fixed_seg: np.ndarray, moving_seg: Optional[np.ndarray], warped_seg: Optional[np.ndarray]) -> None:
    if fixed_seg is not None and np.any(fixed_seg > 0):
        ax.contour(fixed_seg > 0, levels=[0.5], colors=["lime"], linewidths=0.8)
    if moving_seg is not None and np.any(moving_seg > 0):
        ax.contour(moving_seg > 0, levels=[0.5], colors=["red"], linewidths=0.8)
    if warped_seg is not None and np.any(warped_seg > 0):
        ax.contour(warped_seg > 0, levels=[0.5], colors=["cyan"], linewidths=0.8)


def load_metrics(pair_dir: Path, prefix: str) -> Dict[str, object]:
    path = pair_dir / f"{prefix}_metrics.json"
    if not path.is_file():
        candidates = sorted(pair_dir.glob("*_metrics.json"))
        if len(candidates) == 1:
            path = candidates[0]
        else:
            raise FileNotFoundError(f"Could not resolve metrics JSON in {pair_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_label_names(data_root: Path, moving_id: str, fixed_id: str) -> Dict[int, str]:
    label_names: Dict[int, str] = {}
    for case_id in (fixed_id, moving_id):
        path = data_root / "metadata" / f"{case_id}.json"
        if not path.is_file():
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        for name, label in meta.get("label_map", {}).items():
            label_names.setdefault(int(label), name)
    return label_names


def metric_rows(metrics: Dict[str, object], label_names: Dict[int, str]) -> List[Dict[str, object]]:
    dice = metrics.get("dice_seg_o", {})
    rows = []
    for label_str, values in dice.get("per_label", {}).items():
        label = int(label_str)
        rows.append(
            {
                "label": label,
                "name": label_names.get(label, f"label_{label}"),
                "before": float(values["before"]),
                "after": float(values["after"]),
                "delta": float(values["delta"]),
            }
        )
    rows.sort(key=lambda row: row["after"])
    return rows


def write_low_dice_csv(path: Path, rows: List[Dict[str, object]], threshold: float) -> None:
    low_rows = [row for row in rows if row["after"] < threshold]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["label", "name", "before", "after", "delta"])
        writer.writeheader()
        writer.writerows(low_rows)


def plot_ct_overview(
    out_path: Path,
    moving: np.ndarray,
    fixed: np.ndarray,
    deformed: np.ndarray,
    seg_focus: Optional[np.ndarray],
    dpi: int,
) -> None:
    indices = central_slices(seg_focus, fixed.shape)
    axes_names = ["sagittal", "coronal", "axial"]
    fig, axs = plt.subplots(3, 5, figsize=(13, 8), constrained_layout=True)
    for axis, index in enumerate(indices):
        fixed_sl = slice2d(fixed, axis, index)
        moving_sl = slice2d(moving, axis, index)
        deformed_sl = slice2d(deformed, axis, index)
        before_diff = np.abs(moving_sl - fixed_sl)
        after_diff = np.abs(deformed_sl - fixed_sl)
        for col, (image, title, cmap, vmax) in enumerate(
            [
                (fixed_sl, f"{axes_names[axis]} fixed", "gray", 1.0),
                (moving_sl, "moving", "gray", 1.0),
                (deformed_sl, "deformed", "gray", 1.0),
                (before_diff, "|moving-fixed|", "magma", 0.5),
                (after_diff, "|deformed-fixed|", "magma", 0.5),
            ]
        ):
            show_image(axs[axis, col], image, title, vmax=vmax, cmap=cmap)
    fig.suptitle("CT alignment overview", fontsize=12)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_seg_overlay(
    out_path: Path,
    fixed_img: np.ndarray,
    moving_seg: np.ndarray,
    fixed_seg: np.ndarray,
    warped_seg: np.ndarray,
    title: str,
    dpi: int,
) -> None:
    indices = central_slices(fixed_seg, fixed_img.shape)
    axes_names = ["sagittal", "coronal", "axial"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, index in enumerate(indices):
        show_image(axs[axis], slice2d(fixed_img, axis, index), f"{axes_names[axis]} {title}")
        overlay_contours(
            axs[axis],
            slice2d(fixed_seg, axis, index),
            slice2d(moving_seg, axis, index),
            slice2d(warped_seg, axis, index),
        )
    fig.suptitle("Contours: fixed=green, moving=red, warped=cyan", fontsize=11)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def jacobian_determinant(dvf_chwd: np.ndarray, stride: int) -> np.ndarray:
    dvf = dvf_chwd[:, ::stride, ::stride, ::stride]
    gradients = [np.gradient(dvf[c], edge_order=1) for c in range(3)]
    j00 = 1.0 + gradients[0][0]
    j01 = gradients[0][1]
    j02 = gradients[0][2]
    j10 = gradients[1][0]
    j11 = 1.0 + gradients[1][1]
    j12 = gradients[1][2]
    j20 = gradients[2][0]
    j21 = gradients[2][1]
    j22 = 1.0 + gradients[2][2]
    return (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )


def plot_flow(out_path: Path, dvf_chwd: np.ndarray, fixed_seg_b: np.ndarray, stride: int, dpi: int) -> Dict[str, float]:
    mag = np.sqrt(np.sum(dvf_chwd**2, axis=0))
    jac = jacobian_determinant(dvf_chwd, stride=max(1, stride))
    indices = central_slices(fixed_seg_b, mag.shape)
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, index in enumerate(indices):
        image = axs[0, axis].imshow(slice2d(mag, axis, index), cmap="viridis", origin="lower")
        axs[0, axis].set_title(f"DVF magnitude axis={axis}", fontsize=9)
        axs[0, axis].set_xticks([])
        axs[0, axis].set_yticks([])
        fig.colorbar(image, ax=axs[0, axis], fraction=0.046)

    jac_indices = tuple(min(index // max(1, stride), jac.shape[axis] - 1) for axis, index in enumerate(indices))
    for axis, index in enumerate(jac_indices):
        image = axs[1, axis].imshow(slice2d(jac, axis, index), cmap="coolwarm", origin="lower", vmin=0.0, vmax=2.0)
        axs[1, axis].set_title(f"Jacobian det axis={axis}", fontsize=9)
        axs[1, axis].set_xticks([])
        axs[1, axis].set_yticks([])
        fig.colorbar(image, ax=axs[1, axis], fraction=0.046)
    fig.suptitle("Flow diagnostics", fontsize=12)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return {
        "flow_mag_mean": float(np.mean(mag)),
        "flow_mag_p95": float(np.percentile(mag, 95)),
        "flow_mag_max": float(np.max(mag)),
        "jacobian_min": float(np.min(jac)),
        "jacobian_p01": float(np.percentile(jac, 1)),
        "jacobian_p99": float(np.percentile(jac, 99)),
        "jacobian_max": float(np.max(jac)),
        "jacobian_nonpositive_fraction": float(np.mean(jac <= 0)),
    }


def plot_oar_bars(out_path: Path, rows: List[Dict[str, object]], worst_k: int, dpi: int) -> None:
    shown = rows[:worst_k]
    if not shown:
        return
    names = [f"{row['label']} {row['name']}" for row in shown]
    before = [row["before"] for row in shown]
    after = [row["after"] for row in shown]
    y = np.arange(len(shown))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(shown))), constrained_layout=True)
    ax.barh(y + 0.18, before, height=0.35, label="before", color="#d95f02")
    ax.barh(y - 0.18, after, height=0.35, label="after", color="#1b9e77")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Dice")
    ax.set_title(f"Worst {len(shown)} OAR Dice after registration")
    ax.legend()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def crop_bounds(mask: np.ndarray, margin: int = 12) -> Optional[Tuple[slice, slice]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + margin + 1, mask.shape)
    return slice(mins[0], maxs[0]), slice(mins[1], maxs[1])


def plot_worst_oar_montage(
    out_path: Path,
    fixed_img: np.ndarray,
    moving_seg: np.ndarray,
    fixed_seg: np.ndarray,
    warped_seg: np.ndarray,
    rows: List[Dict[str, object]],
    worst_k: int,
    dpi: int,
) -> None:
    shown = rows[: min(worst_k, 8)]
    if not shown:
        return
    fig, axs = plt.subplots(len(shown), 3, figsize=(10, max(3, 2.2 * len(shown))), constrained_layout=True)
    if len(shown) == 1:
        axs = axs[None, :]
    for row_idx, row in enumerate(shown):
        label = int(row["label"])
        fixed_mask = fixed_seg == label
        if not np.any(fixed_mask):
            fixed_mask = np.logical_or(moving_seg == label, warped_seg == label)
        coords = np.argwhere(fixed_mask)
        z = int(np.rint(coords[:, 2].mean())) if coords.size else fixed_img.shape[2] // 2
        bounds = crop_bounds(fixed_mask[:, :, z])
        images = [
            (moving_seg == label, "moving red"),
            (warped_seg == label, "warped cyan"),
            (fixed_seg == label, "fixed green"),
        ]
        base = fixed_img[:, :, z].T
        if bounds is not None:
            sx, sy = bounds
            base = fixed_img[sx, sy, z].T
        for col, (mask, title) in enumerate(images):
            ax = axs[row_idx, col]
            show_image(ax, base, f"{row['name']} {title}\nDice {row['after']:.3f}")
            mask2d = mask[:, :, z]
            fixed2d = fixed_seg[:, :, z] == label
            if bounds is not None:
                sx, sy = bounds
                mask2d = mask2d[sx, sy]
                fixed2d = fixed2d[sx, sy]
            if col != 2 and np.any(mask2d):
                ax.contour(mask2d.T, levels=[0.5], colors=["red" if col == 0 else "cyan"], linewidths=0.9)
            if np.any(fixed2d):
                ax.contour(fixed2d.T, levels=[0.5], colors=["lime"], linewidths=0.8)
    fig.suptitle("Worst OAR local overlays on fixed CT", fontsize=12)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pair_dir = Path(args.pair_dir)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else pair_dir / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"{args.moving_id}_to_{args.fixed_id}"

    moving = load_npy(case_path(data_root, "images", args.moving_id))
    fixed = load_npy(case_path(data_root, "images", args.fixed_id))
    moving_seg_o = load_npy(case_path(data_root, "seg_o", args.moving_id))
    fixed_seg_o = load_npy(case_path(data_root, "seg_o", args.fixed_id))
    moving_seg_b = load_npy(case_path(data_root, "seg_b", args.moving_id))
    fixed_seg_b = load_npy(case_path(data_root, "seg_b", args.fixed_id))

    deformed = load_npy(find_pair_file(pair_dir, prefix, "deformed_img.npy"))
    warped_seg_o = load_npy(find_pair_file(pair_dir, prefix, "deformed_seg_o.npy"))
    warped_seg_b = load_npy(find_pair_file(pair_dir, prefix, "deformed_seg_b.npy"))
    dvf_chwd = load_npy(find_pair_file(pair_dir, prefix, "dvf_chwd.npy"))
    metrics = load_metrics(pair_dir, prefix)
    label_names = load_label_names(data_root, args.moving_id, args.fixed_id)
    rows = metric_rows(metrics, label_names)

    plot_ct_overview(output_dir / "01_ct_alignment_overview.png", moving, fixed, deformed, fixed_seg_b, args.dpi)
    plot_seg_overlay(output_dir / "02_seg_o_overlay.png", fixed, moving_seg_o, fixed_seg_o, warped_seg_o, "seg_o", args.dpi)
    plot_seg_overlay(output_dir / "03_seg_b_overlay.png", fixed, moving_seg_b, fixed_seg_b, warped_seg_b, "seg_b", args.dpi)
    flow_stats = plot_flow(output_dir / "04_flow_diagnostics.png", dvf_chwd, fixed_seg_b, args.jacobian_stride, args.dpi)
    plot_oar_bars(output_dir / "05_low_oar_dice_bar.png", rows, args.worst_k, args.dpi)
    plot_worst_oar_montage(
        output_dir / "06_worst_oar_local_overlays.png",
        fixed,
        moving_seg_o,
        fixed_seg_o,
        warped_seg_o,
        rows,
        args.worst_k,
        args.dpi,
    )
    write_low_dice_csv(output_dir / "low_oar_dice.csv", rows, args.low_dice_threshold)

    report = {
        "moving_id": args.moving_id,
        "fixed_id": args.fixed_id,
        "pair_dir": str(pair_dir),
        "flow_stats": flow_stats,
        "worst_oars": rows[: args.worst_k],
        "low_dice_threshold": args.low_dice_threshold,
        "low_dice_count": sum(1 for row in rows if row["after"] < args.low_dice_threshold),
    }
    (output_dir / "viz_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote visual diagnostics to {output_dir}")
    print(
        "[INFO] Flow: "
        f"mag_p95={flow_stats['flow_mag_p95']:.3f}, "
        f"mag_max={flow_stats['flow_mag_max']:.3f}, "
        f"jac_min={flow_stats['jacobian_min']:.3f}, "
        f"jac<=0 frac={flow_stats['jacobian_nonpositive_fraction']:.6f}"
    )
    if rows:
        print("[INFO] Worst OAR Dice after registration:")
        for row in rows[: min(args.worst_k, 8)]:
            print(f"[INFO] label={row['label']:>3} {row['name']}: {row['before']:.4f} -> {row['after']:.4f}")


if __name__ == "__main__":
    main()
