"""Visualize Stage-2 vs MUSA+ Stage-3 small-OAR alignment for one pair."""

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


DEFAULT_SMALL_OAR_NAMES = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create small-OAR Stage2 vs MUSA+ Stage3 visual diagnostics.")
    parser.add_argument("--pair-dir", required=True, help="Directory from infer_musa_plus_prepared_pair.py.")
    parser.add_argument("--data-root", default="data", help="Prepared data root with images, seg_o, metadata.")
    parser.add_argument("--moving-id", required=True, help="Moving case ID.")
    parser.add_argument("--fixed-id", required=True, help="Fixed case ID.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: <pair-dir>/viz_musa_plus.")
    parser.add_argument("--prefix", default=None, help="Output prefix. Default: <moving-id>_to_<fixed-id>.")
    parser.add_argument("--small-oar-labels", default=None, help="Optional comma-separated small-OAR labels.")
    parser.add_argument(
        "--small-oar-names",
        default=",".join(DEFAULT_SMALL_OAR_NAMES),
        help="Small-OAR structure names used when resolving labels from metadata.",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Number of labels to show in local montages.")
    parser.add_argument("--margin", type=int, default=10, help="Voxel margin for local crops.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution.")
    return parser.parse_args()


def load_npy(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def load_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_label_list(value: Optional[str]) -> List[int]:
    if value is None or not value.strip():
        return []
    return sorted({int(part.strip()) for part in value.split(",") if part.strip()})


def load_label_names(data_root: Path, moving_id: str, fixed_id: str) -> Dict[int, str]:
    label_names: Dict[int, str] = {}
    for case_id in (fixed_id, moving_id):
        path = data_root / "metadata" / f"{case_id}.json"
        if not path.is_file():
            continue
        meta = load_json(path)
        for name, label in meta.get("label_map", {}).items():
            label_names.setdefault(int(label), str(name))
    return label_names


def resolve_small_labels(
    metrics: Dict[str, object],
    label_names: Dict[int, str],
    explicit_labels: Optional[str],
    small_oar_names: str,
) -> List[int]:
    labels = parse_label_list(explicit_labels)
    if labels:
        return labels
    metric_labels = metrics.get("small_oar_per_label", {}).get("labels", [])
    if metric_labels:
        return [int(label) for label in metric_labels]
    requested = {name.strip() for name in small_oar_names.split(",") if name.strip()}
    name_to_label = {name: label for label, name in label_names.items()}
    return sorted(int(name_to_label[name]) for name in requested if name in name_to_label)


def pair_file(pair_dir: Path, prefix: str, suffix: str) -> Path:
    path = pair_dir / f"{prefix}_{suffix}"
    if path.is_file():
        return path
    candidates = sorted(pair_dir.glob(f"*_{suffix}"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not resolve {suffix} in {pair_dir}; expected {path}")


def slice2d(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return array[index, :, :].T
    if axis == 1:
        return array[:, index, :].T
    if axis == 2:
        return array[:, :, index].T
    raise ValueError(f"Invalid axis: {axis}")


def show_image(ax, image: np.ndarray, title: str, cmap: str = "gray", vmin: float = 0.0, vmax: float = 1.0) -> None:
    ax.imshow(image, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def contour(ax, mask2d: np.ndarray, color: str, linewidth: float = 0.9, linestyle: str = "solid") -> None:
    if np.any(mask2d):
        ax.contour(mask2d > 0, levels=[0.5], colors=[color], linewidths=linewidth, linestyles=linestyle)


def label_mask(seg: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    if not labels:
        return np.zeros_like(seg, dtype=bool)
    return np.isin(seg, np.asarray(labels, dtype=seg.dtype))


def center_slices(mask: np.ndarray, shape: Sequence[int]) -> Tuple[int, int, int]:
    if np.any(mask):
        coords = np.argwhere(mask)
        center = np.rint(coords.mean(axis=0)).astype(int)
        return tuple(int(np.clip(center[axis], 0, shape[axis] - 1)) for axis in range(3))
    return tuple(int(size // 2) for size in shape)


def crop_bounds(mask2d: np.ndarray, shape2d: Sequence[int], margin: int) -> Tuple[slice, slice]:
    if not np.any(mask2d):
        return slice(0, shape2d[0]), slice(0, shape2d[1])
    coords = np.argwhere(mask2d)
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + margin + 1, np.asarray(shape2d))
    return slice(int(mins[0]), int(maxs[0])), slice(int(mins[1]), int(maxs[1]))


def small_label_rows(metrics: Dict[str, object], label_names: Dict[int, str]) -> List[Dict[str, object]]:
    table = metrics.get("small_oar_per_label", {}).get("per_label", {})
    rows = []
    for label_str, values in table.items():
        label = int(label_str)
        rows.append(
            {
                "label": label,
                "name": label_names.get(label, f"label_{label}"),
                "before": float(values.get("before", 0.0)),
                "stage2": float(values.get("stage2", 0.0)),
                "stage3": float(values.get("final", 0.0)),
                "delta": float(values.get("delta_final_vs_stage2", 0.0)),
            }
        )
    rows.sort(key=lambda row: row["delta"], reverse=True)
    return rows


def write_label_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["label", "name", "before", "stage2", "stage3", "delta"])
        writer.writeheader()
        writer.writerows(rows)


def plot_global_small_overlay(
    path: Path,
    fixed_img: np.ndarray,
    moving_small: np.ndarray,
    fixed_small: np.ndarray,
    stage2_small: np.ndarray,
    stage3_small: np.ndarray,
    dpi: int,
) -> None:
    focus = fixed_small | stage2_small | stage3_small
    indices = center_slices(focus, fixed_img.shape)
    axis_names = ["Sagittal", "Coronal", "Axial"]
    columns = [
        ("Moving vs fixed", [("fixed", fixed_small, "lime"), ("moving", moving_small, "red")]),
        ("Stage2 vs fixed", [("fixed", fixed_small, "lime"), ("stage2", stage2_small, "cyan")]),
        ("Stage3 vs fixed", [("fixed", fixed_small, "lime"), ("stage3", stage3_small, "yellow")]),
        (
            "All contours",
            [
                ("fixed", fixed_small, "lime"),
                ("moving", moving_small, "red"),
                ("stage2", stage2_small, "cyan"),
                ("stage3", stage3_small, "yellow"),
            ],
        ),
    ]
    fig, axs = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
    for row_idx, (axis, index) in enumerate(enumerate(indices)):
        base = slice2d(fixed_img, axis, index)
        for col_idx, (title, masks) in enumerate(columns):
            ax = axs[row_idx, col_idx]
            show_image(ax, base, f"{axis_names[axis]} {title}")
            for _, mask, color in masks:
                contour(ax, slice2d(mask, axis, index), color)
    fig.suptitle("Small-OAR contours on fixed CT: fixed=green, moving=red, Stage2=cyan, Stage3=yellow", fontsize=12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_ct_difference(
    path: Path,
    fixed_img: np.ndarray,
    stage2_img: np.ndarray,
    stage3_img: np.ndarray,
    focus: np.ndarray,
    dpi: int,
) -> None:
    indices = center_slices(focus, fixed_img.shape)
    axis_names = ["Sagittal", "Coronal", "Axial"]
    fig, axs = plt.subplots(3, 5, figsize=(15, 9), constrained_layout=True)
    for row_idx, (axis, index) in enumerate(enumerate(indices)):
        fixed = slice2d(fixed_img, axis, index)
        stage2 = slice2d(stage2_img, axis, index)
        stage3 = slice2d(stage3_img, axis, index)
        panels = [
            (fixed, "Fixed CT", "gray", 1.0),
            (stage2, "Stage2 warped CT", "gray", 1.0),
            (stage3, "Stage3 warped CT", "gray", 1.0),
            (np.abs(stage2 - fixed), "|Stage2-fixed|", "magma", 0.45),
            (np.abs(stage3 - fixed), "|Stage3-fixed|", "magma", 0.45),
        ]
        for col_idx, (image, title, cmap, vmax) in enumerate(panels):
            show_image(axs[row_idx, col_idx], image, f"{axis_names[axis]} {title}", cmap=cmap, vmax=vmax)
    fig.suptitle("Local image agreement around small-OAR ROI", fontsize=12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def select_rows(rows: List[Dict[str, object]], mode: str, top_k: int) -> List[Dict[str, object]]:
    if mode == "gains":
        return sorted(rows, key=lambda row: float(row["delta"]), reverse=True)[:top_k]
    return sorted(rows, key=lambda row: float(row["delta"]))[:top_k]


def plot_label_montage(
    path: Path,
    fixed_img: np.ndarray,
    moving_seg: np.ndarray,
    fixed_seg: np.ndarray,
    stage2_seg: np.ndarray,
    stage3_seg: np.ndarray,
    rows: List[Dict[str, object]],
    title: str,
    margin: int,
    dpi: int,
) -> None:
    if not rows:
        return
    fig, axs = plt.subplots(len(rows), 4, figsize=(13, max(2.5, 2.4 * len(rows))), constrained_layout=True)
    if len(rows) == 1:
        axs = axs[None, :]
    column_specs = [
        ("moving", moving_seg, "red"),
        ("Stage2", stage2_seg, "cyan"),
        ("Stage3", stage3_seg, "yellow"),
        ("Stage2/Stage3", None, "white"),
    ]
    for row_idx, row in enumerate(rows):
        label = int(row["label"])
        masks3d = [
            moving_seg == label,
            fixed_seg == label,
            stage2_seg == label,
            stage3_seg == label,
        ]
        focus = np.logical_or.reduce(masks3d)
        coords = np.argwhere(focus)
        z_index = int(np.rint(coords[:, 2].mean())) if coords.size else fixed_img.shape[2] // 2
        focus2d = focus[:, :, z_index]
        sx, sy = crop_bounds(focus2d, focus2d.shape, margin=margin)
        base = fixed_img[sx, sy, z_index].T
        fixed2d = (fixed_seg[sx, sy, z_index] == label).T
        for col_idx, (col_title, seg, color) in enumerate(column_specs):
            ax = axs[row_idx, col_idx]
            show_image(ax, base, f"{row['name']} {col_title}\n{row['stage2']:.3f}->{row['stage3']:.3f} ({row['delta']:+.3f})")
            contour(ax, fixed2d, "lime", linewidth=0.8)
            if seg is None:
                contour(ax, (stage2_seg[sx, sy, z_index] == label).T, "cyan", linewidth=0.9)
                contour(ax, (stage3_seg[sx, sy, z_index] == label).T, "yellow", linewidth=0.9)
            else:
                contour(ax, (seg[sx, sy, z_index] == label).T, color, linewidth=0.9)
    fig.suptitle(f"{title}: fixed=green, moving=red, Stage2=cyan, Stage3=yellow", fontsize=12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pair_dir = Path(args.pair_dir)
    data_root = Path(args.data_root)
    prefix = args.prefix or f"{args.moving_id}_to_{args.fixed_id}"
    output_dir = Path(args.output_dir) if args.output_dir else pair_dir / "viz_musa_plus"
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_img = load_npy(data_root / "images" / f"{args.fixed_id}.npy")
    moving_seg_o = load_npy(data_root / "seg_o" / f"{args.moving_id}.npy")
    fixed_seg_o = load_npy(data_root / "seg_o" / f"{args.fixed_id}.npy")
    stage2_img = load_npy(pair_file(pair_dir, prefix, "stage2_deformed_img.npy"))
    stage3_img = load_npy(pair_file(pair_dir, prefix, "musa_plus_deformed_img.npy"))
    stage2_seg_o = load_npy(pair_file(pair_dir, prefix, "stage2_deformed_seg_o.npy"))
    stage3_seg_o = load_npy(pair_file(pair_dir, prefix, "musa_plus_deformed_seg_o.npy"))
    metrics = load_json(pair_file(pair_dir, prefix, "musa_plus_metrics.json"))

    label_names = load_label_names(data_root, args.moving_id, args.fixed_id)
    small_labels = resolve_small_labels(metrics, label_names, args.small_oar_labels, args.small_oar_names)
    if not small_labels:
        raise ValueError("Could not resolve small-OAR labels from metrics, metadata, or --small-oar-labels.")

    moving_small = label_mask(moving_seg_o, small_labels)
    fixed_small = label_mask(fixed_seg_o, small_labels)
    stage2_small = label_mask(stage2_seg_o, small_labels)
    stage3_small = label_mask(stage3_seg_o, small_labels)
    focus = fixed_small | stage2_small | stage3_small
    rows = small_label_rows(metrics, label_names)

    plot_global_small_overlay(
        output_dir / "01_small_oar_stage2_vs_stage3_overlay.png",
        fixed_img,
        moving_small,
        fixed_small,
        stage2_small,
        stage3_small,
        args.dpi,
    )
    plot_ct_difference(output_dir / "02_small_oar_ct_difference.png", fixed_img, stage2_img, stage3_img, focus, args.dpi)
    plot_label_montage(
        output_dir / "03_top_small_oar_gains.png",
        fixed_img,
        moving_seg_o,
        fixed_seg_o,
        stage2_seg_o,
        stage3_seg_o,
        select_rows(rows, "gains", args.top_k),
        "Top small-OAR gains",
        args.margin,
        args.dpi,
    )
    plot_label_montage(
        output_dir / "04_small_oar_worst_deltas.png",
        fixed_img,
        moving_seg_o,
        fixed_seg_o,
        stage2_seg_o,
        stage3_seg_o,
        select_rows(rows, "worst", args.top_k),
        "Worst small-OAR deltas",
        args.margin,
        args.dpi,
    )
    write_label_csv(output_dir / "small_oar_label_dice_stage2_vs_stage3.csv", rows)

    report = {
        "moving_id": args.moving_id,
        "fixed_id": args.fixed_id,
        "small_oar_labels": small_labels,
        "small_oar_mean_stage2": metrics.get("small_oar_per_label", {}).get("mean_stage2"),
        "small_oar_mean_stage3": metrics.get("small_oar_per_label", {}).get("mean_final"),
        "small_oar_mean_delta": metrics.get("small_oar_per_label", {}).get("mean_delta"),
        "figures": [
            "01_small_oar_stage2_vs_stage3_overlay.png",
            "02_small_oar_ct_difference.png",
            "03_top_small_oar_gains.png",
            "04_small_oar_worst_deltas.png",
        ],
    }
    (output_dir / "viz_musa_plus_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[INFO] Wrote MUSA+ visual diagnostics to {output_dir}")
    print(
        "[INFO] Small-OAR mean Dice "
        f"{report['small_oar_mean_stage2']:.4f} -> {report['small_oar_mean_stage3']:.4f} "
        f"({report['small_oar_mean_delta']:+.4f})"
    )


if __name__ == "__main__":
    main()
