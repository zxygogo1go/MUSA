"""Plot visual summaries for Stage-2 vs adaptive Stage-3 comparison outputs."""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create figures from compare_musa_plus_stage2_stage3.py outputs.")
    parser.add_argument("--comparison-dir", required=True, help="Directory containing stage2_vs_stage3_*.csv files.")
    parser.add_argument("--output-dir", default=None, help="Figure output directory. Default: <comparison-dir>/figures.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution.")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def label_pair(pair: str) -> str:
    return pair.replace("segrap_", "").replace("_to_", "->")


def plot_main_dice_bars(summary_rows: List[Dict[str, str]], out_path: Path, dpi: int) -> None:
    wanted = ["All OAR label Dice", "Small-OAR label Dice", "Large-OAR label Dice", "Bone label Dice"]
    rows = [next(row for row in summary_rows if row["metric"] == metric) for metric in wanted]
    labels = ["All OAR", "Small OAR", "Large OAR", "Bone"]
    stage2 = np.asarray([as_float(row, "stage2_mean") for row in rows])
    stage3 = np.asarray([as_float(row, "stage3_mean") for row in rows])
    stage2_std = np.asarray([as_float(row, "stage2_std") for row in rows])
    stage3_std = np.asarray([as_float(row, "stage3_std") for row in rows])

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
    ax.bar(x - width / 2, stage2, width, yerr=stage2_std, label="M05+MUSA Stage2", color="#4c78a8", capsize=3)
    ax.bar(
        x + width / 2,
        stage3,
        width,
        yerr=stage3_std,
        label="M05+MUSA+ Stage3 safe",
        color="#f58518",
        capsize=3,
    )
    for idx, change in enumerate(stage3 - stage2):
        ax.text(idx, max(stage2[idx] + stage2_std[idx], stage3[idx] + stage3_std[idx]) + 0.025, f"{change:+.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Dice")
    ax.set_ylim(0.0, 0.9)
    ax.set_title("Stage3 safe improves small-OAR and overall OAR Dice")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_small_oar_slope(pair_rows: List[Dict[str, str]], out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.2), constrained_layout=True)
    x = np.asarray([0, 1])
    for row in pair_rows:
        y = np.asarray([as_float(row, "small_oar_stage2"), as_float(row, "small_oar_stage3")])
        ax.plot(x, y, marker="o", linewidth=2, alpha=0.82)
        ax.text(1.03, y[1], label_pair(row["pair"]), va="center", fontsize=8)
    ax.set_xlim(-0.08, 1.38)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Stage2", "Stage3 safe"])
    ax.set_ylabel("Small-OAR Dice")
    ax.set_title("Per-pair small-OAR Dice improvement")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_safety_tradeoff(pair_rows: List[Dict[str, str]], out_path: Path, dpi: int) -> None:
    small_delta = np.asarray([as_float(row, "small_oar_delta") for row in pair_rows])
    large_worst = np.asarray([as_float(row, "large_oar_worst_delta") for row in pair_rows])
    difficulty = np.asarray([as_float(row, "difficulty") for row in pair_rows])
    jac = np.asarray([as_float(row, "stage3_jac_roi_nonpos") for row in pair_rows])
    sizes = 240 + 1200 * (difficulty - difficulty.min()) / (np.ptp(difficulty) + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 5.4), constrained_layout=True)
    scatter = ax.scatter(small_delta, large_worst, s=sizes, c=jac * 100.0, cmap="viridis", edgecolor="black", linewidth=0.7)
    for row, x_value, y_value in zip(pair_rows, small_delta, large_worst):
        ax.text(x_value + 0.006, y_value, label_pair(row["pair"]), fontsize=8, va="center")
    ax.axhline(-0.02, color="#d62728", linestyle="--", linewidth=1.2, label="large worst drop -0.02")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Small-OAR Dice gain (Stage3 - Stage2)")
    ax.set_ylabel("Large-OAR worst-label Dice change")
    ax.set_title("Gain-preservation trade-off")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("ROI Jac <= 0 (%)")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_difficulty_buckets(bucket_rows: List[Dict[str, str]], out_path: Path, dpi: int) -> None:
    if not bucket_rows:
        return
    labels = [row["bucket"].capitalize() for row in bucket_rows]
    small_delta = np.asarray([as_float(row, "small_oar_delta_mean") for row in bucket_rows])
    large_worst = np.asarray([as_float(row, "large_oar_worst_delta_mean") for row in bucket_rows])
    jac = np.asarray([as_float(row, "stage3_jac_roi_nonpos_mean") for row in bucket_rows])

    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax1.bar(x - 0.18, small_delta, width=0.36, color="#54a24b", label="Small-OAR gain")
    ax1.bar(x + 0.18, large_worst, width=0.36, color="#e45756", label="Large worst change")
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Dice change")
    ax1.grid(axis="y", alpha=0.24)
    ax2 = ax1.twinx()
    ax2.plot(x, jac * 100.0, marker="o", color="#4c78a8", linewidth=2, label="ROI Jac <= 0")
    ax2.set_ylabel("ROI Jac <= 0 (%)")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")
    ax1.set_title("Difficulty buckets: harder pairs gain more small-OAR Dice")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_pair_heatmap(pair_rows: List[Dict[str, str]], out_path: Path, dpi: int) -> None:
    columns = [
        ("all_oar_delta", "All OAR\nDelta"),
        ("small_oar_delta", "Small\nDelta"),
        ("large_oar_delta", "Large\nDelta"),
        ("large_oar_worst_delta", "Large Worst\nDelta"),
        ("bone_delta", "Bone\nDelta"),
        ("stage3_jac_roi_nonpos", "ROI Jac<=0"),
    ]
    matrix = np.asarray([[as_float(row, key) for key, _ in columns] for row in pair_rows], dtype=np.float64)
    display = matrix.copy()
    display[:, -1] *= 10.0
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    im = ax.imshow(display, cmap="coolwarm", aspect="auto", vmin=-0.08, vmax=0.36)
    ax.set_yticks(np.arange(len(pair_rows)))
    ax.set_yticklabels([label_pair(row["pair"]) for row in pair_rows])
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([label for _, label in columns])
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text = f"{value:.3f}" if col_idx != matrix.shape[1] - 1 else f"{value * 100:.2f}%"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8)
    ax.set_title("Per-pair Stage3 changes and topology")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    comparison_dir = Path(args.comparison_dir)
    output_dir = Path(args.output_dir) if args.output_dir else comparison_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(comparison_dir / "stage2_vs_stage3_summary.csv")
    pair_rows = read_csv(comparison_dir / "stage2_vs_stage3_by_pair.csv")
    bucket_path = comparison_dir / "stage2_vs_stage3_difficulty_buckets.csv"
    bucket_rows = read_csv(bucket_path) if bucket_path.exists() else []

    plot_main_dice_bars(summary_rows, output_dir / "01_main_dice_bars.png", args.dpi)
    plot_small_oar_slope(pair_rows, output_dir / "02_small_oar_per_pair_slope.png", args.dpi)
    plot_safety_tradeoff(pair_rows, output_dir / "03_gain_preservation_tradeoff.png", args.dpi)
    plot_difficulty_buckets(bucket_rows, output_dir / "04_difficulty_bucket_summary.png", args.dpi)
    plot_pair_heatmap(pair_rows, output_dir / "05_pair_metric_heatmap.png", args.dpi)

    print(f"[INFO] Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
