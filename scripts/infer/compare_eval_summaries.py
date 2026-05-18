"""Compare two DIR-MUSA evaluation summary CSV files."""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_COLUMNS = [
    "seg_o_before",
    "seg_o_after",
    "seg_o_delta",
    "seg_b_before",
    "seg_b_after",
    "seg_b_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two eval_prepared_pairs.py summary.csv files.")
    parser.add_argument("--a-name", default="M01", help="Display name for model A.")
    parser.add_argument("--a-summary", required=True, help="Path to model A summary.csv.")
    parser.add_argument("--b-name", default="M05", help="Display name for model B.")
    parser.add_argument("--b-summary", required=True, help="Path to model B summary.csv.")
    parser.add_argument("--output-dir", required=True, help="Output directory for comparison files.")
    parser.add_argument("--dpi", type=int, default=160, help="PNG resolution.")
    return parser.parse_args()


def read_summary(path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        missing = {"moving_id", "fixed_id", *METRIC_COLUMNS}.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            key = (row["moving_id"], row["fixed_id"])
            rows[key] = {column: float(row[column]) for column in METRIC_COLUMNS}
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def paired_rows(a_rows: Dict[Tuple[str, str], Dict[str, float]], b_rows: Dict[Tuple[str, str], Dict[str, float]]):
    common = sorted(set(a_rows).intersection(b_rows))
    if not common:
        raise ValueError("The two summaries share no moving/fixed pairs")
    return common


def mean_std(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(array)), "std": float(np.std(array))}


def build_comparison(a_name: str, a_rows, b_name: str, b_rows):
    common = paired_rows(a_rows, b_rows)
    out_rows = []
    for moving_id, fixed_id in common:
        row = {"moving_id": moving_id, "fixed_id": fixed_id}
        for column in METRIC_COLUMNS:
            a_value = a_rows[(moving_id, fixed_id)][column]
            b_value = b_rows[(moving_id, fixed_id)][column]
            row[f"{a_name}_{column}"] = a_value
            row[f"{b_name}_{column}"] = b_value
            row[f"{b_name}_minus_{a_name}_{column}"] = b_value - a_value
        out_rows.append(row)
    return out_rows


def summarize(comparison_rows: List[Dict[str, float]], a_name: str, b_name: str) -> Dict[str, object]:
    summary: Dict[str, object] = {"num_pairs": len(comparison_rows)}
    for column in METRIC_COLUMNS:
        a_values = [row[f"{a_name}_{column}"] for row in comparison_rows]
        b_values = [row[f"{b_name}_{column}"] for row in comparison_rows]
        diff_values = [row[f"{b_name}_minus_{a_name}_{column}"] for row in comparison_rows]
        summary[f"{a_name}_{column}"] = mean_std(a_values)
        summary[f"{b_name}_{column}"] = mean_std(b_values)
        summary[f"{b_name}_minus_{a_name}_{column}"] = mean_std(diff_values)
        summary[f"{b_name}_wins_{column}"] = int(np.sum(np.asarray(diff_values) > 0))
        summary[f"{a_name}_wins_{column}"] = int(np.sum(np.asarray(diff_values) < 0))
    return summary


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_mean_bars(path: Path, summary: Dict[str, object], a_name: str, b_name: str, dpi: int) -> None:
    labels = ["seg_o after", "seg_o gain", "seg_b after", "seg_b gain"]
    columns = ["seg_o_after", "seg_o_delta", "seg_b_after", "seg_b_delta"]
    a_values = [summary[f"{a_name}_{column}"]["mean"] for column in columns]
    b_values = [summary[f"{b_name}_{column}"]["mean"] for column in columns]
    x = np.arange(len(columns))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.bar(x - width / 2, a_values, width, label=a_name, color="#4c78a8")
    ax.bar(x + width / 2, b_values, width, label=b_name, color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(1.0, max(a_values + b_values) * 1.1))
    ax.set_ylabel("Dice")
    ax.set_title("Mean validation Dice comparison")
    ax.legend()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_pair_scatter(path: Path, rows: List[Dict[str, object]], a_name: str, b_name: str, metric: str, dpi: int) -> None:
    a_values = np.asarray([row[f"{a_name}_{metric}"] for row in rows], dtype=np.float64)
    b_values = np.asarray([row[f"{b_name}_{metric}"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.scatter(a_values, b_values, s=26, alpha=0.8)
    low = min(float(np.min(a_values)), float(np.min(b_values)), 0.0)
    high = max(float(np.max(a_values)), float(np.max(b_values)), 1.0)
    ax.plot([low, high], [low, high], "--", color="black", linewidth=1)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(f"{a_name} {metric}")
    ax.set_ylabel(f"{b_name} {metric}")
    ax.set_title(f"Per-pair {metric}: points above line favor {b_name}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    a_rows = read_summary(Path(args.a_summary))
    b_rows = read_summary(Path(args.b_summary))
    rows = build_comparison(args.a_name, a_rows, args.b_name, b_rows)
    summary = summarize(rows, args.a_name, args.b_name)

    write_csv(output_dir / "comparison_by_pair.csv", rows)
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_mean_bars(output_dir / "comparison_mean_bars.png", summary, args.a_name, args.b_name, args.dpi)
    plot_pair_scatter(output_dir / "scatter_seg_o_after.png", rows, args.a_name, args.b_name, "seg_o_after", args.dpi)
    plot_pair_scatter(output_dir / "scatter_seg_b_after.png", rows, args.a_name, args.b_name, "seg_b_after", args.dpi)
    plot_pair_scatter(output_dir / "scatter_seg_o_delta.png", rows, args.a_name, args.b_name, "seg_o_delta", args.dpi)
    plot_pair_scatter(output_dir / "scatter_seg_b_delta.png", rows, args.a_name, args.b_name, "seg_b_delta", args.dpi)

    print(f"[INFO] Compared {summary['num_pairs']} shared pairs")
    for column in ("seg_o_after", "seg_o_delta", "seg_b_after", "seg_b_delta"):
        diff = summary[f"{args.b_name}_minus_{args.a_name}_{column}"]
        print(
            f"[INFO] {args.b_name}-{args.a_name} {column}: "
            f"{diff['mean']:+.6f} +/- {diff['std']:.6f}; "
            f"{args.b_name} wins {summary[f'{args.b_name}_wins_{column}']} pairs, "
            f"{args.a_name} wins {summary[f'{args.a_name}_wins_{column}']} pairs"
        )
    print(f"[INFO] Wrote comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
