"""Aggregate per-label Stage-2 vs MUSA+ Stage-3 Dice from eval metrics."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


GROUPS = (
    ("small_oar", "small_oar_per_label"),
    ("large_oar", "large_oar_per_label"),
    ("bone", "bone_per_label"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-label Dice comparison tables from eval_musa_plus_prepared_pairs.py outputs."
    )
    parser.add_argument("--eval-dir", required=True, help="Directory produced by eval_musa_plus_prepared_pairs.py.")
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Optional explicit musa_plus_pair_metrics.json path. Defaults to <eval-dir>/musa_plus_pair_metrics.json.",
    )
    parser.add_argument("--data-root", default="data", help="Prepared data root with metadata label maps.")
    parser.add_argument("--output-dir", required=True, help="Output directory for per-label CSV/JSON files.")
    return parser.parse_args()


def load_metrics(eval_dir: Path, metrics_json: Optional[Path]) -> List[Dict[str, object]]:
    path = metrics_json or (eval_dir / "musa_plus_pair_metrics.json")
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("pairs"), list):
            return payload["pairs"]
        raise ValueError(f"{path} must contain a list of pair metrics or a dict with a 'pairs' list")

    pair_dir = eval_dir / "pairs"
    files = sorted(pair_dir.glob("*_musa_plus_metrics.json"))
    if not files:
        raise FileNotFoundError(
            f"Could not find {path} or per-pair JSON files under {pair_dir}. "
            "Run eval_musa_plus_prepared_pairs.py with --save-pair-metrics."
        )
    return [json.loads(file_path.read_text(encoding="utf-8")) for file_path in files]


def load_label_names(data_root: Path) -> Dict[int, str]:
    names: Dict[int, str] = {}
    metadata_dir = data_root / "metadata"
    if not metadata_dir.is_dir():
        return names
    for path in sorted(metadata_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for name, label in payload.get("label_map", {}).items():
            names.setdefault(int(label), str(name))
    return names


def mean_std(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def flatten_rows(metrics_rows: Sequence[Dict[str, object]], label_names: Dict[int, str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for metrics in metrics_rows:
        moving_id = str(metrics["moving_id"])
        fixed_id = str(metrics["fixed_id"])
        pair = f"{moving_id}_to_{fixed_id}"
        difficulty = float(metrics.get("stage3", {}).get("difficulty", 0.0))
        for group_name, metrics_key in GROUPS:
            table = metrics.get(metrics_key, {})
            per_label = table.get("per_label", {}) if isinstance(table, dict) else {}
            for label_str, values in per_label.items():
                label = int(label_str)
                stage2 = float(values.get("stage2", 0.0))
                stage3 = float(values.get("final", 0.0))
                rows.append(
                    {
                        "moving_id": moving_id,
                        "fixed_id": fixed_id,
                        "pair": pair,
                        "difficulty": difficulty,
                        "group": group_name,
                        "label": label,
                        "name": label_names.get(label, f"label_{label}"),
                        "before": float(values.get("before", 0.0)),
                        "stage2": stage2,
                        "stage3": stage3,
                        "delta": float(values.get("delta_final_vs_stage2", stage3 - stage2)),
                    }
                )
    return rows


def summarize_per_label(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    summaries = []
    keys = sorted({(str(row["group"]), int(row["label"]), str(row["name"])) for row in rows})
    for group, label, name in keys:
        selected = [row for row in rows if row["group"] == group and int(row["label"]) == label]
        before = mean_std(float(row["before"]) for row in selected)
        stage2 = mean_std(float(row["stage2"]) for row in selected)
        stage3 = mean_std(float(row["stage3"]) for row in selected)
        delta_values = np.asarray([float(row["delta"]) for row in selected], dtype=np.float64)
        delta = mean_std(delta_values)
        summaries.append(
            {
                "group": group,
                "label": label,
                "name": name,
                "num_pairs": len(selected),
                "before_mean": before["mean"],
                "stage2_mean": stage2["mean"],
                "stage2_std": stage2["std"],
                "stage3_mean": stage3["mean"],
                "stage3_std": stage3["std"],
                "delta_mean": delta["mean"],
                "delta_std": delta["std"],
                "delta_median": delta["median"],
                "delta_min": delta["min"],
                "delta_max": delta["max"],
                "stage3_wins": int(np.sum(delta_values > 0)),
                "stage2_wins": int(np.sum(delta_values < 0)),
                "drops_gt_0_02": int(np.sum(delta_values < -0.02)),
                "drops_gt_0_05": int(np.sum(delta_values < -0.05)),
            }
        )
    summaries.sort(key=lambda row: (str(row["group"]), -float(row["delta_mean"]), int(row["label"])))
    return summaries


def summarize_groups(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    summaries = []
    for group, _ in GROUPS:
        selected = [row for row in rows if row["group"] == group]
        if not selected:
            continue
        stage2 = mean_std(float(row["stage2"]) for row in selected)
        stage3 = mean_std(float(row["stage3"]) for row in selected)
        delta_values = np.asarray([float(row["delta"]) for row in selected], dtype=np.float64)
        delta = mean_std(delta_values)
        summaries.append(
            {
                "group": group,
                "num_label_pair_observations": len(selected),
                "num_labels": len({int(row["label"]) for row in selected}),
                "stage2_mean": stage2["mean"],
                "stage2_std": stage2["std"],
                "stage3_mean": stage3["mean"],
                "stage3_std": stage3["std"],
                "delta_mean": delta["mean"],
                "delta_std": delta["std"],
                "delta_median": delta["median"],
                "delta_min": delta["min"],
                "delta_max": delta["max"],
                "stage3_wins": int(np.sum(delta_values > 0)),
                "stage2_wins": int(np.sum(delta_values < 0)),
                "drops_gt_0_02": int(np.sum(delta_values < -0.02)),
                "drops_gt_0_05": int(np.sum(delta_values < -0.05)),
            }
        )
    return summaries


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(eval_dir, Path(args.metrics_json) if args.metrics_json else None)
    label_names = load_label_names(Path(args.data_root))
    rows = flatten_rows(metrics, label_names)
    per_label = summarize_per_label(rows)
    group_summary = summarize_groups(rows)

    write_csv(output_dir / "per_label_by_pair.csv", rows)
    write_csv(output_dir / "per_label_summary.csv", per_label)
    write_csv(output_dir / "per_label_group_summary.csv", group_summary)
    (output_dir / "per_label_summary.json").write_text(
        json.dumps(
            {
                "source_eval_dir": str(eval_dir),
                "num_pairs": len(metrics),
                "group_summary": group_summary,
                "per_label_summary": per_label,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[INFO] Aggregated {len(rows)} label-pair observations from {len(metrics)} pairs")
    for group in group_summary:
        print(
            f"[INFO] {group['group']}: "
            f"{group['stage2_mean']:.4f}->{group['stage3_mean']:.4f} "
            f"({group['delta_mean']:+.4f}), "
            f"wins={group['stage3_wins']}/{group['num_label_pair_observations']}, "
            f"drops>0.05={group['drops_gt_0_05']}"
        )
    print(f"[INFO] Wrote per-label outputs to {output_dir}")


if __name__ == "__main__":
    main()
