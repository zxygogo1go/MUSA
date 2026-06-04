"""Summarize Stage-2 vs MUSA+ Stage-3 performance from MUSA+ eval metrics."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a full Stage-2 vs adaptive Stage-3 comparison from eval_musa_plus_prepared_pairs.py outputs."
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Directory produced by eval_musa_plus_prepared_pairs.py.",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Optional explicit musa_plus_pair_metrics.json path. Defaults to <eval-dir>/musa_plus_pair_metrics.json.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for comparison CSV/JSON/Markdown files.")
    parser.add_argument("--stage2-name", default="M05+MUSA Stage2", help="Display name for the Stage-2 baseline.")
    parser.add_argument(
        "--stage3-name",
        default="M05+MUSA+ adaptive Stage3 safe",
        help="Display name for the Stage-3 method.",
    )
    return parser.parse_args()


def _get(mapping: Dict[str, object], path: Sequence[str], default: Optional[float] = None) -> Optional[float]:
    current: object = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    if current is None:
        return default
    return float(current)


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


def row_from_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    stage3 = metrics["stage3"]
    stage2_jac = stage3.get("stage2_jacobian", {})
    final_jac = stage3.get("final_jacobian", {})
    residual_mag = stage3.get("residual_magnitude", {})
    stage2_mag = stage3.get("stage2_dvf_magnitude", {})
    final_mag = stage3.get("final_dvf_magnitude", {})
    small = metrics["small_oar_per_label"]
    large = metrics["large_oar_per_label"]
    bone = metrics["bone_per_label"]

    all_oar_stage2 = _get(metrics, ("dice_seg_o_stage2", "mean_after"), 0.0)
    all_oar_stage3 = _get(metrics, ("dice_seg_o_musa_plus", "mean_after"), 0.0)
    seg_b_stage2 = _get(metrics, ("dice_seg_b_stage2", "mean_after"), 0.0)
    seg_b_stage3 = _get(metrics, ("dice_seg_b_musa_plus", "mean_after"), 0.0)

    row = {
        "moving_id": metrics["moving_id"],
        "fixed_id": metrics["fixed_id"],
        "pair": f"{metrics['moving_id']}_to_{metrics['fixed_id']}",
        "difficulty": _get(stage3, ("difficulty",), 0.0),
        "roi_radius": _get(stage3, ("roi_radius",), 0.0),
        "residual_scale": _get(stage3, ("residual_scale",), 0.0),
        "all_oar_stage2": all_oar_stage2,
        "all_oar_stage3": all_oar_stage3,
        "all_oar_delta": all_oar_stage3 - all_oar_stage2,
        "small_oar_stage2": float(small["mean_stage2"]),
        "small_oar_stage3": float(small["mean_final"]),
        "small_oar_delta": float(small["mean_delta"]),
        "small_oar_worst_delta": float(small["worst_delta"]),
        "small_oar_drop_gt_0_02": int(small["num_drop_gt_0_02"]),
        "small_oar_drop_gt_0_05": int(small["num_drop_gt_0_05"]),
        "large_oar_stage2": float(large["mean_stage2"]),
        "large_oar_stage3": float(large["mean_final"]),
        "large_oar_delta": float(large["mean_delta"]),
        "large_oar_worst_delta": float(large["worst_delta"]),
        "large_oar_drop_gt_0_02": int(large["num_drop_gt_0_02"]),
        "large_oar_drop_gt_0_05": int(large["num_drop_gt_0_05"]),
        "bone_stage2": float(bone["mean_stage2"]),
        "bone_stage3": float(bone["mean_final"]),
        "bone_delta": float(bone["mean_delta"]),
        "bone_worst_delta": float(bone["worst_delta"]),
        "bone_drop_gt_0_02": int(bone["num_drop_gt_0_02"]),
        "bone_drop_gt_0_05": int(bone["num_drop_gt_0_05"]),
        "seg_b_stage2": seg_b_stage2,
        "seg_b_stage3": seg_b_stage3,
        "seg_b_delta": seg_b_stage3 - seg_b_stage2,
        "stage2_jac_nonpos": _get(stage2_jac, ("global_nonpos_ratio",), 0.0),
        "stage3_jac_nonpos": _get(final_jac, ("global_nonpos_ratio",), 0.0),
        "jac_nonpos_delta": _get(final_jac, ("global_nonpos_ratio",), 0.0)
        - _get(stage2_jac, ("global_nonpos_ratio",), 0.0),
        "stage2_jac_roi_nonpos": _get(stage2_jac, ("roi_nonpos_ratio",), 0.0),
        "stage3_jac_roi_nonpos": _get(final_jac, ("roi_nonpos_ratio",), 0.0),
        "jac_roi_nonpos_delta": _get(final_jac, ("roi_nonpos_ratio",), 0.0)
        - _get(stage2_jac, ("roi_nonpos_ratio",), 0.0),
        "stage2_jac_roi_min": _get(stage2_jac, ("roi_min",), 0.0),
        "stage3_jac_roi_min": _get(final_jac, ("roi_min",), 0.0),
        "stage2_dvf_roi_p95": _get(stage2_mag, ("roi_p95",), 0.0),
        "stage3_dvf_roi_p95": _get(final_mag, ("roi_p95",), 0.0),
        "dvf_roi_p95_delta": _get(final_mag, ("roi_p95",), 0.0) - _get(stage2_mag, ("roi_p95",), 0.0),
        "residual_roi_p95": _get(residual_mag, ("roi_p95",), 0.0),
        "residual_roi_max": _get(residual_mag, ("roi_max",), 0.0),
    }
    return row


def mean_std(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def metric_summary(rows: List[Dict[str, object]], metric: Dict[str, object], num_pairs: int) -> Dict[str, object]:
    stage2_key = metric.get("stage2")
    stage3_key = metric.get("stage3")
    delta_key = metric.get("delta")
    higher_better = bool(metric.get("higher_better", True))

    stage2_stats = mean_std(float(row[stage2_key]) for row in rows) if stage2_key else None
    stage3_stats = mean_std(float(row[stage3_key]) for row in rows) if stage3_key else None
    delta_stats = mean_std(float(row[delta_key]) for row in rows) if delta_key else None

    wins_stage3 = ""
    wins_stage2 = ""
    if stage2_key and stage3_key:
        differences = np.asarray([float(row[stage3_key]) - float(row[stage2_key]) for row in rows], dtype=np.float64)
        if higher_better:
            wins_stage3 = int(np.sum(differences > 0))
            wins_stage2 = int(np.sum(differences < 0))
        else:
            wins_stage3 = int(np.sum(differences < 0))
            wins_stage2 = int(np.sum(differences > 0))
    elif delta_key:
        deltas = np.asarray([float(row[delta_key]) for row in rows], dtype=np.float64)
        wins_stage3 = int(np.sum(deltas > 0)) if higher_better else int(np.sum(deltas < 0))
        wins_stage2 = int(np.sum(deltas < 0)) if higher_better else int(np.sum(deltas > 0))

    return {
        "metric": metric["name"],
        "stage2_mean": "" if stage2_stats is None else stage2_stats["mean"],
        "stage2_std": "" if stage2_stats is None else stage2_stats["std"],
        "stage3_mean": "" if stage3_stats is None else stage3_stats["mean"],
        "stage3_std": "" if stage3_stats is None else stage3_stats["std"],
        "change_mean": "" if delta_stats is None else delta_stats["mean"],
        "change_std": "" if delta_stats is None else delta_stats["std"],
        "stage3_wins": wins_stage3,
        "stage2_wins": wins_stage2,
        "num_pairs": num_pairs,
        "direction": "higher_better" if higher_better else "lower_better",
        "notes": metric.get("notes", ""),
    }


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    numeric_keys = [key for key in rows[0] if key not in ("moving_id", "fixed_id", "pair")]
    return {"num_pairs": len(rows), **{key: mean_std(float(row[key]) for row in rows) for key in numeric_keys}}


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def difficulty_bucket_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Summarize comparison metrics by easy/medium/hard difficulty tertiles."""

    if len(rows) < 3:
        return []
    difficulties = np.asarray([float(row["difficulty"]) for row in rows], dtype=np.float64)
    q_easy, q_hard = np.quantile(difficulties, [1.0 / 3.0, 2.0 / 3.0])
    buckets = [
        ("easy", [row for row in rows if float(row["difficulty"]) <= q_easy]),
        ("medium", [row for row in rows if q_easy < float(row["difficulty"]) <= q_hard]),
        ("hard", [row for row in rows if float(row["difficulty"]) > q_hard]),
    ]
    out_rows = []
    keys = [
        "small_oar_stage2",
        "small_oar_stage3",
        "small_oar_delta",
        "large_oar_stage2",
        "large_oar_stage3",
        "large_oar_delta",
        "large_oar_worst_delta",
        "bone_delta",
        "stage2_jac_roi_nonpos",
        "stage3_jac_roi_nonpos",
        "residual_roi_p95",
    ]
    for name, bucket in buckets:
        if not bucket:
            continue
        bucket_difficulties = [float(row["difficulty"]) for row in bucket]
        entry: Dict[str, object] = {
            "bucket": name,
            "num_pairs": len(bucket),
            "difficulty_min": min(bucket_difficulties),
            "difficulty_max": max(bucket_difficulties),
            "difficulty_mean": float(np.mean(bucket_difficulties)),
        }
        for key in keys:
            entry[f"{key}_mean"] = float(np.mean([float(row[key]) for row in bucket]))
        out_rows.append(entry)
    return out_rows


def fmt(value: object, digits: int = 4) -> str:
    if value == "" or value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def write_markdown(path: Path, summary_rows: List[Dict[str, object]], stage2_name: str, stage3_name: str) -> None:
    lines = [
        f"# {stage2_name} vs {stage3_name}",
        "",
        "| Metric | Stage2 mean | Stage3 mean | Change | Stage3 wins | Direction |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["metric"]),
                    fmt(row["stage2_mean"]),
                    fmt(row["stage3_mean"]),
                    fmt(row["change_mean"]),
                    f"{row['stage3_wins']}/{row['num_pairs']}" if row["stage3_wins"] != "" else "-",
                    str(row["direction"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(eval_dir, Path(args.metrics_json) if args.metrics_json else None)
    rows = [row_from_metrics(item) for item in metrics]
    rows.sort(key=lambda row: (str(row["moving_id"]), str(row["fixed_id"])))

    metric_specs = [
        {"name": "All OAR label Dice", "stage2": "all_oar_stage2", "stage3": "all_oar_stage3", "delta": "all_oar_delta"},
        {
            "name": "Small-OAR label Dice",
            "stage2": "small_oar_stage2",
            "stage3": "small_oar_stage3",
            "delta": "small_oar_delta",
        },
        {
            "name": "Large-OAR label Dice",
            "stage2": "large_oar_stage2",
            "stage3": "large_oar_stage3",
            "delta": "large_oar_delta",
        },
        {"name": "Bone label Dice", "stage2": "bone_stage2", "stage3": "bone_stage3", "delta": "bone_delta"},
        {
            "name": "Seg-b mean Dice",
            "stage2": "seg_b_stage2",
            "stage3": "seg_b_stage3",
            "delta": "seg_b_delta",
        },
        {
            "name": "ROI non-positive Jacobian ratio",
            "stage2": "stage2_jac_roi_nonpos",
            "stage3": "stage3_jac_roi_nonpos",
            "delta": "jac_roi_nonpos_delta",
            "higher_better": False,
        },
        {
            "name": "Global non-positive Jacobian ratio",
            "stage2": "stage2_jac_nonpos",
            "stage3": "stage3_jac_nonpos",
            "delta": "jac_nonpos_delta",
            "higher_better": False,
        },
        {
            "name": "ROI DVF magnitude p95",
            "stage2": "stage2_dvf_roi_p95",
            "stage3": "stage3_dvf_roi_p95",
            "delta": "dvf_roi_p95_delta",
            "higher_better": False,
        },
        {
            "name": "Large-OAR worst label delta",
            "delta": "large_oar_worst_delta",
            "notes": "Stage3 relative to Stage2; closer to 0 is better for preservation.",
        },
        {
            "name": "Bone worst label delta",
            "delta": "bone_worst_delta",
            "notes": "Stage3 relative to Stage2; closer to 0 is better for preservation.",
        },
        {
            "name": "Small-OAR worst label delta",
            "delta": "small_oar_worst_delta",
            "notes": "Stage3 relative to Stage2.",
        },
        {
            "name": "Stage3 residual DVF ROI p95",
            "delta": "residual_roi_p95",
            "higher_better": False,
            "notes": "Magnitude of added Stage3 residual field.",
        },
    ]
    summary_rows = [metric_summary(rows, metric, len(rows)) for metric in metric_specs]
    full_summary = summarize(rows)
    bucket_rows = difficulty_bucket_rows(rows)

    write_csv(output_dir / "stage2_vs_stage3_by_pair.csv", rows)
    write_csv(output_dir / "stage2_vs_stage3_summary.csv", summary_rows)
    if bucket_rows:
        write_csv(output_dir / "stage2_vs_stage3_difficulty_buckets.csv", bucket_rows)
    (output_dir / "stage2_vs_stage3_summary.json").write_text(
        json.dumps(
            {
                "stage2_name": args.stage2_name,
                "stage3_name": args.stage3_name,
                "source_eval_dir": str(eval_dir),
                "num_pairs": len(rows),
                "summary_rows": summary_rows,
                "difficulty_buckets": bucket_rows,
                "all_numeric_summary": full_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(output_dir / "stage2_vs_stage3_report.md", summary_rows, args.stage2_name, args.stage3_name)

    print(f"[INFO] Compared {len(rows)} pairs")
    for row in summary_rows[:8]:
        print(
            f"[INFO] {row['metric']}: "
            f"stage2={fmt(row['stage2_mean'])}, stage3={fmt(row['stage3_mean'])}, "
            f"change={fmt(row['change_mean'])}, wins={row['stage3_wins']}/{row['num_pairs']}"
        )
    print(f"[INFO] Wrote comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
