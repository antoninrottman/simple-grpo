#!/usr/bin/env python3
"""Aggregate evaluation results from results_run* directories into a CSV and plots."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib optional
    plt = None

ROOT = Path(__file__).resolve().parents[1]

BASELINE_EPS = 1e-9
ALLOWED_TASKS = {
    "lighteval|math_500|0",
    "lighteval|gpqa:diamond|0",
    "lighteval|gsm8k|0",
}
TASK_METRICS = {
    "lighteval|math_500|0": {"math_pass@1:1_samples"},
    "lighteval|gpqa:diamond|0": {"gpqa_pass@1:1_samples"},
    "lighteval|gsm8k|0": {"extractive_match"},
}


def find_run_metadata(root: Path) -> Iterable[Path]:
    return root.glob("results_run*/**/run_metadata.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def normalize_beta(raw_beta) -> float:
    """Convert configured beta to a float, defaulting missing values to 0."""

    if raw_beta is None:
        return 0.0
    if isinstance(raw_beta, (int, float)):
        return float(raw_beta)
    try:
        return float(raw_beta)
    except (TypeError, ValueError):
        return 0.0


def is_baseline_beta(beta: float) -> bool:
    return math.isclose(beta, 0.0, abs_tol=BASELINE_EPS)


def pick_preferred_record(records: List[Dict]) -> Optional[Dict]:
    if not records:
        return None
    if len(records) == 1:
        return records[0]

    def sort_key(rec: Dict):
        timestamp = rec.get("timestamp_utc") or ""
        # Ensure deterministic ordering even when timestamps are identical.
        source = rec.get("results_source") or ""
        return (timestamp, source)

    return sorted(records, key=sort_key, reverse=True)[0]


def collect_records(root: Path):
    grouped: Dict[tuple, Dict[float, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for meta_path in find_run_metadata(root):
        meta = load_json(meta_path)
        run_dir = meta_path.parent
        eval_dir = run_dir / "evaluation_results"
        if not eval_dir.exists():
            continue
        json_files = list(eval_dir.glob("**/results_*.json"))
        if not json_files:
            continue

        beta = normalize_beta(meta.get("grpo_beta"))
        sweep_name = meta.get("sweep_name")
        model_key = meta.get("model_key")
        model_name = meta.get("model_name")
        lora_r = meta.get("lora_r")
        lora_alpha = meta.get("lora_alpha")
        evaluation_mode = meta.get("evaluation_mode")
        run_name = meta.get("run_name")
        timestamp_utc = meta.get("timestamp_utc")
        slurm_job_id = meta.get("slurm_job_id")

        for json_path in json_files:
            data = load_json(json_path)
            results = data.get("results", {})
            for task, metrics in results.items():
                if task not in ALLOWED_TASKS:
                    continue
                allowed_metrics = TASK_METRICS.get(task)
                if not allowed_metrics:
                    continue
                for metric_name, value in metrics.items():
                    if metric_name.endswith("_stderr"):
                        continue
                    if metric_name not in allowed_metrics:
                        continue
                    stderr = metrics.get(f"{metric_name}_stderr")
                    config_key = (
                        model_key,
                        model_name,
                        lora_r,
                        lora_alpha,
                        evaluation_mode,
                        task,
                        metric_name,
                    )
                    record = {
                        "sweep_name": sweep_name,
                        "model_key": model_key,
                        "model_name": model_name,
                        "lora_r": lora_r,
                        "lora_alpha": lora_alpha,
                        "evaluation_mode": evaluation_mode,
                        "task": task,
                        "metric": metric_name,
                        "value": value,
                        "stderr": stderr,
                        "run_name": run_name,
                        "timestamp_utc": timestamp_utc,
                        "slurm_job_id": slurm_job_id,
                        "results_source": str(json_path.relative_to(root)),
                        "grpo_beta": beta,
                    }
                    grouped[config_key][beta].append(record)

    rows = []
    for key, beta_records in grouped.items():
        (
            model_key,
            model_name,
            lora_r,
            lora_alpha,
            evaluation_mode,
            task,
            metric_name,
        ) = key

        baseline_candidates = []
        comparison_records = {}
        for beta_value, records in beta_records.items():
            selected = pick_preferred_record(records)
            if selected is None:
                continue
            if is_baseline_beta(beta_value):
                baseline_candidates.append(selected)
            else:
                comparison_records[beta_value] = selected

        baseline = pick_preferred_record(baseline_candidates)

        if comparison_records:
            for beta_value in sorted(comparison_records):
                kl_record = comparison_records[beta_value]
                row = {
                    "model_key": model_key,
                    "model_name": model_name,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "evaluation_mode": evaluation_mode,
                    "task": task,
                    "metric": metric_name,
                    "baseline_sweep_name": baseline.get("sweep_name") if baseline else None,
                    "baseline_run_name": baseline.get("run_name") if baseline else None,
                    "baseline_pass_at_1": baseline.get("value") if baseline else None,
                    "baseline_stderr": baseline.get("stderr") if baseline else None,
                    "baseline_timestamp_utc": baseline.get("timestamp_utc") if baseline else None,
                    "baseline_slurm_job_id": baseline.get("slurm_job_id") if baseline else None,
                    "baseline_results_source": baseline.get("results_source") if baseline else None,
                    "kl_beta": beta_value,
                    "kl_sweep_name": kl_record.get("sweep_name"),
                    "kl_run_name": kl_record.get("run_name"),
                    "kl_pass_at_1": kl_record.get("value"),
                    "kl_stderr": kl_record.get("stderr"),
                    "kl_timestamp_utc": kl_record.get("timestamp_utc"),
                    "kl_slurm_job_id": kl_record.get("slurm_job_id"),
                    "kl_results_source": kl_record.get("results_source"),
                }
                if baseline and baseline.get("value") is not None and kl_record.get("value") is not None:
                    row["kl_minus_baseline"] = kl_record["value"] - baseline["value"]
                else:
                    row["kl_minus_baseline"] = None
                rows.append(row)
        elif baseline:
            row = {
                "model_key": model_key,
                "model_name": model_name,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "evaluation_mode": evaluation_mode,
                "task": task,
                "metric": metric_name,
                "baseline_sweep_name": baseline.get("sweep_name"),
                "baseline_run_name": baseline.get("run_name"),
                "baseline_pass_at_1": baseline.get("value"),
                "baseline_stderr": baseline.get("stderr"),
                "baseline_timestamp_utc": baseline.get("timestamp_utc"),
                "baseline_slurm_job_id": baseline.get("slurm_job_id"),
                "baseline_results_source": baseline.get("results_source"),
                "kl_beta": None,
                "kl_sweep_name": None,
                "kl_run_name": None,
                "kl_pass_at_1": None,
                "kl_stderr": None,
                "kl_timestamp_utc": None,
                "kl_slurm_job_id": None,
                "kl_results_source": None,
                "kl_minus_baseline": None,
            }
            rows.append(row)

    def sort_key(rec: Dict):
        return (
            rec.get("model_key") or "",
            rec.get("model_name") or "",
            rec.get("lora_r") or 0,
            rec.get("lora_alpha") or 0,
            rec.get("evaluation_mode") or "",
            rec.get("task") or "",
            rec.get("metric") or "",
            rec.get("kl_beta") if rec.get("kl_beta") is not None else -1.0,
            rec.get("baseline_run_name") or "",
            rec.get("kl_run_name") or "",
        )

    rows.sort(key=sort_key)
    return rows


def write_csv(records, output_path: Path):
    if not records:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "model_key",
        "model_name",
        "lora_r",
        "lora_alpha",
        "evaluation_mode",
        "task",
        "metric",
        "baseline_sweep_name",
        "baseline_run_name",
        "baseline_pass_at_1",
        "baseline_stderr",
        "baseline_timestamp_utc",
        "baseline_slurm_job_id",
        "baseline_results_source",
        "kl_beta",
        "kl_sweep_name",
        "kl_run_name",
        "kl_pass_at_1",
        "kl_stderr",
        "kl_timestamp_utc",
        "kl_slurm_job_id",
        "kl_results_source",
        "kl_minus_baseline",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def make_plot(records, output_path: Path):
    if not records:
        return
    if plt is None:
        print("matplotlib not available; skipping plot generation")
        return

    comparable = [
        rec
        for rec in records
        if rec.get("baseline_pass_at_1") is not None or rec.get("kl_pass_at_1") is not None
    ]
    if not comparable:
        return

    cols = min(3, len(comparable)) or 1
    rows = math.ceil(len(comparable) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)

    palette = ("#5b8def", "#f2994a")

    for idx, rec in enumerate(comparable):
        ax = axes[idx // cols][idx % cols]
        values = []
        stderrs = []
        labels = []

        base_value = rec.get("baseline_pass_at_1")
        if base_value is not None:
            values.append(base_value)
            stderrs.append(rec.get("baseline_stderr") or 0.0)
            labels.append("β=0")

        kl_value = rec.get("kl_pass_at_1")
        if kl_value is not None:
            beta = rec.get("kl_beta")
            beta_label = f"β={beta}" if beta is not None else "β>0"
            values.append(kl_value)
            stderrs.append(rec.get("kl_stderr") or 0.0)
            labels.append(beta_label)

        colors = [palette[i % len(palette)] for i in range(len(values))]
        ax.bar(range(len(values)), values, yerr=stderrs, capsize=4, color=colors)

        title = f"{rec['model_key']} | {rec['task']}"
        subtitle = rec.get("metric")
        if subtitle:
            title = f"{subtitle}\n{title}"
        ax.set_title(title)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
        ax.set_ylabel("pass@1 (1 sample)")
        ax.grid(axis="y", alpha=0.3)
        delta = rec.get("kl_minus_baseline")
        if delta is not None:
            ax.text(
                0.5,
                0.9,
                f"Δ={delta:+.3f}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="#333333",
            )

    total_axes = rows * cols
    for idx in range(len(comparable), total_axes):
        axes[idx // cols][idx % cols].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Collect evaluation results into CSV and plots.")
    parser.add_argument("--csv", dest="csv_path", default="results_summary.csv", help="Output CSV path")
    parser.add_argument("--plot", dest="plot_path", default="results_summary.png", help="Output plot image path")
    args = parser.parse_args()

    records = collect_records(ROOT)
    csv_path = Path(args.csv_path)
    plot_path = Path(args.plot_path)

    write_csv(records, csv_path)
    make_plot(records, plot_path)

    print(f"Collected {len(records)} metric rows.")
    print(f"CSV written to {csv_path}")
    if plot_path.exists():
        print(f"Plot written to {plot_path}")


if __name__ == "__main__":
    main()
