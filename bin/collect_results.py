#!/usr/bin/env python3
"""Aggregate evaluation results from results_run* directories into a CSV and plots."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib optional
    plt = None

ROOT = Path(__file__).resolve().parents[1]


def find_run_metadata(root: Path) -> Iterable[Path]:
    return root.glob("results_run*/**/run_metadata.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def collect_records(root: Path):
    records = []
    for meta_path in find_run_metadata(root):
        meta = load_json(meta_path)
        run_dir = meta_path.parent
        eval_dir = run_dir / "evaluation_results"
        if not eval_dir.exists():
            continue
        json_files = list(eval_dir.glob("**/results_*.json"))
        if not json_files:
            continue
        base = {
            "sweep_name": meta.get("sweep_name"),
            "model_key": meta.get("model_key"),
            "model_name": meta.get("model_name"),
            "run_name": meta.get("run_name"),
            "grpo_beta": meta.get("grpo_beta"),
            "lora_r": meta.get("lora_r"),
            "lora_alpha": meta.get("lora_alpha"),
            "evaluation_mode": meta.get("evaluation_mode"),
            "timestamp_utc": meta.get("timestamp_utc"),
            "slurm_job_id": meta.get("slurm_job_id"),
        }
        for json_path in json_files:
            data = load_json(json_path)
            results = data.get("results", {})
            for task, metrics in results.items():
                for metric_name, value in metrics.items():
                    if metric_name.endswith("_stderr"):
                        continue
                    stderr = metrics.get(f"{metric_name}_stderr")
                    record = {
                        **base,
                        "task": task,
                        "metric": metric_name,
                        "value": value,
                        "stderr": stderr,
                        "results_source": str(json_path.relative_to(root)),
                    }
                    records.append(record)
    return records


def write_csv(records, output_path: Path):
    if not records:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "sweep_name",
        "model_key",
        "model_name",
        "run_name",
        "grpo_beta",
        "lora_r",
        "lora_alpha",
        "evaluation_mode",
        "timestamp_utc",
        "slurm_job_id",
        "task",
        "metric",
        "value",
        "stderr",
        "results_source",
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

    metric_groups = defaultdict(list)
    for rec in records:
        key = f"{rec['task']}::{rec['metric']}"
        metric_groups[key].append(rec)

    metrics = sorted(metric_groups)
    if not metrics:
        return

    cols = min(3, len(metrics)) or 1
    rows = math.ceil(len(metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)

    for idx, metric_key in enumerate(metrics):
        ax = axes[idx // cols][idx % cols]
        group = metric_groups[metric_key]
        group.sort(key=lambda r: (r.get("model_key"), r.get("run_name")))
        labels = [f"{r['model_key']}\n{r['run_name']}" for r in group]
        values = [r["value"] for r in group]
        stderrs = [r.get("stderr") or 0.0 for r in group]
        ax.bar(range(len(group)), values, yerr=stderrs, capsize=4, color="#5b8def")
        task, metric = metric_key.split("::", 1)
        ax.set_title(f"{metric}\n[{task}]")
        ax.set_xticks(range(len(group)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Value")
        ax.grid(axis="y", alpha=0.3)

    total_axes = rows * cols
    for idx in range(len(metrics), total_axes):
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
