#!/usr/bin/env python3

"""Build a simple 4D tensor: model → beta → rank → task → {mean, stderr}.

Access pattern: tensor[model][beta][rank][task]['mean'|'stderr']

- model, task, mean, stderr are strings
- beta (float: 0.0 or 0.05)
- rank (int)
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

from collect_results import ROOT, collect_records

from collections import defaultdict
Tensor = Dict[str, Dict[float, Dict[int, Dict[str, Dict[str, str]]]]]


def _to_str(x: Optional[object]) -> str:
    return "" if x is None else str(x)


def build_tensor(records: Iterable[Mapping[str, object]]) -> Tensor:
    tensor: Tensor = {}
    for rec in records:
        model = (rec.get("model_name") or rec.get("model_key") or "unknown_model")
        model = str(model)
        rank = int(rec.get("lora_r") or 0)
        task = str(rec.get("task") or "unknown_task")

        # # Baseline as beta=0.0
        base_mean = rec.get("baseline_pass_at_1")
        base_stderr = rec.get("baseline_stderr")
        if base_mean is not None:
            tensor.setdefault(model, {}).setdefault(0.0, {}).setdefault(rank, {})[task] = {
                "mean": _to_str(f'{base_mean*100:.2f}'),
                "stderr": _to_str(f'{base_stderr*100:.3f}'),
            }

        # KL run (beta typically 0.05)
        kl_beta = rec.get("kl_beta")
        kl_mean = rec.get("kl_pass_at_1")
        kl_stderr = rec.get("kl_stderr")
        if kl_beta is not None:
            beta = float(kl_beta)
            tensor.setdefault(model, {}).setdefault(beta, {}).setdefault(rank, {})[task] = {
                "mean": _to_str(f'{kl_mean*100:.2f}'),
                "stderr": _to_str(f'{kl_stderr*100:.3f}'),
            }

    return tensor

def write_latex_table(tensor):
    betas = [0.0, 0.05]
    result_table="/home/rottman/simple-grpo/tmp/result_table.txt"
    with open(result_table, 'w') as f:
        f.write("%begin{tabular}{l|l|l|ccc}\n%hline\n")
        f.write("Model & beta & rank & GPQA & GSM8K & MATH-500 \\\ ")
    for model in tensor: 
        for beta in betas:
            with open(result_table, 'a') as f:
                f.write("%hline\n")
                f.write("\n")
                f.write("%multirow{6}{*}{"+model+"} & %multirow{6}{*}{"+_to_str(beta)+"} \\\ ")
            for r in tensor[model][beta]:
                with open(result_table, 'a') as f:
                    f.write(f"& & r={r}")
                for task in tensor[model][beta][r]:
                    cell = tensor[model][beta][r][task]
                    mean = cell['mean']
                    stderr = "%stderr{"+cell['stderr']+"}"
                    with open(result_table, 'a') as f:
                        f.write(f"& {mean} {stderr} ")
                with open(result_table, 'a') as f:
                    f.write('\\\ \n')
    with open(result_table, 'a') as f:
        f.write("%hline \n%end{tabular}")


from collections import defaultdict

def task_avg(tensor):
    agg = defaultdict(list)
    avg_diff = defaultdict(dict)  # nested dict: model -> task -> avg diff
    
    model_aliases = {
        "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct",
        "google/gemma-3-1b-it": "Gemma-3-1b-it",
        "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B-Instruct"
    }

    task_aliases = {
        "lighteval|gpqa:diamond|0": "GPQA",
        "lighteval|gsm8k|0": "GSM8K",
        "lighteval|math_500|0": "MATH-500"
    }
    for model, betas in tensor.items():
        if 0.0 not in betas or 0.05 not in betas:
            continue
        for rank in betas[0.0]:
            if rank not in betas[0.05]:
                continue
            for task in betas[0.0][rank]:
                if task in betas[0.05][rank]:
                    no_kl = float(betas[0.0][rank][task]["mean"])
                    kl    = float(betas[0.05][rank][task]["mean"])
                    agg[(model, task)].append(no_kl - kl)

    # compute averages
    for (model, task), diffs in agg.items():
        model = model_aliases.get(model, model)  # fallback to original if not in dict
        task = task_aliases.get(task, task)
        avg = sum(diffs)/len(diffs)
        avg_diff[model][task] = round(avg, 4)

    return avg_diff



import matplotlib.pyplot as plt
import numpy as np
import os

def plot_avg_diffs(avg_diff,
                   save_path="/home/rottman/simple-grpo/tmp/avg_diffs.png"):
    """
    avg_diff: dict of dicts {model: {task: avg_diff_value}}
    Values are differences in percentage points (Δ accuracy).
    """
    models = list(avg_diff.keys())
    all_tasks = sorted({t for d in avg_diff.values() for t in d.keys()})
    task_labels = [t for t in all_tasks]

    x = np.arange(len(all_tasks))
    bar_width = 0.25

    # Muted but distinct colors
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(4.5, 2.5))

    for i, model in enumerate(models):
        vals = [avg_diff[model].get(task, 0) for task in all_tasks]
        positions = x + i * bar_width
        ax.bar(positions, vals, bar_width,
               label=model,
               color=colors[i % len(colors)],
               edgecolor="black",
               linewidth=0.4)

    # Baseline at 0
    ax.axhline(0, color="black", linewidth=0.8)

    # X-axis ticks
    ax.set_xticks(x + bar_width*(len(models)-1)/2)
    ax.set_xticklabels(task_labels, fontsize=9)

    # Labels
    ax.set_ylabel("Δ Accuracy (pp)", fontsize=9)

    # Legend
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # Minimal style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save PNG
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {save_path}")


def main() -> None:
    records = collect_records(ROOT)
    tensor = build_tensor(records)

    write_latex_table(tensor)
    diffs = task_avg(tensor)

    plot_avg_diffs(diffs)
    differences_table="/home/rottman/simple-grpo/tmp/differences_table.txt"
    with open(differences_table,"w") as f:
        f.write(_to_str(diffs))


if __name__ == "__main__":
    main()
