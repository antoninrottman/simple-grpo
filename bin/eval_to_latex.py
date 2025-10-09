#!/usr/bin/env python3

"""Build a simple 4D tensor: model → beta → rank → task → {mean, stderr}.

Access pattern: tensor[model][beta][rank][task]['mean'|'stderr']

- model, task, mean, stderr are strings
- beta (float: 0.0 or 0.05)
- rank (int)
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

from collect_evals import ROOT, collect_records
from collect_train_metrics import collect_train_metrics, SWEEP_DIRS

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
                "stderr": _to_str(f'{base_stderr*100:.2f}'),
            }

        # KL run (beta typically 0.05)
        kl_beta = rec.get("kl_beta")
        kl_mean = rec.get("kl_pass_at_1")
        kl_stderr = rec.get("kl_stderr")
        if kl_beta is not None:
            beta = float(kl_beta)
            tensor.setdefault(model, {}).setdefault(beta, {}).setdefault(rank, {})[task] = {
                "mean": _to_str(f'{kl_mean*100:.2f}'),
                "stderr": _to_str(f'{kl_stderr*100:.2f}'),
            }

    return tensor

_MODEL_NAME_CANONICAL = (
    ("meta-llama", "Llama"),
    ("llama", "Llama"),
    ("qwen", "Qwen"),
    ("gemma", "gemma"),
)


def _canonicalize_model_name(model_name: str) -> str:
    lowered = model_name.lower()
    for marker, canonical in _MODEL_NAME_CANONICAL:
        if marker in lowered:
            return canonical
    return model_name


def preprocess_df_train_metrics(df, model_name):
    subset = df[df["beta"].isin([0.0, 0.05])].copy()
    subset["rank"] = subset["rank"].astype(int)
    pivot = subset.pivot_table(
        index=["model", "rank"],
        columns="beta",
        values="train_runtime",
        aggfunc="first",
    ).dropna(subset=[0.0, 0.05])

    pivot["runtime_pct_delta"] = (pivot[0.0] - pivot[0.05]) / pivot[0.05] * 100.0
    diffs = pivot["runtime_pct_delta"].reset_index()

    target_model = _canonicalize_model_name(str(model_name))
    series = (
        diffs[diffs["model"] == target_model]
        .sort_values("rank")
        .drop_duplicates("rank", keep="first")
    )

    if series.empty:
        return {}  # no matching runtime data for this model

    return series.set_index("rank")["runtime_pct_delta"].to_dict()

def write_latex_table(tensor,df):
    betas = [0.0, 0.05]
    result_table="/home/rottman/simple-grpo/tmp/result_table.txt"

    for model in tensor:
        series_by_rank = preprocess_df_train_metrics(df, model)

        with open(result_table, 'a') as f:
            f.write("%begin{table}[t]\n%centering\n%scriptsize")
            f.write("%begin{tabular}{clcccc}\n%hline\n")
            f.write("$%beta$ & rank & GPQA & GSM8K & MATH-500 & Speedup \\\ \hline \hline\n\n")
        for beta in betas:
            with open(result_table, 'a') as f:
                f.write("%hline\n")
                f.write("\n")
                f.write("%multirow{6}{*}{"+_to_str(beta)+"} \n")

            # write the performance for each task, for all ranks
            for r in tensor[model][beta]:
                with open(result_table, 'a') as f:
                    f.write(f"& r={r}")
                for task in tensor[model][beta][r]:
                    cell = tensor[model][beta][r][task]
                    mean = cell['mean']
                    stderr = "%stderr{"+cell['stderr']+"}"
                    with open(result_table, 'a') as f:
                        f.write(f"& {mean} {stderr} ")
                
                
                if beta == 0.0:
                    runtime_delta = series_by_rank.get(r)
                    runtime_str = "-" if runtime_delta is None else f"{runtime_delta:.2f}"
                    with open(result_table, 'a') as f:
                        f.write(f'& {runtime_str}\\\ \n')
                else:
                    with open(result_table, 'a') as f:
                        f.write(f'& - \\\ \n')

        with open(result_table, 'a') as f:
            f.write("%hline \n%end{tabular}\n%caption{"+model+"}\n%label{tab:"+model+"}\n%end{table}\n\n\n")


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

    df = collect_train_metrics(SWEEP_DIRS)

    write_latex_table(tensor,df)
    diffs = task_avg(tensor)

    plot_avg_diffs(diffs)
    differences_table="/home/rottman/simple-grpo/tmp/avg_diffs.txt"
    with open(differences_table,"w") as f:
        f.write(_to_str(diffs))



if __name__ == "__main__":
    main()
