from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable
import io

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pandas as pd

MODEL_PATTERN = re.compile(r"(Qwen|gemma|Llama)", re.IGNORECASE)
BETA_PATTERN = re.compile(r"_b(-?\d+(?:[._]\d+)?(?:e[-+]?\d+)?)", re.IGNORECASE)
RANK_PATTERN = re.compile(r"__r(\d+)")

MODEL_CANONICAL = {
    "qwen": "Qwen",
    "gemma": "gemma",
    "llama": "Llama",
}

REQUIRED_KEYS = (
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
)


def infer_model(name: str) -> str:
    match = MODEL_PATTERN.search(name)
    if not match:
        return "unknown"
    return MODEL_CANONICAL.get(match.group(1).lower(), match.group(1))


def infer_rank(name: str) -> int | None:
    match = RANK_PATTERN.search(name)
    if match:
        return int(match.group(1))
    return None


def infer_beta(name: str) -> float | None:
    match = BETA_PATTERN.search(name)
    if match:
        try:
            raw = match.group(1).replace("_", ".")
            return float(raw)
        except ValueError:
            return None
    return None


def extract_metrics(log_path: Path) -> dict | None:
    metric_line = None
    for line in log_path.read_text().splitlines():
        if "train_runtime" in line:
            metric_line = line.strip()
    if metric_line is None:
        return None
    try:
        parsed = ast.literal_eval(metric_line)
    except (SyntaxError, ValueError):
        return None
    return {key: parsed.get(key) for key in REQUIRED_KEYS}


def collect_train_metrics(sweep_dirs: Iterable[str | Path]) -> pd.DataFrame:
    records: list[dict] = []
    for sweep_dir in sweep_dirs:
        sweep_path = Path(sweep_dir)
        if not sweep_path.exists():
            continue
        for log_file in sweep_path.glob("*.out"):
            metrics = extract_metrics(log_file)
            if not metrics:
                continue
            record = {
                "sweep": str(sweep_path),
                "file": log_file.name,
                "model": infer_model(log_file.name),
                "rank": infer_rank(log_file.name),
                "beta": infer_beta(log_file.name),
            }
            record.update(metrics)
            records.append(record)
    columns = [
        "sweep",
        "file",
        "model",
        "rank",
        "beta",
        *REQUIRED_KEYS,
    ]
    return pd.DataFrame.from_records(records, columns=columns)


SWEEP_DIRS: list[str] = [
    "logs/llama_r_sweep_results",
    "logs/results_run_20250926-083327",
    "logs/results_run_20250926-133418",
] 



# plotting function

PALETTE = {
    "Qwen": "#1f77b4",
    "gemma": "#ff7f0e",
    "Llama": "#2ca02c",
}

MARKERS = {"Qwen": "o", "gemma": "^", "Llama": "s"}
LINESTYLES = {"Qwen": "-", "gemma": "-.", "Llama": ":"}
LINESTYLES_BETA = {0.0: "--", 0.05: "-"}
MARKERS_BETA = {0.0: "o", 0.05: "o"}


def _configure_style() -> None:
    plt.style.use("seaborn-v0_8")
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.5,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )


def _load_metrics_table(path: str | Path) -> pd.DataFrame:
    raw_text = Path(path).read_text()
    for parser in ("csv", "fwf"):
        buf = io.StringIO(raw_text)
        try:
            if parser == "csv":
                df = pd.read_csv(buf, sep=r"\s+")
            else:
                df = pd.read_fwf(buf)
        except Exception:
            continue
        if df.empty:
            continue
        first_col = df.columns[0]
        if not str(first_col).strip() or str(first_col).startswith("Unnamed"):
            df = df.drop(columns=first_col)
        if {"model", "rank", "beta", "train_runtime"}.issubset(df.columns):
            for col in ("rank", "beta", "train_runtime"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.dropna(subset=["model", "rank", "beta", "train_runtime"])
    raise ValueError("Unable to parse metrics table at tmp/train_metrics.txt")


def plot_runtime():
    table_path = Path("/home/rottman/simple-grpo/tmp/train_metrics.txt")
    df = _load_metrics_table(table_path)

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

    _configure_style()
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f0f0f0")
    ax.set_xscale("log", base=2)

    for model in sorted(diffs["model"].unique(), key=lambda m: (m not in PALETTE, m)):
        series = diffs[diffs["model"] == model].sort_values("rank")
        ax.plot(
            series["rank"],
            series["runtime_pct_delta"],
            marker=MARKERS.get(model, "o"),
            markersize=5,
            linewidth=1.4,
            linestyle=LINESTYLES.get(model, "-"),
            label=model,
            color=PALETTE.get(model),
        )

    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="-", alpha=0.6)
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel(r"Relative runtime change (%)")

    ranks = sorted(diffs["rank"].unique())
    ax.set_xticks(ranks)
    ax.get_xaxis().set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.set_xlim(min(ranks) * 0.9, max(ranks) * 1.1)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}%"))
    ax.margins(x=0.02, y=0.15)
    ax.legend(title="Model", loc="upper left", frameon=True, fancybox=True, framealpha=0.85)

    fig.tight_layout()
    output_base = Path("/home/rottman/simple-grpo/tmp/runtime_plot")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")


def load_gsm8k_summary(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    subset = df[df["task"].str.contains("gsm8k", case=False, na=False)].copy()
    subset.rename(columns={"lora_r": "rank"}, inplace=True)
    subset["model"] = subset["model_name"].apply(infer_model)
    subset["rank"] = pd.to_numeric(subset["rank"], errors="coerce")
    subset.dropna(subset=["rank", "baseline_pass_at_1", "kl_pass_at_1"], inplace=True)
    subset["relative_improvement_pct"] = (
        (subset["kl_pass_at_1"] - subset["baseline_pass_at_1"]) / subset["baseline_pass_at_1"]
    ) * 100.0
    return subset[["model", "rank", "relative_improvement_pct"]]


def plot_runtime_and_gsm8k(train_df: pd.DataFrame, summary_path: str | Path) -> None:
    runtime_df = train_df[train_df["beta"].isin([0.0, 0.05])].copy()
    runtime_df.dropna(subset=["model", "rank", "train_runtime"], inplace=True)
    runtime_df["rank"] = runtime_df["rank"].astype(int)
    runtime_df["runtime_minutes"] = runtime_df["train_runtime"] / 60.0
    runtime_df["beta_label"] = runtime_df["beta"].map({0.0: r"$\beta=0.0$", 0.05: r"$\beta=0.05$"})

    gsm_df = load_gsm8k_summary(summary_path)

    # compute relative runtime change (beta=0.0 vs 0.05)
    runtime_pivot = runtime_df.pivot_table(
        index=["model", "rank"],
        columns="beta",
        values="runtime_minutes",
        aggfunc="first",
    ).dropna(subset=[0.0, 0.05])
    runtime_pivot["runtime_pct_delta"] = (runtime_pivot[0.0] - runtime_pivot[0.05]) / runtime_pivot[0.05] * 100.0
    runtime_rel = runtime_pivot.reset_index()[["model", "rank", "runtime_pct_delta"]]

    gsm_df = load_gsm8k_summary(summary_path)

    _configure_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    all_ranks = sorted(set(runtime_rel["rank"]).union(set(gsm_df["rank"].astype(int))))
    x = range(len(all_ranks))
    rank_index = {rank: idx for idx, rank in enumerate(all_ranks)}

    # bar chart for GSM8K improvements
    bar_width = 0.15
    models_bar = sorted(gsm_df["model"].unique(), key=lambda m: (m not in PALETTE, m))
    offsets = {model: (i - (len(models_bar)-1)/2) * bar_width for i, model in enumerate(models_bar)}
    for model in models_bar:
        subset_bar = gsm_df[gsm_df["model"] == model]
        ax.bar(
            [rank_index[r] + offsets[model] for r in subset_bar["rank"]],
            subset_bar["relative_improvement_pct"],
            width=bar_width,
            color=PALETTE.get(model),
            alpha=0.35,
            label=f"{model} GSM8K" if model == models_bar[0] else None,
        )

    # line plot for runtime relative change
    for model in sorted(runtime_rel["model"].unique(), key=lambda m: (m not in PALETTE, m)):
        subset_line = runtime_rel[runtime_rel["model"] == model].sort_values("rank")
        ax.plot(
            [rank_index[r] for r in subset_line["rank"]],
            subset_line["runtime_pct_delta"],
            color=PALETTE.get(model),
            linestyle=LINESTYLES.get(model, "-"),
            marker=MARKERS.get(model, "o"),
            markersize=5,
            linewidth=1.4,
            label=f"{model} runtime",
        )

    ax.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Relative change (%)")
    ax.set_xlabel("LoRA rank")
    ax.set_xticks(list(range(len(all_ranks))))
    ax.set_xticklabels([str(r) for r in all_ranks])
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}%"))
    ax.margins(x=0.05)
    ax.legend(loc="upper left", frameon=True, fancybox=True, framealpha=0.85, fontsize=7)

    fig.tight_layout()
    out_base = Path("/home/rottman/simple-grpo/tmp/runtime_gsm8k")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")

if __name__ == "__main__":
    df = collect_train_metrics(SWEEP_DIRS)
    with open("/home/rottman/simple-grpo/tmp/train_metrics.txt", 'w') as f:
        f.write(df.to_string())
    plot_runtime()
    plot_runtime_and_gsm8k(df, "/home/rottman/simple-grpo/tmp/results_summary.csv")
