#!/usr/bin/env python3
"""Minimal, single-file KL-vs-step plotter with simple knobs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration – tweak these constants and rerun the script.
# ---------------------------------------------------------------------------
OUTPUT_PATH = Path("tmp/kl_vs_rank_beta005.png")

SWEEPS: Mapping[str, Union[str, Path, Tuple[str, Union[str, Path]]]] = {
    "gemma": "outputs/results_run_20250926-133418",
    "qwen": "outputs/results_run_20250926-083327",
    "llama": "outputs/llama_r_sweep_results",
}

BETAS: Sequence[float] | None = [0.05]  # Set to None to plot every beta.
LOG_SCALE = True
SHOW_POINTS = False
DOWNSAMPLE_EVERY = 10  # Use 1 for no downsampling.
CHECKPOINT = 300  # Which checkpoint directory to read trainer_state.json from.


# ---------------------------------------------------------------------------
# Lightweight data helpers (trimmed down from the original plotter).
# ---------------------------------------------------------------------------
SweepSpec = Union[str, Path, Tuple[str, Union[str, Path]]]


def _load_json(path: Path) -> Dict[str, object]:
    with path.open() as fh:
        return json.load(fh)


def _log_history(trainer_state: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    history = trainer_state.get("log_history") or []
    return history if isinstance(history, Sequence) else []


def _sanitize_metric_name(key: str) -> str:
    return key.replace("/", "_").replace(":", "_")


def _normalise(value: object) -> object:
    return np.nan if value is None else value


def _iter_sweeps(sweeps: Mapping[str, SweepSpec] | Sequence[SweepSpec]) -> Iterable[Tuple[str, Path]]:
    if isinstance(sweeps, Mapping):
        for alias, spec in sweeps.items():
            yield alias, Path(spec)
    else:
        for spec in sweeps:
            if isinstance(spec, tuple) and len(spec) == 2:
                alias, path = spec
                yield str(alias), Path(path)
            else:
                path = Path(spec)
                yield path.name, path


def load_training_logs(sweeps: Mapping[str, SweepSpec] | Sequence[SweepSpec]) -> pd.DataFrame:
    target = f"checkpoint-{CHECKPOINT}/trainer_state.json"
    rows: list[dict[str, object]] = []

    for alias, root in _iter_sweeps(sweeps):
        for trainer_state_path in root.rglob(target):
            run_dir = trainer_state_path.parent.parent.parent
            run_metadata_path = run_dir / "run_metadata.json"
            if not run_metadata_path.exists():
                continue

            meta = _load_json(run_metadata_path)
            trainer_state = _load_json(trainer_state_path)
            history = _log_history(trainer_state)
            if not history:
                continue

            try:
                beta_val = float(meta.get("grpo_beta"))
            except (TypeError, ValueError):
                continue

            base: MutableMapping[str, object] = {
                "sweep_alias": alias,
                "run_name": meta.get("run_name"),
                "model_key": meta.get("model_key"),
                "model_name": meta.get("model_name"),
                "grpo_beta": beta_val,
                "lora_r": meta.get("lora_r"),
            }

            for entry in history:
                if not isinstance(entry, Mapping):
                    continue
                step = entry.get("step")
                if step is None:
                    continue
                try:
                    step_int = int(step)
                except (TypeError, ValueError):
                    continue

                record = dict(base)
                record["step"] = step_int
                for key, value in entry.items():
                    if key == "step":
                        continue
                    record[f"metric_{_sanitize_metric_name(key)}"] = _normalise(value)
                rows.append(record)

    if not rows:
        raise ValueError("No log records found – check SWEEPS and CHECKPOINT")

    df = pd.DataFrame(rows)
    df.sort_values(["sweep_alias", "grpo_beta", "lora_r", "step"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _prepare_pivots(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    subset = df[df["metric_kl"].notna()]
    if BETAS is not None:
        subset = subset[subset["grpo_beta"].isin(BETAS)]
    if subset.empty:
        raise ValueError("No KL data available for requested beta filter")

    pivots: dict[str, pd.DataFrame] = {}
    scatters: dict[str, pd.DataFrame] = {}
    for alias, alias_df in subset.groupby("sweep_alias"):
        pivot = (
            alias_df.groupby(["step", "lora_r"])["metric_kl"].mean().unstack("lora_r").sort_index(axis=1)
        )
        pivot = pivot.sort_index()
        if pivot.empty:
            continue
        pivots[alias] = pivot
        scatters[alias] = alias_df[["step", "lora_r", "metric_kl"]].copy()

    if not pivots:
        raise ValueError("No pivots created; nothing to plot")
    return pivots, scatters


def plot_kl(df: pd.DataFrame, output_path: Path) -> None:
    pivots, scatters = _prepare_pivots(df)
    all_ranks = sorted({rank for pivot in pivots.values() for rank in pivot.columns})
    cmap = plt.get_cmap("viridis", len(all_ranks) if all_ranks else 1)
    rank_colors = {rank: cmap(i) for i, rank in enumerate(all_ranks)}

    n_panels = len(pivots)
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 4 * n_panels), sharex=False)
    if n_panels == 1:
        axes = [axes]

    for ax, (alias, pivot) in zip(axes, pivots.items()):
        plot_df = pivot
        scatter_df = scatters[alias]

        if DOWNSAMPLE_EVERY and DOWNSAMPLE_EVERY > 1:
            sampled = plot_df.iloc[::DOWNSAMPLE_EVERY]
            if not sampled.empty and sampled.index[-1] != plot_df.index[-1]:
                sampled = pd.concat([sampled, plot_df.iloc[[-1]]])
            plot_df = sampled.sort_index()
            scatter_df = scatter_df[scatter_df["step"].isin(plot_df.index)].copy()

        if LOG_SCALE:
            positive = scatter_df[scatter_df["metric_kl"] > 0]
            if positive.empty:
                raise ValueError(f"Log-scale requested but sweep '{alias}' has no positive KL values")
            ax.set_yscale("log")

        if SHOW_POINTS:
            for rank in all_ranks:
                rank_points = scatter_df[scatter_df["lora_r"] == rank]
                if rank_points.empty:
                    continue
                if LOG_SCALE:
                    rank_points = rank_points[rank_points["metric_kl"] > 0]
                    if rank_points.empty:
                        continue
                ax.scatter(
                    rank_points["step"],
                    rank_points["metric_kl"],
                    color=rank_colors[rank],
                    alpha=0.3,
                    s=20,
                    edgecolors="none",
                )

        steps = plot_df.index.to_numpy(dtype=float)
        for rank in all_ranks:
            if rank not in plot_df.columns:
                continue
            series = plot_df[rank].to_numpy()
            if LOG_SCALE:
                series = np.where(series > 0, series, np.nan)
            label = f"rank {int(rank)}" if float(rank).is_integer() else f"rank {rank}"
            ax.plot(
                steps,
                series,
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=5,
                label=label,
                color=rank_colors[rank],
            )

        ax.set_title(str(alias))
        ax.set_ylabel("Mean KL")
        ax.grid(True, linestyle="--", alpha=0.3)
        if len(steps) > 0:
            ax.set_xlim(left=float(steps.min()), right=float(steps.max()))
        ax.set_xlabel("Training Step")
        if len(all_ranks) <= 12:
            ax.legend(loc="upper left", ncol=2, frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    df = load_training_logs(SWEEPS)
    plot_kl(df, OUTPUT_PATH)
    print(f"Saved KL plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
