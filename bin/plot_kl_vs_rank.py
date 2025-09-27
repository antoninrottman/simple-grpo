#!/usr/bin/env python3
"""Utilities for loading GRPO sweep logs and plotting KL trends."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SweepSpec = Union[str, Path, Tuple[str, Union[str, Path]]]

DEFAULT_LOG_KEYS: Sequence[str] | None = None


def _load_json(path: Path) -> Dict[str, object]:
    with path.open() as f:
        return json.load(f)


def _log_history(trainer_state: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    history = trainer_state.get("log_history") or []
    return history if isinstance(history, Sequence) else []


def _sanitize_metric_name(key: str) -> str:
    return key.replace("/", "_").replace(":", "_")


def _normalise_metric_value(value: object) -> object:
    return np.nan if value is None else value


def _iter_sweep_specs(
    sweep_specs: Mapping[str, SweepSpec] | Sequence[SweepSpec],
) -> Iterable[Tuple[Path, str | None]]:
    if isinstance(sweep_specs, Mapping):
        for alias, path in sweep_specs.items():
            yield Path(path), str(alias)
    else:
        for spec in sweep_specs:
            if isinstance(spec, tuple) and len(spec) == 2:
                alias, path = spec
                yield Path(path), str(alias)
            else:
                yield Path(spec), None


def load_training_logs(
    sweep_specs: Mapping[str, SweepSpec] | Sequence[SweepSpec],
    *,
    checkpoint: int = 300,
    step_interval: int | None = None,
    include_last_step: bool = True,
    log_keys: Sequence[str] | None = DEFAULT_LOG_KEYS,
    run_aliases: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with one row per (sweep, rank, step) log record."""
    target = f"checkpoint-{checkpoint}/trainer_state.json"
    rows: list[dict[str, object]] = []

    for root, sweep_alias in _iter_sweep_specs(sweep_specs):
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

            run_name = meta.get("run_name")
            run_alias = run_aliases.get(run_name, run_name) if run_aliases else run_name

            base: MutableMapping[str, object] = {
                "sweep_name": meta.get("sweep_name", root.name),
                "sweep_alias": sweep_alias or meta.get("sweep_name", root.name),
                "run_name": run_name,
                "run_alias": run_alias,
                "run_dir": str(run_dir),
                "trainer_state_path": str(trainer_state_path),
                "model_key": meta.get("model_key"),
                "model_name": meta.get("model_name"),
                "grpo_beta": beta_val,
                "lora_r": meta.get("lora_r"),
                "lora_alpha": meta.get("lora_alpha"),
                "global_step_max": trainer_state.get("max_steps"),
                "num_train_epochs": trainer_state.get("num_train_epochs"),
                "num_input_tokens_seen": trainer_state.get("num_input_tokens_seen"),
                "train_batch_size": trainer_state.get("train_batch_size"),
                "best_global_step": _normalise_metric_value(trainer_state.get("best_global_step")),
                "best_metric": _normalise_metric_value(trainer_state.get("best_metric")),
                "best_model_checkpoint": trainer_state.get("best_model_checkpoint"),
            }

            logged_steps: set[int] = set()
            last_entry: Mapping[str, object] | None = None

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
                last_entry = entry

                if step_interval is not None and step_interval > 0 and step_int % step_interval != 0:
                    continue

                logged_steps.add(step_int)
                record = dict(base)
                record["step"] = step_int

                keys_to_keep = entry.keys() if log_keys is None else log_keys
                for key in keys_to_keep:
                    if key == "step":
                        continue
                    col = f"metric_{_sanitize_metric_name(key)}"
                    record[col] = _normalise_metric_value(entry.get(key))

                rows.append(record)

            if include_last_step and last_entry is not None:
                step_val = last_entry.get("step")
                try:
                    step_int = int(step_val)
                except (TypeError, ValueError):
                    step_int = None
                if step_int is not None and step_int not in logged_steps:
                    record = dict(base)
                    record["step"] = step_int
                    keys_to_keep = last_entry.keys() if log_keys is None else log_keys
                    for key in keys_to_keep:
                        if key == "step":
                            continue
                        col = f"metric_{_sanitize_metric_name(key)}"
                        record[col] = _normalise_metric_value(last_entry.get(key))
                    rows.append(record)

    if not rows:
        raise ValueError("No log records found for the provided sweep paths")

    df = pd.DataFrame(rows)
    sort_cols = [col for col in ("sweep_alias", "sweep_name", "grpo_beta", "lora_r", "step") if col in df]
    if sort_cols:
        df.sort_values(sort_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def metric_slice(
    df: pd.DataFrame,
    *,
    metric: str,
    sweep_col: str = "sweep_alias",
) -> pd.Series:
    """Return a Series indexed by (sweep, beta, rank, step) for ``metric``."""
    if metric not in df:
        raise KeyError(f"metric '{metric}' not present in DataFrame columns")
    if sweep_col not in df:
        if sweep_col == "sweep_alias" and "sweep_name" in df:
            sweep_col = "sweep_name"
        else:
            raise KeyError(f"column '{sweep_col}' not present in DataFrame")
    return df.set_index([sweep_col, "grpo_beta", "lora_r", "step"])[metric]


def _prepare_pivots(
    df: pd.DataFrame,
    *,
    betas: Sequence[float] | None = None,
    exclude_betas: Sequence[float] | None = None,
    sweep_col: str = "sweep_alias",
) -> dict[str, pd.DataFrame]:
    if sweep_col not in df:
        sweep_col = "sweep_name"

    subset = df.copy()
    if betas is not None:
        subset = subset[subset["grpo_beta"].isin(betas)]
    if exclude_betas is not None:
        subset = subset[~subset["grpo_beta"].isin(exclude_betas)]

    subset = subset[subset["metric_kl"].notna()]
    if subset.empty:
        raise ValueError("No KL data available for the requested beta filters")

    pivots: dict[str, pd.DataFrame] = {}
    for alias, alias_df in subset.groupby(sweep_col):
        pivot = (
            alias_df.groupby(["step", "lora_r"])["metric_kl"]
            .mean()
            .unstack("lora_r")
            .sort_index(axis=1)
        )
        pivot = pivot.sort_index()
        if not pivot.empty:
            pivots[str(alias)] = pivot

    if not pivots:
        raise ValueError("No KL data available after grouping by sweep")

    return pivots


def plot_kl_panels(
    df: pd.DataFrame,
    output_path: Path,
    *,
    betas: Sequence[float] | None = None,
    exclude_betas: Sequence[float] | None = None,
    sweep_col: str = "sweep_alias",
    max_points: int = 12,
    log_scale: bool = False,
) -> None:
    """Plot KL-vs-step curves for each model on separate panels."""

    pivots = _prepare_pivots(
        df,
        betas=betas,
        exclude_betas=exclude_betas,
        sweep_col=sweep_col,
    )

    all_ranks = sorted({rank for pivot in pivots.values() for rank in pivot.columns})
    cmap = plt.get_cmap("viridis", len(all_ranks) if all_ranks else 1)
    rank_colors = {rank: cmap(i) for i, rank in enumerate(all_ranks)}

    n_panels = len(pivots)
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 4 * n_panels), sharex=False)
    if n_panels == 1:
        axes = [axes]

    for ax, (alias, pivot) in zip(axes, pivots.items()):
        plot_df = pivot
        if max_points and len(plot_df) > max_points:
            idx = np.linspace(0, len(plot_df) - 1, max_points, dtype=int)
            idx = np.unique(idx)
            plot_df = plot_df.iloc[idx]

        steps = plot_df.index.to_numpy(dtype=float)

        for rank in all_ranks:
            if rank not in plot_df.columns:
                continue
            series = plot_df[rank].to_numpy()
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
        ax.set_xlim(left=steps.min(), right=steps.max())
        ax.set_xlabel("Training Step")

        if log_scale:
            positive = plot_df.values[plot_df.values > 0]
            if positive.size == 0:
                raise ValueError("Cannot enable log scale: KL contains non-positive values")
            ax.set_yscale("log")

        if len(all_ranks) <= 12:
            ax.legend(loc="upper left", ncol=2, frameon=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_kl_panels_from_sweeps(
    sweeps: Mapping[str, SweepSpec] | Sequence[SweepSpec],
    output_path: Path,
    *,
    betas: Sequence[float] | None = None,
    exclude_betas: Sequence[float] | None = None,
    checkpoint: int = 300,
    step_interval: int | None = None,
    max_points: int = 12,
    log_scale: bool = False,
) -> None:
    """Helper that loads logs and plots KL panels in one step."""
    df = load_training_logs(
        sweeps,
        checkpoint=checkpoint,
        step_interval=step_interval,
    )
    plot_kl_panels(
        df,
        output_path,
        betas=betas,
        exclude_betas=exclude_betas,
        max_points=max_points,
        log_scale=log_scale,
    )


__all__ = [
    "load_training_logs",
    "metric_slice",
    "plot_kl_panels",
    "plot_kl_panels_from_sweeps",
]
