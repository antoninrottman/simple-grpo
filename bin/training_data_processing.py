#!/usr/bin/env python3
"""GRPO sweep utilities for per-step aggregation and visualisation.

Typical usage inside a notebook::

    from bin.training_data_processing import aggregate_sweeps, metric_slice, plot_kl_vs_rank

    sweeps = {
        "gemma": "outputs/results_run_20250926-133418",
        "qwen": "outputs/results_run_20250926-083327",
        "llama": "outputs/llama_r_sweep_results",
    }

    df = aggregate_sweeps(sweeps, step_interval=None)
    reward_series = metric_slice(df, metric="metric_reward")
    gemma_curve = reward_series.loc["gemma", 0.05, 16]

    plot_path = Path("tmp/kl_vs_rank_beta005.png")
    plot_kl_vs_rank(df, plot_path, betas=[0.05])

The aggregation keeps **every** recorded training step (unless a custom
``step_interval`` is supplied), normalises missing values to ``NaN`` and stores
all metrics appearing in the trainer log with the prefix ``metric_``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

SweepSpec = Union[str, Path, Tuple[str, Union[str, Path]]]

# ``None`` = capture every key from the log entry.
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


def iter_run_records(
    sweep_specs: Mapping[str, SweepSpec] | Sequence[SweepSpec],
    *,
    checkpoint: int = 300,
    step_interval: int | None = None,
    include_last_step: bool = True,
    log_keys: Sequence[str] | None = DEFAULT_LOG_KEYS,
    run_aliases: Mapping[str, str] | None = None,
) -> Iterable[Dict[str, object]]:
    """Yield one record per run *and* log step from the provided sweep paths."""
    target = f"checkpoint-{checkpoint}/trainer_state.json"

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

            base_record: MutableMapping[str, object] = {
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
                record = dict(base_record)
                record["step"] = step_int

                keys_to_keep = entry.keys() if log_keys is None else log_keys
                for key in keys_to_keep:
                    if key == "step":
                        continue
                    col = f"metric_{_sanitize_metric_name(key)}"
                    record[col] = _normalise_metric_value(entry.get(key))

                yield record

            if include_last_step and last_entry is not None:
                step_val = last_entry.get("step")
                try:
                    step_int = int(step_val)
                except (TypeError, ValueError):
                    step_int = None
                if step_int is not None and step_int not in logged_steps:
                    record = dict(base_record)
                    record["step"] = step_int
                    keys_to_keep = last_entry.keys() if log_keys is None else log_keys
                    for key in keys_to_keep:
                        if key == "step":
                            continue
                        col = f"metric_{_sanitize_metric_name(key)}"
                        record[col] = _normalise_metric_value(last_entry.get(key))
                    yield record


def aggregate_sweeps(
    sweep_specs: Mapping[str, SweepSpec] | Sequence[SweepSpec],
    *,
    checkpoint: int = 300,
    step_interval: int | None = None,
    include_last_step: bool = True,
    log_keys: Sequence[str] | None = DEFAULT_LOG_KEYS,
    run_aliases: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    rows = list(
        iter_run_records(
            sweep_specs,
            checkpoint=checkpoint,
            step_interval=step_interval,
            include_last_step=include_last_step,
            log_keys=log_keys,
            run_aliases=run_aliases,
        )
    )
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
    if metric not in df:
        raise KeyError(f"metric '{metric}' not present in DataFrame columns")
    if sweep_col not in df:
        if sweep_col == "sweep_alias" and "sweep_name" in df:
            sweep_col = "sweep_name"
        else:
            raise KeyError(f"column '{sweep_col}' not present in DataFrame")
    return df.set_index([sweep_col, "grpo_beta", "lora_r", "step"])[metric]


def plot_kl_vs_rank(
    df: pd.DataFrame,
    output_path: Path,
    *,
    betas: Sequence[float] | None = None,
    exclude_betas: Sequence[float] | None = None,
    sweep_col: str = "sweep_alias",
    log_scale: bool = False,
) -> None:
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

    pivot = (
        subset.groupby(["step", "lora_r"])["metric_kl"].mean().unstack("lora_r").sort_index(axis=1)
    )
    steps = pivot.index.to_numpy(dtype=float)
    ranks = pivot.columns.to_numpy(dtype=float)
    Z = pivot.to_numpy()

    def _edges(values: np.ndarray) -> np.ndarray:
        if values.size == 1:
            delta = values[0] * 0.5 if values[0] != 0 else 0.5
            return np.array([values[0] - delta, values[0] + delta], dtype=float)
        diffs = np.diff(values)
        start = values[0] - diffs[0] / 2
        end = values[-1] + diffs[-1] / 2
        mids = (values[:-1] + values[1:]) / 2
        return np.concatenate(([start], mids, [end]))

    step_edges = _edges(steps)
    rank_edges = _edges(ranks)

    norm = None
    if log_scale:
        positive = Z[Z > 0]
        if positive.size == 0:
            raise ValueError("Cannot apply logarithmic scale: KL contains non-positive values")
        norm = mcolors.LogNorm(vmin=positive.min(), vmax=positive.max())

    fig, (ax_line, ax_heat) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        gridspec_kw={"height_ratios": [1, 2]},
        sharex=False,
    )

    for rank, series in pivot.items():
        ax_line.plot(steps, series.to_numpy(), label=f"rank {int(rank) if float(rank).is_integer() else rank}")
    ax_line.set_ylabel("Mean KL")
    ax_line.set_yscale("log")
    ax_line.set_xlabel("Training Step")
    if pivot.shape[1] <= 12:
        ax_line.legend(loc="upper right", ncol=2)
    ax_line.grid(True, linestyle="--", alpha=0.3)

    mesh = ax_heat.pcolormesh(rank_edges, step_edges, Z, cmap="magma", norm=norm, shading="auto")
    ax_heat.set_xlabel("LoRA Rank")
    ax_heat.set_ylabel("Training Step")
    if betas is not None:
        beta_text = ", ".join(f"{b:.2f}" for b in sorted(set(betas)))
        title_suffix = f"β ∈ {{{beta_text}}}"
    elif exclude_betas is not None:
        beta_text = ", ".join(f"{b:.2f}" for b in sorted(set(exclude_betas)))
        title_suffix = f"β ∉ {{{beta_text}}}"
    else:
        title_suffix = "all β"
    ax_heat.set_title(f"Mean KL vs Rank ({title_suffix})")
    ax_heat.set_xticks(ranks)
    ax_heat.set_xticklabels([int(float(r)) if float(r).is_integer() else float(r) for r in ranks], rotation=45, ha="right")

    if len(steps) <= 20:
        ax_heat.set_yticks(steps)
        ax_heat.set_yticklabels([int(float(s)) if float(s).is_integer() else float(s) for s in steps])
    else:
        idx = np.linspace(0, steps.size - 1, min(10, steps.size), dtype=int)
        idx = np.unique(idx)
        chosen_steps = steps[idx]
        ax_heat.set_yticks(chosen_steps)
        ax_heat.set_yticklabels([int(float(s)) if float(s).is_integer() else float(s) for s in chosen_steps])

    cbar = fig.colorbar(mesh, ax=ax_heat)
    cbar.set_label("Mean KL")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


__all__ = [
    "aggregate_sweeps",
    "iter_run_records",
    "metric_slice",
    "plot_kl_vs_rank",
]


def main() -> None:
    sweeps = {
        "gemma": "outputs/results_run_20250926-133418",
        "qwen": "outputs/results_run_20250926-083327",
        "llama": "outputs/llama_r_sweep_results",
    }

    df = aggregate_sweeps(sweeps, step_interval=None)
    output_path = Path("tmp/kl_vs_rank_beta005.png")
    try:
        plot_kl_vs_rank(df, output_path, betas=[0.05])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Saved KL heatmap to {output_path}")


if __name__ == "__main__":
    main()
