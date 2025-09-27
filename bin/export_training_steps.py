#!/usr/bin/env python3
"""Helpers for exporting per-step training logs into CSV files."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Union

import pandas as pd

from training_data_processing import aggregate_sweeps

SweepSpec = Union[str, Path]


def export_training_steps(
    sweeps: Mapping[str, SweepSpec] | Sequence[SweepSpec],
    *,
    output_path: Path | None = None,
    checkpoint: int = 300,
    step_interval: int | None = None,
) -> pd.DataFrame:
    """Aggregate per-step logs and optionally write them to ``output_path``.

    Parameters
    ----------
    sweeps
        Either a mapping of aliases to sweep directories or a plain sequence of
        sweep directories (aliases will default to directory names).
    output_path
        If provided, the resulting DataFrame is written to this CSV path.
    checkpoint
        Which checkpoint folder to read each ``trainer_state.json`` from.
    step_interval
        Optional stride for keeping only every Nth logged step.
    """
    df = aggregate_sweeps(
        sweeps,
        checkpoint=checkpoint,
        step_interval=step_interval,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def main() -> None:
    sweeps = {
        "gemma": "outputs/results_run_20250926-133418",
        "qwen": "outputs/results_run_20250926-083327",
        "llama": "outputs/llama_r_sweep_results",
    }
    output_path = Path("tmp/per_step_metrics.csv")
    df = export_training_steps(sweeps, output_path=output_path)
    print(f"Exported {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
