#!/usr/bin/env python3
"""Plot KL divergence versus LoRA rank for selected sweeps."""
from __future__ import annotations

import argparse
from pathlib import Path

from training_data_processing import aggregate_sweeps, plot_kl_vs_rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot KL vs rank heatmap from training logs")
    parser.add_argument(
        "--sweep",
        action="append",
        dest="sweeps",
        default=[],
        help="Sweep alias and path (alias=path). Repeat for multiple sweeps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the heatmap image",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=300,
        help="Checkpoint step to read trainer_state logs from (default: 300)",
    )
    parser.add_argument(
        "--beta",
        dest="betas",
        action="append",
        type=float,
        help="Restrict to one or more beta values (repeat flag for multiple)",
    )
    parser.add_argument(
        "--exclude-beta",
        dest="exclude_betas",
        action="append",
        type=float,
        help="Exclude one or more beta values from the plot",
    )
    return parser.parse_args()


def parse_sweep_specs(specs: list[str]) -> dict[str, str]:
    if not specs:
        raise SystemExit("At least one --sweep alias=path entry is required")
    mapping: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid sweep spec '{spec}'. Expected alias=path format.")
        alias, path = spec.split("=", 1)
        mapping[alias.strip()] = path.strip()
    return mapping


def main() -> None:
    args = parse_args()
    sweeps = parse_sweep_specs(args.sweeps)

    df = aggregate_sweeps(sweeps, checkpoint=args.checkpoint, step_interval=None)
    plot_kl_vs_rank(
        df,
        args.output,
        betas=args.betas,
        exclude_betas=args.exclude_betas,
    )
    print(f"Saved KL vs rank heatmap to {args.output}")


if __name__ == "__main__":
    main()
