#!/usr/bin/env python3
"""Generate KL vs step plots for configured sweeps."""
from __future__ import annotations

import argparse
from pathlib import Path

from plot_kl_vs_rank import plot_kl_panels_from_sweeps


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KL vs step with optional downsampling.")
    parser.add_argument(
        "--downsample",
        type=int,
        default=10,
        help="Keep every Nth training step in the plot (default: 10, use 1 to disable).",
    )
    args = parser.parse_args()
    downsample_every = args.downsample if args.downsample and args.downsample > 1 else None

    sweeps = {
        "gemma": "outputs/results_run_20250926-133418",
        "qwen": "outputs/results_run_20250926-083327",
        "llama": "outputs/llama_r_sweep_results",
    }

    plot_kl_panels_from_sweeps(
        sweeps,
        Path("tmp/kl_vs_rank_beta005.png"),
        betas=[0.05],
        log_scale=True,
        show_points=False,
        downsample_every=downsample_every,
    )
    print("Saved KL plot to tmp/kl_vs_rank_beta005.png")


if __name__ == "__main__":
    main()
