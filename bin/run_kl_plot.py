#!/usr/bin/env python3
"""Generate KL vs step plots for configured sweeps."""
from __future__ import annotations

from pathlib import Path

from plot_kl_vs_rank import plot_kl_panels_from_sweeps


def main() -> None:
    sweeps = {
        "gemma": "outputs/results_run_20250926-133418",
        "qwen": "outputs/results_run_20250926-083327",
        "llama": "outputs/llama_r_sweep_results",
    }

    plot_kl_panels_from_sweeps(
        sweeps,
        Path("tmp/kl_vs_rank_beta005.png"),
        betas=[0.05],
        max_points=20,
        log_scale=True,
    )
    print("Saved KL plot to tmp/kl_vs_rank_beta005.png")


if __name__ == "__main__":
    main()
