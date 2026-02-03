# simple-grpo

Slurm-based training and evaluation runs for GRPO + LoRA across a small set of Hugging Face instruction-tuned models. Each model has its own recipe folder, and runs are submitted via the scripts in `bin/`.

## Repository layout

- `bin/`: Submission and utility scripts.
- `recipes/<hf-model-id>/`: Model-specific training/eval scripts (e.g., `train_grpo.py`).
- `outputs/`: Results are written to `outputs/<sweep>/<model>/<run>/` by default.
- `logs/`: Slurm stdout/stderr per run.
- `main.py`: Environment sanity check (prints versions, GPU info, runs a tiny torch matmul).
- `hpc_test.sh`: Simple Slurm smoke test that runs `main.py`.

## Supported recipes

- `google-gemma-3-1b-it`
- `Qwen-Qwen2.5-1.5B-Instruct`
- `Qwen-Qwen2.5-3B-Instruct`
- `meta-llama-Llama-3.2-1B-Instruct`
- `meta-llama-Llama-3.2-3B-Instruct`

## Setup notes

- Dependencies are declared in `pyproject.toml`.
- The Slurm job script expects an active virtual environment at `./.venv` when it runs.
- CUDA 11.8 modules are loaded inside `bin/grpo_train_job.sh`.

## Submit a single run

Use `bin/submit_grpo_run.sh` to submit one job with specific hyperparameters:

```bash
bin/submit_grpo_run.sh \
  --model google-gemma-3-1b-it \
  --beta 0.05 \
  --lora-r 16
```

Common options:

- `--sweep-name NAME`: Top-level results folder (defaults to `results_run_YYYYmmdd-HHMMSS`).
- `--run-name NAME`: Override the run name used in `outputs/`.
- `--eval-mode CLI|NONE`: Run evaluation after training or skip it.
- `--dry-run`: Print the `sbatch` command without submitting.

## Submit a full sweep

Use `bin/submit_full_sweep.sh` to submit a grid over models, betas, and LoRA ranks:

```bash
bin/submit_full_sweep.sh \
  --sweep-name results_run_20260203-120000 \
  --betas 0,0.05 \
  --lora-values 4,8,16 \
  --eval-mode CLI
```

Defaults in the script:

- Models: all supported recipes above
- Betas: `0`, `0.05`
- LoRA ranks: `4,8,16,32,64,128`

Add `--dry-run` to print planned submissions without launching jobs.

## Evaluation

If `--eval-mode CLI` is used, `bin/grpo_train_job.sh` runs `bin/run_eval.sh` after training. The job switches modules and activates the open-r1 environment before running evaluation. This evaluates the merged model with `lighteval` on:

- `math_500`
- `gpqa:diamond`
- `gsm8k`

Outputs are written to `outputs/<sweep>/<model>/<run>/evaluation_results/`.

## Results layout

Each run syncs a compact artifact set back to the repo:

- `outputs/<sweep>/<model>/<run>/lora_adapters/`: LoRA adapter checkpoints and configs
- `outputs/<sweep>/<model>/<run>/evaluation_results/`: lighteval JSON outputs
- `outputs/<sweep>/<model>/<run>/run_metadata.json`: run metadata, hyperparams, timestamps

## Utilities

- `bin/collect_evals.py`: Aggregate evaluation JSONs into a CSV and (optionally) plots.
- `bin/collect_train_metrics.py`: Summarize training metrics across runs.
- `bin/eval_to_latex.py`: Convert eval summaries into LaTeX tables.
- `bin/run_kl_plot.py`: Plot KL/metric comparisons across betas.
- `bin/training_monitor.py`: Tail and summarize training logs.
