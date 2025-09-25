#!/bin/bash
#SBATCH --no-requeue
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=180G

set -euo pipefail

module load gcc/11.3.0
module load cuda/11.8.0
source /home/rottman/simple-grpo/.venv/bin/activate

MODEL_KEY=${MODEL_KEY:?MODEL_KEY must be set to google-gemma-3-1b-it, Qwen-Qwen2.5-1.5B-Instruct, Qwen-Qwen2.5-3B-Instruct, meta-llama-Llama-3.2-1B-Instruct, or meta-llama-Llama-3.2-3B-Instruct}
RUN_NAME=${RUN_NAME:?RUN_NAME must be set}
SWEEP_NAME=${SWEEP_NAME:?SWEEP_NAME must be set (e.g. results_run_YYYYmmdd-HHMM)}
GRPO_BETA=${GRPO_BETA:?GRPO_BETA must be set}
LORA_R=${LORA_R:?LORA_R must be set}
LORA_ALPHA=${LORA_ALPHA:-$LORA_R}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/izar/rottman/simple-grpo}
HOME_REPO=${HOME_REPO:-/home/rottman/simple-grpo}
RECIPE_ROOT=${HOME_REPO}/recipes
RESULTS_ROOT=${RESULTS_ROOT:-$HOME_REPO/outputs}
EVAL_MODE=${EVAL_MODE:-CLI}
HF_MODEL_NAME=${HF_MODEL_NAME:-}

case "$MODEL_KEY" in
  google-gemma-3-1b-it)
    DEFAULT_MODEL_NAME="google/gemma-3-1b-it"
    ;;
  Qwen-Qwen2.5-1.5B-Instruct)
    DEFAULT_MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
    ;;
  Qwen-Qwen2.5-3B-Instruct)
    DEFAULT_MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
    ;;
  meta-llama-Llama-3.2-1B-Instruct)
    DEFAULT_MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
    ;;
  meta-llama-Llama-3.2-3B-Instruct)
    DEFAULT_MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
    ;;
  *)
    echo "Unknown MODEL_KEY: $MODEL_KEY" >&2
    exit 1
    ;;
esac
MODEL_NAME=${HF_MODEL_NAME:-$DEFAULT_MODEL_NAME}

echo "[grpo_train_job] model=$MODEL_KEY hf_model=$MODEL_NAME beta=$GRPO_BETA lora_r=$LORA_R lora_alpha=$LORA_ALPHA run=$RUN_NAME sweep=$SWEEP_NAME"

SCRATCH_RUN_ROOT="$SCRATCH_ROOT/$SWEEP_NAME/$RUN_NAME"
SCRATCH_RECIPE_DIR="$SCRATCH_RUN_ROOT/$MODEL_KEY"
SCRATCH_BIN_DIR="$SCRATCH_RUN_ROOT/bin"

mkdir -p "$RESULTS_ROOT/$SWEEP_NAME"

mkdir -p "$SCRATCH_RECIPE_DIR" "$SCRATCH_BIN_DIR"

RSYNC=$(command -v rsync || true)
if [[ -n "$RSYNC" ]]; then
  rsync -a --delete --exclude 'outputs' --exclude 'wandb' "$RECIPE_ROOT/$MODEL_KEY/" "$SCRATCH_RECIPE_DIR/"
  rsync -a --delete "$HOME_REPO/bin/" "$SCRATCH_BIN_DIR/"
else
  cp -a "$RECIPE_ROOT/$MODEL_KEY/." "$SCRATCH_RECIPE_DIR/"
  cp -a "$HOME_REPO/bin/." "$SCRATCH_BIN_DIR/"
fi

cd "$SCRATCH_RECIPE_DIR"

echo "[grpo_train_job] working directory: $SCRATCH_RECIPE_DIR"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export MODEL_NAME
export OUTPUT_DIR="$SCRATCH_RECIPE_DIR/outputs/$RUN_NAME"
export RUN_NAME
export EVAL_OUTPUT_DIR="$OUTPUT_DIR/evaluation_results"
export MERGED_DIR="$OUTPUT_DIR/merged_model"
export LORA_R
export LORA_ALPHA
export GRPO_BETA
export EVAL="$EVAL_MODE"
export WANDB_PROJECT="llama_sweep_25sept"

mkdir -p "$EVAL_OUTPUT_DIR" "$MERGED_DIR"

python train_grpo.py


if [[ "$EVAL_MODE" == "CLI" ]]; then

  # set up openr1 environment to run the evaluation.
  module purge
  module load cmake/3.23.1
  module load gcc/11.3.0
  source /home/rottman/builds/openr1/bin/activate
  
  echo "[grpo_train_job] running evaluation via CLI"
  bash "$SCRATCH_BIN_DIR/run_eval.sh" "$MERGED_DIR" "$EVAL_OUTPUT_DIR" "$MODEL_NAME"
fi

STAGING_DIR="$SCRATCH_RUN_ROOT/staging"
ADAPTER_STAGE="$STAGING_DIR/lora_adapters"
EVAL_STAGE="$STAGING_DIR/evaluation_results"
mkdir -p "$ADAPTER_STAGE" "$EVAL_STAGE"

shopt -s nullglob
for ckpt in "$OUTPUT_DIR"/checkpoint-*; do
  [[ -d "$ckpt" ]] || continue
  ckpt_name=$(basename "$ckpt")
  dest="$ADAPTER_STAGE/$ckpt_name"
  mkdir -p "$dest"
  files=(
    "adapter_model.safetensors"
    "adapter_config.json"
    "added_tokens.json"
    "all_results.json"
    "chat_template.jinja"
    "config.json"
    "merges.txt"
    "special_tokens_map.json"
    "tokenizer_config.json"
    "tokenizer.json"
    "tokenizer.model"
    "trainer_state.json"
    "training_args.bin"
    "train_results.json"
    "vocab.json"
  )
  for file in "${files[@]}"; do
    if [[ -f "$ckpt/$file" ]]; then
      cp "$ckpt/$file" "$dest/$file"
    fi
  done
done
shopt -u nullglob

if [[ -d "$EVAL_OUTPUT_DIR" ]]; then
  if [[ -n "$RSYNC" ]]; then
    rsync -a "$EVAL_OUTPUT_DIR/" "$EVAL_STAGE/"
  else
    cp -a "$EVAL_OUTPUT_DIR/." "$EVAL_STAGE/"
  fi
fi

cat >"$STAGING_DIR/run_metadata.json" <<EOF
{
  "model_key": "$MODEL_KEY",
  "model_name": "$MODEL_NAME",
  "run_name": "$RUN_NAME",
  "sweep_name": "$SWEEP_NAME",
  "grpo_beta": $GRPO_BETA,
  "lora_r": $LORA_R,
  "lora_alpha": $LORA_ALPHA,
  "evaluation_mode": "$EVAL_MODE",
  "scratch_output_dir": "$OUTPUT_DIR",
  "slurm_job_id": "${SLURM_JOB_ID:-unknown}",
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

RESULTS_DEST="$RESULTS_ROOT/$SWEEP_NAME/$MODEL_KEY/$RUN_NAME"
mkdir -p "$RESULTS_DEST" "$RESULTS_DEST/lora_adapters" "$RESULTS_DEST/evaluation_results"

echo "[grpo_train_job] syncing results to $RESULTS_DEST"

if [[ -n "$RSYNC" ]]; then
  rsync -a "$ADAPTER_STAGE/" "$RESULTS_DEST/lora_adapters/"
  rsync -a "$EVAL_STAGE/" "$RESULTS_DEST/evaluation_results/"
else
  cp -a "$ADAPTER_STAGE/." "$RESULTS_DEST/lora_adapters/"
  cp -a "$EVAL_STAGE/." "$RESULTS_DEST/evaluation_results/"
fi
cp "$STAGING_DIR/run_metadata.json" "$RESULTS_DEST/run_metadata.json"

LATEST_CKPT=$(ls -1d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [[ -n "$LATEST_CKPT" ]]; then
  LAST_CKPT_DEST="$RESULTS_DEST/last_checkpoint"
  echo "[grpo_train_job] copying latest checkpoint $(basename "$LATEST_CKPT") to $RESULTS_DEST"
  if [[ -n "$RSYNC" ]]; then
    rsync -a --delete "$LATEST_CKPT/" "$LAST_CKPT_DEST/"
  else
    rm -rf "$LAST_CKPT_DEST"
    mkdir -p "$LAST_CKPT_DEST"
    cp -a "$LATEST_CKPT/." "$LAST_CKPT_DEST/"
  fi
fi
