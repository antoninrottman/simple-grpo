#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SUBMIT_SCRIPT="$SCRIPT_DIR/submit_grpo_run.sh"

if [[ ! -x $SUBMIT_SCRIPT ]]; then
  echo "submit_grpo_run.sh is missing or not executable" >&2
  exit 1
fi

DEFAULT_MODELS=(gemma llama qwen)
DEFAULT_BETAS=(0 0.05)
DEFAULT_LORA_VALUES=(1 2 4 8 16 32 64)
PAIR_MODE=1
SWEEP_NAME=""
EVAL_MODE="CLI"
DRY_RUN=0
SELECTED_MODELS=()
SELECTED_BETAS=()
SELECTED_LORA=()

usage() {
  cat <<'EOF'
Usage: submit_full_sweep.sh [options]

Options:
  --sweep-name NAME        Override results_run_* directory name
  --models LIST            Comma-separated subset of models (default: gemma,llama,qwen)
  --betas LIST             Comma-separated beta values (default: 0,0.05)
  --lora-values LIST       Comma-separated LoRA values (default: 1,2,4,8,16,32,64)
  --full-grid              Use full grid over LoRA rank/alpha (default pairs each value)
  --eval-mode MODE         CLI (default) or NONE
  --dry-run                Print sbatch commands without submitting
  -h, --help               Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --sweep-name)
      SWEEP_NAME=$2; shift 2 ;;
    --models)
      IFS=, read -r -a SELECTED_MODELS <<< "$2"; shift 2 ;;
    --betas)
      IFS=, read -r -a SELECTED_BETAS <<< "$2"; shift 2 ;;
    --lora-values)
      IFS=, read -r -a SELECTED_LORA <<< "$2"; shift 2 ;;
    --full-grid)
      PAIR_MODE=0; shift ;;
    --eval-mode)
      EVAL_MODE=$2; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift; break ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

MODELS=(${SELECTED_MODELS[@]:-${DEFAULT_MODELS[@]}})
BETAS=(${SELECTED_BETAS[@]:-${DEFAULT_BETAS[@]}})
LORA_VALUES=(${SELECTED_LORA[@]:-${DEFAULT_LORA_VALUES[@]}})

for model in "${MODELS[@]}"; do
  case $model in
    gemma|llama|qwen) ;;
    *) echo "Invalid model: $model" >&2; exit 1 ;;
  esac
done

case $EVAL_MODE in
  CLI|NONE) ;;
  *) echo "Invalid eval mode: $EVAL_MODE" >&2; exit 1 ;;
esac

if [[ -z $SWEEP_NAME ]]; then
  SWEEP_NAME="results_run_$(date -u +%Y%m%d-%H%M%S)"
fi

declare -a JOB_ROWS
JOBS_COUNT=0

submit_combo() {
  local model=$1
  local beta=$2
  local r=$3
  local alpha=$4
  local -a args=(
    --model "$model"
    --beta "$beta"
    --lora-r "$r"
    --lora-alpha "$alpha"
    --sweep-name "$SWEEP_NAME"
    --eval-mode "$EVAL_MODE"
  )
  if [[ $DRY_RUN -eq 1 ]]; then
    args+=(--dry-run)
  fi
  local output
  output=$("$SUBMIT_SCRIPT" "${args[@]}")
  printf "%s
" "$output"
  local job_id="-"
  if [[ $DRY_RUN -eq 0 && $output =~ job_id=([0-9]+) ]]; then
    job_id=${BASH_REMATCH[1]}
  fi
  JOB_ROWS+=("$model,$beta,$r,$alpha,$job_id")
  ((JOBS_COUNT++))
}

for model in "${MODELS[@]}"; do
  for beta in "${BETAS[@]}"; do
    if [[ $PAIR_MODE -eq 1 ]]; then
      for val in "${LORA_VALUES[@]}"; do
        submit_combo "$model" "$beta" "$val" "$val"
      done
    else
      for r in "${LORA_VALUES[@]}"; do
        for alpha in "${LORA_VALUES[@]}"; do
          submit_combo "$model" "$beta" "$r" "$alpha"
        done
      done
    fi
  done
done

printf "
[submit_full_sweep] sweep=%s jobs=%d dry_run=%s eval=%s
"   "$SWEEP_NAME" "$JOBS_COUNT" "$DRY_RUN" "$EVAL_MODE"
printf "model,beta,lora_r,lora_alpha,job_id
"
for row in "${JOB_ROWS[@]}"; do
  printf "%s
" "$row"
done
