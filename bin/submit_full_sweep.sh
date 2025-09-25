#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SUBMIT_SCRIPT="$SCRIPT_DIR/submit_grpo_run.sh"

if [[ ! -x $SUBMIT_SCRIPT ]]; then
  echo "submit_grpo_run.sh is missing or not executable" >&2
  exit 1
fi

DEFAULT_MODELS=(google-gemma-3-1b-it Qwen-Qwen2.5-1.5B-Instruct Qwen-Qwen2.5-3B-Instruct meta-llama-Llama-3.2-1B-Instruct meta-llama-Llama-3.2-3B-Instruct)
DEFAULT_BETAS=(0 0.05)
DEFAULT_LORA_VALUES=( 4 8 16 32 64 128)
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
  --models LIST            Comma-separated subset of models (default: google-gemma-3-1b-it,Qwen-Qwen2.5-1.5B-Instruct,Qwen-Qwen2.5-3B-Instruct,meta-llama-Llama-3.2-1B-Instruct,meta-llama-Llama-3.2-3B-Instruct)
  --betas LIST             Comma-separated beta values (default: 0,0.05)
  --lora-values LIST       Comma-separated LoRA ranks (default: 1,2,4,8,16,32,64)
  --eval-mode MODE         CLI (default) or NONE
  --dry-run                Print sbatch commands without submitting
  -h, --help               Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --sweep-name) SWEEP_NAME=$2; shift 2 ;;
    --models) IFS=, read -r -a SELECTED_MODELS <<<"$2"; shift 2 ;;
    --betas) IFS=, read -r -a SELECTED_BETAS <<<"$2"; shift 2 ;;
    --lora-values) IFS=, read -r -a SELECTED_LORA <<<"$2"; shift 2 ;;
    --eval-mode) EVAL_MODE=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

MODELS=(${SELECTED_MODELS[@]:-${DEFAULT_MODELS[@]}})
BETAS=(${SELECTED_BETAS[@]:-${DEFAULT_BETAS[@]}})
LORA_VALUES=(${SELECTED_LORA[@]:-${DEFAULT_LORA_VALUES[@]}})

for model in "${MODELS[@]}"; do
  case $model in
    google-gemma-3-1b-it|Qwen-Qwen25-1.5B-Instruct|Qwen-Qwen2.5-3B-Instruct|meta-llama-Llama-3.2-1B-Instruct|meta-llama-Llama-3.2-3B-Instruct) ;;
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

TMPDIR=$(mktemp -d -t submit_full_sweep.XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT

COMBOS=()
for model in "${MODELS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for r in "${LORA_VALUES[@]}"; do
      COMBOS+=("$model,$beta,$r")
    done
  done
done

JOBS_COUNT=${#COMBOS[@]}
if [[ $JOBS_COUNT -eq 0 ]]; then
  echo "No jobs to submit." >&2
  exit 0
fi

declare -a PIDS=()
declare -a LOGS=()
declare -a META=()
declare -a SUMMARY=()

submit_combo() {
  local model=$1 beta=$2 r=$3
  local -a args=(
    --model "$model"
    --beta "$beta"
    --lora-r "$r"
    --sweep-name "$SWEEP_NAME"
    --eval-mode "$EVAL_MODE"
  )
  if [[ $DRY_RUN -eq 1 ]]; then
    "$SUBMIT_SCRIPT" "${args[@]}"
    SUMMARY+=("$model,$beta,$r,$r,-")
    return
  fi
  echo "[submit_full_sweep] submitting model=$model beta=$beta r=$r"
  local log="$TMPDIR/job_${#PIDS[@]}.log"
  (
    set +e
    "$SUBMIT_SCRIPT" "${args[@]}" >"$log" 2>&1
  ) &
  PIDS+=("$!")
  LOGS+=("$log")
  META+=("$model,$beta,$r,$r")
}

for combo in "${COMBOS[@]}"; do
  IFS=, read -r model beta r <<<"$combo"
  submit_combo "$model" "$beta" "$r"
done

if [[ $DRY_RUN -eq 1 ]]; then
  printf '
[submit_full_sweep] sweep=%s jobs=%d eval=%s (dry run)
' "$SWEEP_NAME" "$JOBS_COUNT" "$EVAL_MODE"
  printf 'model,beta,lora_r,lora_alpha,job_id
'
  for row in "${SUMMARY[@]}"; do
    printf '%s
' "$row"
  done
  exit 0
fi

for idx in "${!PIDS[@]}"; do
  pid=${PIDS[$idx]}
  log=${LOGS[$idx]}
  meta=${META[$idx]}
  if ! wait "$pid"; then
    echo "[submit_full_sweep] warning: submission failed for $meta" >&2
  fi
  output=$(<"$log")
  printf '%s
' "$output"
  job_id="-"
  if [[ $output =~ job_id=([0-9]+) ]]; then
    job_id=${BASH_REMATCH[1]}
  elif [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    job_id=${BASH_REMATCH[1]}
  fi
  SUMMARY+=("$meta,$job_id")
  rm -f "$log"
done

printf '
[submit_full_sweep] sweep=%s jobs=%d eval=%s
' "$SWEEP_NAME" "$JOBS_COUNT" "$EVAL_MODE"
printf 'model,beta,lora_r,lora_alpha,job_id
'
for row in "${SUMMARY[@]}"; do
  printf '%s
' "$row"
done
