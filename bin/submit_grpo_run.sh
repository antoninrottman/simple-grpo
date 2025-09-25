#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: submit_grpo_run.sh --model MODEL --beta VALUE --lora-r VALUE [options]

Required arguments:
  --model MODEL            One of: google-gemma-3-1b-it, Qwen-Qwen2.5-1.5B-Instruct, Qwen-Qwen2.5-3B-Instruct, meta-llama-Llama-3.2-1B-Instruct, meta-llama-Llama-3.2-3B-Instruct
  --beta VALUE             GRPO beta value (e.g. 0 or 0.05)
  --lora-r VALUE           LoRA rank (integer)

Optional arguments:
  --sweep-name NAME        Top-level results directory name (default: results_run_<timestamp>)
  --run-name NAME          Override run identifier used for OUTPUT_DIR and results
  --job-name NAME          Override Slurm job name
  --hf-model NAME          Override Hugging Face model name
  --eval-mode MODE         CLI (default) or NONE
  --scratch-root PATH      Scratch root (default: /scratch/izar/rottman/simple-grpo)
  --results-root PATH      Root where results_run_* directories are written (default: repo_root/outputs)
  --dry-run                Print sbatch command without submitting
  -h, --help               Show this message
EOF
}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEFAULT_SCRATCH="/scratch/izar/rottman/simple-grpo"
DEFAULT_EVAL_MODE="CLI"
DEFAULT_RESULTS_ROOT="$REPO_ROOT/outputs"

MODEL=""
BETA=""
LORA_R=""
SWEEP_NAME=""
RUN_NAME=""
JOB_NAME=""
HF_MODEL=""
EVAL_MODE="$DEFAULT_EVAL_MODE"
SCRATCH_ROOT="$DEFAULT_SCRATCH"
RESULTS_ROOT="$DEFAULT_RESULTS_ROOT"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL=$2; shift 2 ;;
    --beta)
      BETA=$2; shift 2 ;;
    --lora-r)
      LORA_R=$2; shift 2 ;;
    --sweep-name)
      SWEEP_NAME=$2; shift 2 ;;
    --run-name)
      RUN_NAME=$2; shift 2 ;;
    --job-name)
      JOB_NAME=$2; shift 2 ;;
    --hf-model)
      HF_MODEL=$2; shift 2 ;;
    --eval-mode)
      EVAL_MODE=$2; shift 2 ;;
    --scratch-root)
      SCRATCH_ROOT=$2; shift 2 ;;
    --results-root)
      RESULTS_ROOT=$2; shift 2 ;;
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

if [[ -z $MODEL || -z $BETA || -z $LORA_R ]]; then
  echo "Missing required argument" >&2
  usage
  exit 1
fi

case $MODEL in
  google-gemma-3-1b-it|Qwen-Qwen2.5-1.5B-Instruct|Qwen-Qwen2.5-3B-Instruct|meta-llama-Llama-3.2-1B-Instruct|meta-llama-Llama-3.2-3B-Instruct) ;;
  *) echo "Invalid model: $MODEL" >&2; exit 1 ;;
esac

case $EVAL_MODE in
  CLI|NONE) ;;
  *) echo "Invalid eval mode: $EVAL_MODE" >&2; exit 1 ;;
esac

sanitize() {
  echo "$1" | tr -c 'A-Za-z0-9' '_'
}

if [[ -z $SWEEP_NAME ]]; then
  SWEEP_NAME="results_run_$(date -u +%Y%m%d-%H%M%S)"
fi

if [[ -z $RUN_NAME ]]; then
  beta_tag=$(sanitize "$BETA")
  RUN_NAME="grpo_${MODEL}_b${beta_tag}_r${LORA_R}"
fi

if [[ -z $JOB_NAME ]]; then
  beta_tag=$(sanitize "$BETA")
  JOB_NAME="grpo-${MODEL}-b${beta_tag}-r${LORA_R}"
fi

mkdir -p "$RESULTS_ROOT/$SWEEP_NAME"
LOG_DIR="$REPO_ROOT/logs/$SWEEP_NAME"
mkdir -p "$LOG_DIR"

EXPORT_VARS=(
  "MODEL_KEY=$MODEL"
  "RUN_NAME=$RUN_NAME"
  "SWEEP_NAME=$SWEEP_NAME"
  "GRPO_BETA=$BETA"
  "LORA_R=$LORA_R"
  "SCRATCH_ROOT=$SCRATCH_ROOT"
  "HOME_REPO=$REPO_ROOT"
  "RESULTS_ROOT=$RESULTS_ROOT"
  "EVAL_MODE=$EVAL_MODE"
)

if [[ -n $HF_MODEL ]]; then
  EXPORT_VARS+=("HF_MODEL_NAME=$HF_MODEL")
fi

LORA_ALPHA=$LORA_R
EXPORT_VARS+=("LORA_ALPHA=$LORA_ALPHA")

OLD_IFS=$IFS
IFS=,
EXPORT_ARG="ALL,${EXPORT_VARS[*]}"
IFS=$OLD_IFS

CMD=(sbatch
  --job-name="$JOB_NAME"
  --output="$LOG_DIR/${RUN_NAME}-%j.out"
  --error="$LOG_DIR/${RUN_NAME}-%j.err"
  --chdir="$REPO_ROOT"
  --export="$EXPORT_ARG"
  "$REPO_ROOT/bin/grpo_train_job.sh")

printf '[submit_grpo_run] sweep=%s model=%s beta=%s lora_r=%s run=%s
'   "$SWEEP_NAME" "$MODEL" "$BETA" "$LORA_R" "$RUN_NAME"

if [[ $DRY_RUN -eq 1 ]]; then
  printf '[submit_grpo_run] dry run: %q ' "${CMD[@]}"
  printf '
'
  exit 0
fi

output=$("${CMD[@]}")
echo "$output"
if [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
  echo "[submit_grpo_run] job_id=${BASH_REMATCH[1]}"
fi
