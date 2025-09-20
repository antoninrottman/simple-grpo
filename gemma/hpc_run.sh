#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name=run1_no_klgemma
#SBATCH --output=_run1_no_kl_gemma-stdout.txt
#SBATCH --error=_run1_no_kl_gemma-stderr.txt
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=180G

module load gcc/11.3.0
module load cuda/11.8.0
source /home/rottman/simple-grpo/.venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

export MODEL_NAME="google/gemma-3-1b-it"
export OUTPUT_DIR="./outputs/run1_no_kl_gemma_gsm8k"
export RUN_NAME="run1_no_kl_gemma_gsm8k"
export EVAL_OUTPUT_DIR="${OUTPUT_DIR}/evaluation_results"
export MERGED_DIR="${OUTPUT_DIR}/merged_model"
export EVAL="CLI" # API or CLI or NONE

mkdir -p $EVAL_OUTPUT_DIR
mkdir -p $MERGED_DIR

# for multi-gpu training (not recommended with LoRA+GRPO)
#accelerate launch --num_processes=2 train_grpo.py

# single GPU training
python train_grpo.py

if [[ "${EVAL:-}" == "CLI" ]]; then
    echo "Running evaluation with CLI"
    bash ../bin/run_eval.sh "$MERGED_DIR" "$EVAL_OUTPUT_DIR"
    exit 0
fi

