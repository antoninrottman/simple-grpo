#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name=test_eval
#SBATCH --output=_test_eval-stdout.txt
#SBATCH --error=_test_eval-stderr.txt
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

export MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
export OUTPUT_DIR="./outputs/test_eval_gsm8k"
export RUN_NAME="test_eval_gsm8k"

export EVAL_OUTPUT_DIR="${OUTPUT_DIR}/evaluation_results"
export BENCHMARKS="lighteval|gsm8k|0|0"  # -> suite|task|few_shot|truncate_few_shots
export MAX_EVAL_SAMPLES=100

# for multi-gpu training (not recommended with LoRA+GRPO)
#accelerate launch --num_processes=2 train_grpo.py

# single GPU training
python train_grpo.py
