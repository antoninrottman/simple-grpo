#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name=500_llama_eval
#SBATCH --output=_500_llama-stdout.txt
#SBATCH --error=_500_llama-stderr.txt
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

export MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
export OUTPUT_DIR="./outputs/500_llama_gsm8k"
export RUN_NAME="500_llama_gsm8k"
export EVAL_OUTPUT_DIR="${OUTPUT_DIR}/evaluation_results"

# for multi-gpu training (not recommended with LoRA+GRPO)
#accelerate launch --num_processes=2 train_grpo.py

# single GPU training
python train_grpo.py
# python eval_grpo.py

