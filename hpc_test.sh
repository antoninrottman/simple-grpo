#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name=ft_lora_kl
#SBATCH --output=_scheduler-stdout.txt
#SBATCH --error=_scheduler-stderr.txt
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=180G

module load gcc/11.3.0
module load cuda/11.8.0
source .venv/bin/activate

python main.py