#!/bin/bash
#SBATCH --job-name=mon_train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --time=24:00:00        # 24h
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1           # 1 GPU
#SBATCH --partition=gpu        # partition GPU (varie selon cluster)

module load Python/3.11.3
module load CUDA/12.1

source activate mon_env          # si conda

python train_on_ceci.py