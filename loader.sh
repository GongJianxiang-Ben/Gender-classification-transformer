#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=360
#SBATCH --job-name=model_test
#SBATCH --output=output_%j.out
#SBATCH --error=output_%j.err

module load cuda/12.9          
module load anaconda

eval "$(conda shell.bash hook)"
conda activate deeplearning

python Vit-model-loader.py \
  --img_dir ./aligned \
  --attr_folder ./"label txt" \
  --dataset adience\
  --start ./checkpoints/Vit-Adience-post-train.ckpt\
  --epochs 50 \
  --batch_size 64 