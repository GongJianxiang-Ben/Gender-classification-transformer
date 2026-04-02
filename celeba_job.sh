#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=4
#SBATCH --time=06:00:00
#SBATCH --job-name=celeba_finetune
#SBATCH --output=output_celeba_%j.out
#SBATCH --error=error_celeba_%j.err

set -e
cd ~/resnet18_project

module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate resnet_v100

unset LD_PRELOAD
export LD_LIBRARY_PATH=""

python celeba_finetune.py \
  --img_dir ./celeba_data/img_align_celeba/img_align_celeba \
  --attr_file ./celeba_data/list_attr_celeba.csv \
  --split_file ./celeba_data/list_eval_partition.csv \
  --checkpoint ./finetuned_resnet18_adience.pth \
  --epochs 10 \
  --batch_size 64 \
  --lr 1e-4 \
  --output celeba_finetuned.pth