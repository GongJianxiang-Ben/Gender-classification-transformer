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

python train_vit.py \
  --img_dir ./celeba_data/img_align_celeba \
  --attr_file ./celeba_data/list_attr_celeba.csv \
  --split_file ./celeba_data/list_eval_partition.csv \
  --checkpoint ./checkpoints/posttrain_on_adience \
  --dataset adience\
  --start ./checkpoints/posttrain_on_adience/vit-best-47-0.0000.ckpt\
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-6 