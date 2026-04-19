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

python ./"gender classification"/Vit_model_loader.py \
  --img_dir ./crop_part1\
  --attr_folder ./"label txt" \
  --dataset UTK\
  --start ./checkpoints-768/addCNN/vit-best-08-0.0000.ckpt\
  --model_name addCNN\
  --backbone online\
  --address rizvandwiki/gender-classification\
  --batch_size 64 