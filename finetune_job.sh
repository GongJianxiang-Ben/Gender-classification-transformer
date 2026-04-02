#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=4
#SBATCH --time=06:00:00
#SBATCH --job-name=resnet18_finetune
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -e
cd ~/resnet18_project

module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate resnet_v100

unset LD_PRELOAD
export LD_LIBRARY_PATH=""

echo "Python: $(which python)"
python -V
nvidia-smi

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"

python finetune.py \
  --data_dir ./audience_data/organized \
  --num_classes 2 \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-4 \
  --output finetuned_resnet18_adience.pth