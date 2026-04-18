# Gender-classification-transformer

## Overview

This repository contains scripts for gender classification using Vision CNN and Transformer (ViT) models.

## Environment setup

The code was tested with:

- PyTorch 2.1.2
- CUDA 11.8

Install the required packages in your Python environment:

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install lightning
pip install torchinfo
pip install "transformers<4.40"
pip install torchmetrics
pip install pillow
pip install "numpy<2"
pip install scikit-learn
```

## Running tests

The test script evaluates the model using a saved checkpoint.

```bash
python ./"gender classification"/Vit/Vit_model_loader.py \
  --img_dir ./crop_part1 \
  --attr_folder ./"label txt" \
  --dataset UTK \
  --start ./checkpoints-768/addCNN/vit-best-08-0.0000.ckpt \
  --model_name addCNN \
  --backbone online \
  --address rizvandwiki/gender-classification \
  --batch_size 64
```

Or submit the batch job:

```bash
sbatch train.sh
```

### Required arguments for testing

- `--img_dir`: path to the image folder
- `--attr_folder`: path to the attribute folder; use `./` if not needed
- `--dataset`: dataset name: `UTK`, `adience`, or `celebA`
- `--start`: path to the checkpoint file to test
- `--model_name`: model type, e.g. `ViT`, `deformConv`, `dilatedConv`, `addCNN`, `VPTshallow`, `VPTdeep`, `depth`, `online`
- `--backbone`: backbone source for ViT only: `online`, `local`, or `None`
- `--address`:
  - `online`: Hugging Face model ID
  - `local`: local checkpoint path
  - `None`: not used

### Optional testing arguments

- `--batch_size`: default `64`
- `--seed`: default `42`
- `--num_workers`: default `4`

## Training the model

The training script uses a dataset split and saves checkpoints.

```bash
python ./"gender classification"/Vit/Vit_deformConv.py \
  --img_dir ./celeba_data/img_align_celeba \
  --attr_file ./celeba_data/list_attr_celeba.csv \
  --split_file ./celeba_data/list_eval_partition.csv \
  --checkpoint ./checkpoints-768/addCNN \
  --dataset both \
  --start online \
  --import_from rizvandwiki/gender-classification \
  --epochs 50 \
  --dilated_size 3 \
  --batch_size 64 \
  --lr 5e-4
```

Or submit the batch job:

```bash
sbatch loader.sh
```

### Required arguments for training

- `--img_dir`: path to the image folder
- `--attr_folder`: path to the attribute folder; use `./` if not needed
- `--split_file`: path to the dataset split file; use `./` if not needed
- `--checkpoint`: directory to save checkpoints
- `--dataset`: dataset type: `both`, `adience`, or `celebA`
- `--start`:
  - `scratch`: train from scratch
  - `online`: use online pretrained backbone
  - `local`: use a local backbone checkpoint (ViT only)
  - path to a checkpoint file: continue training from that checkpoint

### Optional training arguments

- `--import_from`:
  - if `--start local`: path to local checkpoint
  - if `--start online`: Hugging Face model ID
- `--epochs`: default `50`
- `--dilated_size`: dilation size for `dilatedConv`, default `2`
- `--lr`: learning rate, default `1e-4`
- `--seed`: default `42`
- `--num_workers`: default `4`
