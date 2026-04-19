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

## ViT

### Running tests

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

#### Required arguments for testing

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

#### Optional testing arguments

- `--batch_size`: default `64`
- `--seed`: default `42`
- `--num_workers`: default `4`

### Training the model

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

#### Required arguments for training

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

#### Optional training arguments

- `--import_from`:
  - if `--start local`: path to local checkpoint
  - if `--start online`: Hugging Face model ID
- `--epochs`: default `50`
- `--dilated_size`: dilation size for `dilatedConv`, default `2`
- `--lr`: learning rate, default `1e-4`
- `--seed`: default `42`
- `--num_workers`: default `4`

### ResNet-18


### Multi-task ViT

> Placeholder for multi-task ViT model usage and instructions.


# ResNet-18 Gender Classification
## SC4001 Neural Networks and Deep Learning

### Overview
ResNet-18 CNN contributions for gender classification.
Pipeline: CelebA pre-training → Adience fine-tuning → Architectural modifications → Multi-task learning → UTKFace evaluation.

### Results

#### Pre-training Comparison
| Model | Adience Test Acc (Fold 0) | F1 | UTKFace Acc |
|---|---|---|---|
| Scratch | 93.47% | 0.9361 | 76.92% |
| CelebA pretrained | 93.52% | 0.9365 | 76.52% |
| ImageNet + Combined | N/A* | — | 83.59% |

#### Architectural Modifications
| Model | Adience Test Acc | F1 | UTKFace Acc |
|---|---|---|---|
| Vanilla | 93.52% | 0.9368 | 76.78% |
| Dilated Conv (d=2) | 92.67% | 0.9279 | 75.32% |
| Deformable Conv | 94.12% | 0.9431 | 78.36% |

### File Descriptions
| File | Description |
|---|---|
| `resnet18.py` | ResNet-18 built from scratch |
| `resnet18_dilated.py` | Dilated conv variant (stages 3-4, d=2) |
| `resnet18_deformable.py` | Deformable conv variant (stages 3-4) |
| `celeba_finetune.py` | CelebA gender pre-training |
| `adience_finetune_folds_v2.py` | Adience fine-tuning (train folds 1-4, test fold 0) |
| `train_variants_v2.py` | Train dilated/deformable on Adience |
| `train_combined.py` | ImageNet pretrained + Adience+CelebA combined |
| `multitask_train.py` | Multi-task gender + age with λ ablation |
| `eval_fold0.py` | Evaluate on Adience fold 0 + UTKFace aligned |
| `utk_data_loader.py` | UTKFace PyTorch dataset loader |
| `gradcam.py` | Grad-CAM heatmap visualization |
| `prepare_data.py` | Prepare Adience data into organized folders |

### How to Run
> **TODO: Modify all paths before running**

```bash
# 1. Pre-train on CelebA
python celeba_finetune.py \
  --img_dir ./celeba_data/img_align_celeba/img_align_celeba \
  --attr_file ./celeba_data/list_attr_celeba.csv \
  --split_file ./celeba_data/list_eval_partition.csv \
  --output celeba_scratch.pth

# 2. Fine-tune on Adience
python adience_finetune_folds_v2.py \
  --img_dir ./audience_data/AdienceBenchmarkGenderAndAgeClassification/faces \
  --label_dir ./audience_data/AdienceBenchmarkGenderAndAgeClassification \
  --checkpoint ./celeba_scratch.pth \
  --output adience_celeba_fold0.pth

# 3. Train variants
python train_variants_v2.py \
  --img_dir ./audience_data/AdienceBenchmarkGenderAndAgeClassification/faces \
  --label_dir ./audience_data/AdienceBenchmarkGenderAndAgeClassification \
  --checkpoint ./celeba_scratch.pth \
  --variant dilated \
  --output adience_dilated_v2.pth

# 4. Evaluate
python eval_fold0.py \
  --adience_dir ./audience_data/AdienceBenchmarkGenderAndAgeClassification \
  --utk_dir ./utkface_data/utkface_aligned_cropped/UTKFace \
  --checkpoint ./adience_celeba_fold0.pth \
  --label "ResNet18_CelebA"
```