# Gender Classification Transformer

This repository contains several stages of experiments for age and gender classification, including CNN- and ViT-based models, head-comparison experiments, multitask baselines, and the final ultimate models on the Adience benchmark.

## Overview

The repository currently includes:

- `resnet/`: earlier ResNet-18 CNN experiments and architectural variants
- `head/`: single-task head-architecture comparison for CNN and ViT
- `Multitask/`: age and gender multi-task learning baselines
- `Vit/`: earlier ViT experiments and model variants
- `ultimate/`: final CNN and ViT models used as the ultimate versions

## Repository Structure

```text
repo_root/
├── Dataset/
│   ├── aligned/
│   ├── fold_0_data.txt
│   ├── fold_1_data.txt
│   ├── fold_2_data.txt
│   ├── fold_3_data.txt
│   └── fold_4_data.txt
├── resnet/
├── Multitask/
│   ├── resnet_multi.py
│   └── vit_multi.py
├── Vit/
├── head/
│   ├── cnn/
│   └── vit/
├── ultimate/
│   └── train_ultimate.py
├── checkpoints/
├── README.md
└── environment.yml
```

## Dataset Layout

All Adience-related code assumes the dataset is prepared under:

```text
repo_root/
├── Dataset/
│   ├── aligned/
│   │   ├── user_id_1/
│   │   ├── user_id_2/
│   │   └── ...
│   ├── fold_0_data.txt
│   ├── fold_1_data.txt
│   ├── fold_2_data.txt
│   ├── fold_3_data.txt
│   └── fold_4_data.txt
└── ...
```

Important notes:
- `Dataset/aligned/` must contain aligned face images.
- `fold_0_data.txt` to `fold_4_data.txt` should follow the Adience 5-fold split.
- All images of the same identity must remain in the same fold.

## Environment Setup

The code was tested with:
- Python 3.10
- PyTorch 2.2.x
- torchvision 0.17.x
- CUDA 11.8

Recommended dependencies:
- `torch`
- `torchvision`
- `pillow`
- `transformers`
- `pytorch-lightning`
- `torchmetrics`
- `numpy`
- `scikit-learn`

If available, use:

```bash
conda env create -f environment.yml
conda activate sc4001
```

Or install manually:

```bash
pip install torch torchvision pillow
pip install lightning
pip install torchinfo
pip install "transformers<4.40"
pip install torchmetrics
pip install "numpy<2"
pip install scikit-learn
```

## Run from the Correct Directory

All commands below assume you are in the repository root:

```bash
cd /path/to/Gender-classification-transformer
```

This is important because the repository root is the actual project root on GitHub.

---
## 1. ResNet Legacy Experiments

The `resnet/` folder contains earlier ResNet-18 CNN experiments for gender classification, including pre-training, Adience fine-tuning, architectural variants, multitask learning, and cross-dataset evaluation.

### Overview

Pipeline:
CelebA pre-training -> Adience fine-tuning -> Architectural modifications -> Multi-task learning -> UTKFace evaluation


### Main Files

| File | Description |
|---|---|
| `resnet18.py` | ResNet-18 built from scratch |
| `resnet18_dilated.py` | Dilated convolution variant |
| `resnet18_deformable.py` | Deformable convolution variant |
| `celeba_finetune.py` | CelebA gender pre-training |
| `adience_finetune_folds_v2.py` | Adience fine-tuning |
| `train_variants_v2.py` | Train dilated/deformable variants |
| `train_combined.py` | ImageNet pretrained + Adience + CelebA combined training |
| `multitask_train.py` | Multi-task gender + age training |
| `eval_fold0.py` | Evaluation on Adience fold 0 and UTKFace |
| `utk_data_loader.py` | UTKFace dataset loader |
| `gradcam.py` | Grad-CAM visualization |
| `prepare_data.py` | Adience data preparation |

### Example Usage

> Modify dataset and checkpoint paths before running.
>Usually takes about 15 mins to see the first training epoch result
```bash
python resnet/celeba_finetune.py \
  --img_dir ./celeba_data/img_align_celeba/img_align_celeba \
  --attr_file ./celeba_data/list_attr_celeba.csv \
  --split_file ./celeba_data/list_eval_partition.csv \
  --output celeba_scratch.pth
```

```bash
python resnet/adience_finetune_folds_v2.py \
  --img_dir ./audience_data/AdienceBenchmarkGenderAndAgeClassification/faces \
  --label_dir ./audience_data/AdienceBenchmarkGenderAndAgeClassification \
  --checkpoint ./celeba_scratch.pth \
  --output adience_celeba_fold0.pth
```

```bash
python resnet/train_variants_v2.py \
  --img_dir ./audience_data/AdienceBenchmarkGenderAndAgeClassification/faces \
  --label_dir ./audience_data/AdienceBenchmarkGenderAndAgeClassification \
  --checkpoint ./celeba_scratch.pth \
  --variant dilated \
  --output adience_dilated_v2.pth
```

```bash
python resnet/eval_fold0.py \
  --adience_dir ./audience_data/AdienceBenchmarkGenderAndAgeClassification \
  --utk_dir ./utkface_data/utkface_aligned_cropped/UTKFace \
  --checkpoint ./adience_celeba_fold0.pth \
  --label "ResNet18_CelebA"
```

---

## 2. ViT Legacy Experiments

The `Vit/` folder contains earlier ViT experiments and model variants.

### Main Files

| File | Description |
|---|---|
| `Vit.py` | Vit built from scratch |
| `Vit_dilatedConv.py` | Dilated convolution embedding Vit|
| `Vit_deformableConv.py` | Deformable convolution embedding Vit |
| `Vit_depth.py` | Add depth to backbone Vit model for post-train |
| `Vit_addCNN.py` | Add CNN layers before transformer layers |
| `Vit_VPTshallow.py` | Perform VPT shallow version post-train |
| `Vit_VPTdeep.py` | Perform VPT deep version post-train  |
| `Vit_model_loader` | load model and test performance |
| `utk_data_loader.py` | UTKFace dataset loader |
| `adience_data_loader.py` | Adience dataset loader |
| `celebA_data_loader.py` | CelebA dataset loader |
| `train.sh` | bash file to submit training |
| `test.sh` | bash file to submit testing |


### Training the Model

```bash
python Vit/Vit_deformConv.py \
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
sbatch test.sh
```

Required arguments for training:
- `--img_dir`: path to the image folder
- `--attr_folder`: path to the attribute folder; use `./` if not needed
- `--split_file`: dataset split file
- `--checkpoint`: directory to save checkpoints
- `--dataset`: `both`, `adience`, or `celebA`
- `--start`:
  - `scratch`
  - `online`
  - `local`
  - or a checkpoint path to resume training

Optional training arguments:
- `--import_from`:
  - if `--start local`: local checkpoint path
  - if `--start online`: Hugging Face model ID
- `--epochs`: default `50`
- `--dilated_size`: default `2`
- `--lr`: default `1e-4`
- `--seed`: default `42`
- `--num_workers`: default `4`


### Running Tests

The test script evaluates a model using a saved checkpoint.

```bash
python Vit/Vit_model_loader.py \
  --img_dir ./crop_part1 \
  --attr_folder "./label txt" \
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

Required arguments for testing:
- `--img_dir`: path to the image folder
- `--attr_folder`: path to the attribute folder; use `./` if not needed
- `--dataset`: `UTK`, `adience`, or `celebA`
- `--start`: path to checkpoint file
- `--model_name`: e.g. `ViT`, `deformConv`, `dilatedConv`, `addCNN`, `VPTshallow`, `VPTdeep`, `depth`, `online`
- `--backbone`: `online`, `local`, or `None`
- `--address`:
  - `online`: Hugging Face model ID
  - `local`: local checkpoint path
  - `None`: not used

Optional testing arguments:
- `--batch_size`: default `64`
- `--seed`: default `42`
- `--num_workers`: default `4`

---

## 3. Head Experiments

This folder is used for comparing different classification heads.

Because the repository structure was reorganized and `head/` is now one level deeper than before, the `head/` scripts should be run with an explicit `--data_dir` argument.

This ensures they correctly resolve:
- `Dataset/aligned`
- `Dataset/fold_*.txt`
- `checkpoint/`
- `log/`

### CNN Heads

Simple head:

```bash
python head/cnn/resnet_simple.py \
  --data_dir . \
  --checkpoint ./checkpoint/celeba_scratch.pth
```

Complex head:

```bash
python head/cnn/resnet_complex.py \
  --data_dir . \
  --checkpoint ./checkpoint/celeba_scratch.pth
```

Complex BN + Dropout head:

```bash
python head/cnn/resnet_complex_bn_dropout.py \
  --data_dir . \
  --checkpoint ./checkpoint/celeba_scratch.pth
```

### ViT Heads

Wait for about 5 mins to get first epoch output

Simple head:

```bash
python head/vit/vit_simple.py \
  --data_dir . \
  --checkpoint ./checkpoint/Vit-CelebA-pretrain.ckpt
```

Complex head:

```bash
python head/vit/vit_complex.py \
  --data_dir . \
  --checkpoint ./checkpoints/Vit-CelebA-pretrain.ckpt
```

Complex BN + Dropout head:

```bash
python head/vit/vit_complex_bn_dropout.py \
  --data_dir . \
  --checkpoint ./checkpoint/Vit-CelebA-pretrain.ckpt
```

With `--data_dir .`, these scripts resolve:
- images from `./Dataset/aligned`
- fold files from `./Dataset/`
- checkpoints from `./checkpoint/`
- logs under `./log/`

`--data_dir .` can also have personal path to parent folder of adience dataset (Dataset) with structure metioned in Dataset Layout.

---

## 4. Multitask Experiments

This folder contains the multitask age + gender baselines.

### CNN Multitask

```bash
python Multitask/resnet_multi.py --lam 0.5\
 --data_dir .\
 --checkpoint ./checkpoints/celeba_scratch.pth
```

### ViT Multitask

```bash
python Multitask/vit_multi.py --lam 0.5\
 --data_dir .\
 --checkpoint ./checkpoints/Vit-CelebA-pretrain.ckpt
```

Common useful arguments:

```text
--lam
--epochs
--batch_size
--num_workers
--checkpoint
--img_dir
--label_dir
```

Examples:

```bash
python Multitask/resnet_multi.py --lam 0.5
python Multitask/vit_multi.py --lam 0.5 -epochs 50
```

---

## 5. Ultimate Experiments

>Need to wait for about 5 mins to see first epoch result

This folder contains the final models. The arg `--data_dir` points to the path of adience dataset(Dataset) itself

### Ultimate CNN

- Backbone: ResNet-18
- Pretraining: torchvision ImageNet-1K
- Modification: Deformable Convolution
- Setting: multitask
- Lambda: 0.5
- Head: simple linear heads

Run:

```bash
python ultimate/train_ultimate.py --arch resnet --epochs 50\
 --data_dir ./Dataset
```

### Ultimate ViT

- Backbone: ViT-B/16
- Pretraining: torchvision ImageNet-1K
- Modification: VPT-Deep
- Prompt length: `K = 100`
- Setting: multitask
- Lambda: 0.5
- Head: simple linear heads

Run:

```bash
python ultimate/train_ultimate.py --arch vit --epochs 50\
 --data_dir ./Dataset
```

### Ultimate ViT with 2-GPU DDP

```bash
torchrun --nproc_per_node=2 ultimate/train_ultimate.py --arch vit --epochs 50 \
    --ddp\
     --data_dir ./Dataset
```

---

## Background Training

ResNet Legacy example:

```bash
mkdir -p log/cnn \
nohup python resnet/celeba_finetune.py \
  --img_dir ./celeba_data/img_align_celeba/img_align_celeba \
  --attr_file ./celeba_data/list_attr_celeba.csv \
  --split_file ./celeba_data/list_eval_partition.csv \
  --output ./checkpoints/resnet18/celeba_scratch.pth \
> log/cnn/finetune_resnet18.logs 2>&1 &
```


ViT Legacy example:

```bash
mkdir -p log/ViT
nohup python Vit/Vit_deformConv.py \
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
  --lr 5e-4\
> log/ViT/train_vit_deofromConv.log 2>&1 &
```

Head CNN example:

```bash
mkdir -p log/cnn
nohup python head/cnn/resnet_simple.py \
  --data_dir . \
  --checkpoint checkpoint/cnn/celeba_scratch.pth \
  > log/cnn/resnet_simple_manual.log 2>&1 &
```

Multitask CNN example:

```bash
mkdir -p log/cnn
nohup python Multitask/resnet_multi.py --lam 0.5 \
  > log/cnn/resnet_multitask_lam0.5_manual.log 2>&1 &
```

Ultimate CNN example:

```bash
mkdir -p ultimate/logs ultimate/.torch-cache
TORCH_HOME=./ultimate/.torch-cache \
nohup python ultimate/train_ultimate.py --arch resnet --epochs 50 \
> ultimate/logs/train_resnet18_ultimate.log 2>&1 &
```

Ultimate ViT example:

```bash
mkdir -p ultimate/logs ultimate/.torch-cache
TORCH_HOME=./ultimate/.torch-cache \
nohup python ultimate/train_ultimate.py --arch vit --epochs 50 \
> ultimate/logs/train_vit_ultimate.log 2>&1 &
```

---

## Outputs

### ResNet Legacy
- checkpoints under `./checkpoints/resnet18`
- logs under `log/cnn`

### ViT Legacy
- checkpoints under `./checkpoints-768`
- logs under `log/ViT`


### head
- checkpoints saved under `checkpoint/cnn/` or `checkpoint/vit/`
- logs saved under `log/cnn/` or `log/vit/`

### Multitask
- CNN checkpoints under `checkpoint/cnn/`
- ViT checkpoints under `checkpoint/vit/`
- logs under `log/cnn/` and `log/vit/`

### ultimate
- checkpoints under `ultimate/outputs/`
- metrics under:
  - `ultimate/outputs/resnet_5fold_metrics.json`
  - `ultimate/outputs/vit_5fold_metrics.json`

---

## Evaluation Protocol

All main Adience experiments follow the 5-fold protocol.

Metrics:
- Gender Accuracy
- Gender F1
- Age Exact Accuracy
- Age 1-off Accuracy
- Age MAE

Age groups:
- `0-2`
- `4-6`
- `8-13`
- `15-20`
- `25-32`
- `38-43`
- `48-53`
- `60+`

---

## Notes

- `head/` is for head-comparison experiments
- `Multitask/` is for multitask baselines
- `ultimate/` is the final recommended version
- `Vit/` contains earlier ViT experiments and variants
- CNN and ViT ultimate models both use pretrained weights from the same source: ImageNet-1K
