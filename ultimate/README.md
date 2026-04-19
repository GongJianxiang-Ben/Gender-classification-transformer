# Ultimate CNN and ViT for Adience Age and Gender Classification

This repository contains the final training pipeline for our ultimate CNN and ultimate ViT models on the Adience benchmark.

## Overview

We implement two final models:

### 1. Ultimate CNN
- Backbone: `ResNet-18`
- Initialization: `torchvision` ImageNet-1K pretrained weights
- Structural change: `Deformable Convolution`
- Task setting: `Multi-task learning`
- Outputs:
  - Gender classification
  - Age-group classification
- Multi-task loss weight: `lambda = 0.5`
- Head: simple linear task heads

### 2. Ultimate ViT
- Backbone: `ViT-B/16`
- Initialization: `torchvision` ImageNet-1K pretrained weights
- Structural change: `VPT-Deep`
- Prompt length: `K = 100`
- Task setting: `Multi-task learning`
- Outputs:
  - Gender classification
  - Age-group classification
- Multi-task loss weight: `lambda = 0.5`
- Head: simple linear task heads

Both models are evaluated using the standard Adience 5-fold cross-validation protocol.

## Dataset

This code assumes the Adience dataset is arranged as follows:

```text
dlj/
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
└── ultimate/
    └── train_ultimate.py
```

The fold files must follow the original Adience 5-fold split. All images of the same identity should remain in the same fold.

## Environment

Recommended environment:
- Python 3.10
- PyTorch 2.2.x
- torchvision 0.17.x
- Pillow

Example packages:

```bash
pip install torch torchvision pillow
```

If you use the original environment, make sure the following libraries are available:
- `torch`
- `torchvision`
- `PIL`

## Training

### Run Ultimate CNN

From the `dlj` directory:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_HOME=/path/to/dlj/ultimate/.torch-cache \
python ultimate/train_ultimate.py --arch resnet --epochs 50
```

### Run Ultimate ViT

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_HOME=/path/to/dlj/ultimate/.torch-cache \
python ultimate/train_ultimate.py --arch vit --epochs 50
```

## Multi-GPU Training for ViT

This code supports single-model multi-GPU training with DDP.

```bash
CUDA_VISIBLE_DEVICES=0,1 TORCH_HOME=/path/to/dlj/ultimate/.torch-cache \
torchrun --nproc_per_node=2 ultimate/train_ultimate.py --arch vit --epochs 50 --ddp
```

## Useful Arguments

```text
--arch
--epochs
--batch_size
--num_workers
--lam
--age_sigma
--lr_backbone
--lr_head
--vpt_k
--prompt_dropout
--vit_unfreeze_backbone
--ddp
```

## Outputs

Training outputs are saved to:

```text
ultimate/outputs/
├── resnet/
│   ├── fold_0.pt
│   ├── fold_1.pt
│   ├── fold_2.pt
│   ├── fold_3.pt
│   └── fold_4.pt
├── vit/
│   ├── fold_0.pt
│   ├── fold_1.pt
│   ├── fold_2.pt
│   ├── fold_3.pt
│   └── fold_4.pt
├── resnet_5fold_metrics.json
└── vit_5fold_metrics.json
```

Each fold checkpoint stores:
- model weights
- fold id
- epoch
- evaluation metrics
- run config

## Evaluation Protocol

We follow the standard Adience evaluation:
- 5-fold cross-validation
- Gender accuracy
- Age exact accuracy
- Age 1-off accuracy
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

## Notes

- Both CNN and ViT use pretrained weights from the same source: `ImageNet-1K`
- CNN uses `ResNet-18`, not ResNet-50
- ViT uses `VPT-Deep (K=100)`
- The final heads are intentionally simple because earlier experiments showed that more complex heads did not provide consistent gains

## Example

From `~/Code/dlj`:

```bash
mkdir -p ultimate/logs ultimate/.torch-cache
CUDA_VISIBLE_DEVICES=0 TORCH_HOME=~/Code/dlj/ultimate/.torch-cache \
nohup python ultimate/train_ultimate.py --arch resnet --epochs 50 \
> ultimate/logs/train_resnet18_ultimate.log 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=1 TORCH_HOME=~/Code/dlj/ultimate/.torch-cache \
nohup python ultimate/train_ultimate.py --arch vit --epochs 50 \
> ultimate/logs/train_vit_ultimate.log 2>&1 &
```

## Contact

If you use this code, please make sure your Adience dataset paths and fold files are prepared correctly before training.
