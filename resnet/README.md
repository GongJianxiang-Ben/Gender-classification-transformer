markdown# ResNet-18 Gender Classification
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