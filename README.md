# ResNet18 Gender Classification
## Pre-trained on CelebA, Fine-tuned on Adience

### Pipeline
1. Train ResNet18 from scratch on CelebA (gender classification)
2. Fine-tune on Adience dataset using 2-phase training

---

### Model Architecture
- **Model**: ResNet18 (built from scratch using PyTorch)
- **Input size**: 224 × 224 × 3
- **Output classes**: 2 (Female, Male)
- **GPU**: NVIDIA Tesla V100-PCIE-32GB
- **PyTorch**: 2.0.1 | **CUDA**: 11.7

---

### Stage 1 — Pre-training on CelebA

| Parameter | Value |
|---|---|
| Dataset | CelebA (Gender attribute) |
| Train samples | 162,770 |
| Val samples | 19,867 |
| Test samples | 19,962 |
| Epochs | 10 |
| Batch size | 64 |
| Learning rate | 1e-4 |
| Optimizer | Adam (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |

**Results:**
| Metric | Value |
|---|---|
| Best Val Accuracy | 98.79% |
| Test Accuracy | 98.41% |
| Saved model | `celeba_scratch.pth` |

---

### Stage 2 — Fine-tuning on Adience (Fold-based)

**Data Split:**
- Folds 0, 1, 2 → Training (10,741 images)
- Fold 3 → Validation (3,306 images)
- Fold 4 → Test / Held-out (3,445 images)

**2-Phase Training:**

| Phase | Epochs | LR | Layers Trained |
|---|---|---|---|
| Phase 1 | 5 | 1e-3 | FC head only (backbone frozen) |
| Phase 2 | 15 | 1e-5 | All layers unfrozen |

| Parameter | Value |
|---|---|
| Batch size | 32 |
| Optimizer | Adam (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Pretrained from | CelebA checkpoint |

**Results:**
| Metric | Value |
|---|---|
| Best Val Accuracy | 76.95% |
| Test Accuracy (fold 4) | 71.67% |
| Saved model | `adience_folds_finetuned.pth` |

---

### Data Augmentation (Training)
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness, contrast, saturation ±0.2)
- Random grayscale (p=0.1)
- Normalize (ImageNet mean/std)

---

### Files
| File | Description |
|---|---|
| `resnet18.py` | ResNet18 architecture from scratch |
| `training_utils.py` | Training utility functions |
| `celeba_finetune.py` | CelebA training script |
| `adience_finetune_folds.py` | Adience fold-based fine-tuning script |
| `celeba_scratch.pth` | CelebA pretrained weights |
| `adience_folds_finetuned.pth` | Final Adience fine-tuned weights |

---

### How to Run

**Stage 1 - Train on CelebA:**
```bash
python celeba_finetune.py \
  --img_dir ./celeba_data/img_align_celeba/img_align_celeba \
  --attr_file ./celeba_data/list_attr_celeba.csv \
  --split_file ./celeba_data/list_eval_partition.csv \
  --epochs 10 \
  --batch_size 64 \
  --lr 1e-4 \
  --output celeba_scratch.pth
```

**Stage 2 - Fine-tune on Adience:**
```bash
python adience_finetune_folds.py \
  --data_dir ./audience_data/AdienceBenchmarkGenderAndAgeClassification \
  --checkpoint ./celeba_scratch.pth \
  --phase1_epochs 5 \
  --phase2_epochs 15 \
  --batch_size 32 \
  --lr_phase1 1e-3 \
  --lr_phase2 1e-5 \
  --output adience_folds_finetuned.pth
```