## ResNet18 Gender Classification - Fine-tuning Results

### Model Architecture
- **Model**: ResNet18 (built from scratch using PyTorch)
- **Input size**: 224 × 224 × 3
- **Output classes**: 2 (Female, Male)

### Training Hardware
- **GPU**: NVIDIA Tesla V100-PCIE-32GB
- **CUDA**: 11.7
- **PyTorch**: 2.0.1

---

### Stage 1 — Fine-tuned on Adience Dataset

| Parameter | Value |
|---|---|
| Dataset | Adience Benchmark Gender Classification |
| Train samples | 10,803 |
| Val samples | 2,700 |
| Epochs | 30 |
| Batch size | 32 |
| Learning rate | 1e-4 |
| Optimizer | Adam (weight decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss |

**Results:**
| Metric | Value |
|---|---|
| Best Val Accuracy | 72.56% |
| Final Train Accuracy | 86.33% |
| Saved model | `finetuned_resnet18_adience.pth` |

---

### Stage 2 — Fine-tuned further on CelebA Dataset

| Parameter | Value |
|---|---|
| Dataset | CelebA (Gender attribute) |
| Train samples | 162,770 |
| Val samples | 19,867 |
| Test samples | 19,962 |
| Epochs | 10 |
| Batch size | 64 |
| Learning rate | 1e-4 |
| Optimizer | Adam (weight decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss |
| Pretrained from | Adience checkpoint |

**Results:**
| Metric | Value |
|---|---|
| Best Val Accuracy | 98.84% |
| Final Train Accuracy | 99.31% |
| Test Accuracy | 98.36% |
| Saved model | `celeba_finetuned.pth` |

---

### Data Augmentation
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness, contrast, saturation ±0.2)
- Normalize (ImageNet mean/std)
