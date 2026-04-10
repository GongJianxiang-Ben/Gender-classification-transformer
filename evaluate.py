import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, classification_report
import numpy as np

from resnet18 import ResNet, BasicBlock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str, required=True)
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split",   type=float, default=0.2)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(args.data_dir, transform=val_tf)
    dataset_len  = len(full_dataset)
    val_size     = int(dataset_len * args.val_split)
    train_size   = dataset_len - val_size

    generator  = torch.Generator().manual_seed(args.seed)
    indices    = torch.randperm(dataset_len, generator=generator).tolist()
    val_indices = indices[train_size:]

    val_ds     = Subset(full_dataset, val_indices)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    print(f"Val samples: {len(val_ds)} | Classes: {full_dataset.classes}")

    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=2)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    all_preds, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            preds   = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc      = correct / total
    f1       = f1_score(all_labels, all_preds, average="weighted")
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    print(f"\n{'='*40}")
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 (weighted): {f1:.4f}")
    print(f"F1 (macro):    {f1_macro:.4f}")
    print(f"\nDetailed Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=full_dataset.classes))
    print(f"{'='*40}")


if __name__ == "__main__":
    main()

