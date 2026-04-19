"""
ResNet18 with ImageNet pretrained weights
Trained on Adience + CelebA combined (half-half WeightedRandomSampler)
Tested on UTKFace
"""
import os
import copy
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import f1_score, classification_report

from utk_data_loader import UTKFaceDataset
from sklearn.model_selection import train_test_split


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Adience Dataset ───────────────────────────────────────────────────────────
class AdienceFoldDataset(Dataset):
    def __init__(self, faces_dir, fold_files, transform=None):
        self.faces_dir = faces_dir
        self.transform = transform
        self.data = []

        for fold_file in fold_files:
            df = pd.read_csv(fold_file, sep="\t")
            for _, row in df.iterrows():
                user_id        = str(row["user_id"])
                original_image = str(row["original_image"])
                gender         = str(row["gender"]).strip().lower()
                if gender not in ["m", "f"]:
                    continue
                img_folder = os.path.join(faces_dir, user_id)
                if not os.path.exists(img_folder):
                    continue
                for fname in os.listdir(img_folder):
                    if original_image in fname:
                        label = 1 if gender == "m" else 0
                        self.data.append((os.path.join(img_folder, fname), label))
                        break

        print(f"Adience: {len(self.data)} samples | "
              f"Female: {sum(1 for _,l in self.data if l==0)} | "
              f"Male: {sum(1 for _,l in self.data if l==1)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── CelebA Dataset ────────────────────────────────────────────────────────────
class CelebAGenderDataset(Dataset):
    def __init__(self, img_dir, attr_file, split_file, split, transform=None):
        self.img_dir   = img_dir
        self.transform = transform

        attr_df  = pd.read_csv(attr_file)
        split_df = pd.read_csv(split_file)

        split_files = split_df[split_df["partition"] == split]["image_id"].values
        merged = attr_df[attr_df["image_id"].isin(split_files)][["image_id", "Male"]].copy()
        merged["label"] = (merged["Male"] == 1).astype(int)
        self.data = merged.reset_index(drop=True)

        print(f"CelebA split {split}: {len(self.data)} samples | "
              f"Female: {(self.data['label']==0).sum()} | "
              f"Male: {(self.data['label']==1).sum()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row      = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        img      = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, row["label"]


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc      = correct / total
    f1       = f1_score(all_labels, all_preds, average="weighted")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    return total_loss / total, acc, f1, f1_macro, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    # Adience
    parser.add_argument("--adience_dir",  type=str, required=True)
    # CelebA
    parser.add_argument("--celeba_img",   type=str, required=True)
    parser.add_argument("--celeba_attr",  type=str, required=True)
    parser.add_argument("--celeba_split", type=str, required=True)
    # UTKFace
    parser.add_argument("--utk_dir",      type=str, required=True)
    # Training
    parser.add_argument("--phase1_epochs",type=int, default=5)
    parser.add_argument("--phase2_epochs",type=int, default=15)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--lr_phase1",    type=float, default=1e-3)
    parser.add_argument("--lr_phase2",    type=float, default=1e-5)
    parser.add_argument("--output",       type=str, default="resnet18_imagenet_combined.pth")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--num_workers",  type=int, default=4)
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Transforms ───────────────────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Datasets ─────────────────────────────────────────────────────────────
    base_dir  = args.adience_dir
    faces_dir = os.path.join(base_dir, "faces")
    fold_files = [os.path.join(base_dir, f"fold_{i}_data.txt") for i in range(5)]

    # Adience: folds 0-3 for training, fold 4 held out for test
    print("\nLoading Adience train folds (0-3)...")
    adience_train_ds = AdienceFoldDataset(
        faces_dir, [fold_files[0], fold_files[1], fold_files[2], fold_files[3]],
        transform=train_tf
    )
    print("Loading Adience test fold (4) - held out...")
    adience_test_ds = AdienceFoldDataset(
        faces_dir, [fold_files[4]],
        transform=val_tf
    )

    # CelebA: train split only
    print("\nLoading CelebA train split...")
    celeba_train_ds = CelebAGenderDataset(
        args.celeba_img, args.celeba_attr, args.celeba_split,
        split=0, transform=train_tf
    )

    # ── Combined dataset with WeightedRandomSampler (half-half) ──────────────
    size_adience = len(adience_train_ds)
    size_celeba  = len(celeba_train_ds)
    total_size   = size_adience + size_celeba

    print(f"\nAdience train: {size_adience} | CelebA train: {size_celeba} | Total: {total_size}")

    combined_ds = ConcatDataset([adience_train_ds, celeba_train_ds])

    # Give equal weight to each dataset regardless of size
    weights = torch.DoubleTensor(
        [1.0 / size_adience] * size_adience +
        [1.0 / size_celeba]  * size_celeba
    )
    sampler = WeightedRandomSampler(weights, num_samples=total_size, replacement=True)

    train_loader = DataLoader(
        combined_ds, batch_size=args.batch_size,
        sampler=sampler, num_workers=args.num_workers, pin_memory=True
    )

    # Adience val (fold 3) for validation during training
    print("\nLoading Adience val fold (3)...")
    adience_val_ds = AdienceFoldDataset(
        faces_dir, [fold_files[3]], transform=val_tf
    )
    val_loader = DataLoader(
        adience_val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    adience_test_loader = DataLoader(
        adience_test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # ── UTKFace test set ──────────────────────────────────────────────────────
    print("\nLoading UTKFace test set...")
    utk_full_ds = UTKFaceDataset(
        img_dir=args.utk_dir,
        transform=val_tf,
        align_with_adience=True,
        age_mode='continuous'
    )
    indices = list(range(len(utk_full_ds)))
    _, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    from torch.utils.data import Subset
    utk_test_ds = Subset(utk_full_ds, test_indices)
    utk_loader  = DataLoader(
        utk_test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    print(f"UTKFace test: {len(utk_test_ds)} images")

    # ── Model: ResNet18 with ImageNet pretrained weights ──────────────────────
    print("\nLoading ResNet18 with ImageNet pretrained weights...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace final FC for binary gender classification
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model = model.to(device)

    criterion    = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    # ── Phase 1: Freeze backbone, train FC head only ──────────────────────────
    print(f"\n{'='*55}")
    print(f"PHASE 1: FC head only | LR={args.lr_phase1} | Epochs={args.phase1_epochs}")
    print(f"{'='*55}")

    for name, param in model.named_parameters():
        param.requires_grad = ("fc" in name)

    optimizer1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_phase1, weight_decay=1e-4
    )
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, T_max=args.phase1_epochs)

    for epoch in range(1, args.phase1_epochs + 1):
        model.train()
        correct, total, total_loss = 0, 0, 0.0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer1.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer1.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(dim=1) == labels).sum().item()
            total      += imgs.size(0)

        _, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, device)
        scheduler1.step()
        print(f"[P1] Epoch [{epoch:02d}/{args.phase1_epochs}] "
              f"Train Acc: {correct/total:.4f} | "
              f"Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  Saved best (val_acc={val_acc:.4f})", flush=True)

    # ── Phase 2: Unfreeze all ─────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"PHASE 2: All layers | LR={args.lr_phase2} | Epochs={args.phase2_epochs}")
    print(f"{'='*55}")

    for param in model.parameters():
        param.requires_grad = True

    optimizer2 = optim.Adam(
        model.parameters(), lr=args.lr_phase2, weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=args.phase2_epochs)

    for epoch in range(1, args.phase2_epochs + 1):
        model.train()
        correct, total, total_loss = 0, 0, 0.0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer2.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer2.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(dim=1) == labels).sum().item()
            total      += imgs.size(0)

        _, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, device)
        scheduler2.step()
        print(f"[P2] Epoch [{epoch:02d}/{args.phase2_epochs}] "
              f"Train Acc: {correct/total:.4f} | "
              f"Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  Saved best (val_acc={val_acc:.4f})", flush=True)

    # ── Final Evaluation ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    model.load_state_dict(torch.load(args.output, map_location=device))

    print("Adience Test (fold 4 held out)...")
    _, adience_acc, adience_f1, adience_f1_macro, preds, labels = evaluate(
        model, adience_test_loader, device)
    print(f"Adience Test Accuracy:   {adience_acc:.4f} ({adience_acc*100:.2f}%)")
    print(f"Adience F1 (weighted):   {adience_f1:.4f}")
    print(f"Adience F1 (macro):      {adience_f1_macro:.4f}")
    print(classification_report(labels, preds, target_names=["Female", "Male"]))

    print("\nUTKFace Cross-Dataset Test...")
    _, utk_acc, utk_f1, utk_f1_macro, utk_preds, utk_labels = evaluate(
        model, utk_loader, device)
    print(f"UTKFace Test Accuracy:   {utk_acc:.4f} ({utk_acc*100:.2f}%)")
    print(f"UTKFace F1 (weighted):   {utk_f1:.4f}")
    print(f"UTKFace F1 (macro):      {utk_f1_macro:.4f}")
    print(classification_report(utk_labels, utk_preds, target_names=["Female", "Male"]))

    print(f"\n{'='*55}")
    print(f"SUMMARY")
    print(f"Best Val Acc:          {best_val_acc:.4f}")
    print(f"Adience Test Acc:      {adience_acc:.4f}")
    print(f"UTKFace Test Acc:      {utk_acc:.4f}")
    print(f"Saved to:              {args.output}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()