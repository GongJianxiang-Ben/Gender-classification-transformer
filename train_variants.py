import os
import copy
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import f1_score

from resnet18 import ResNet, BasicBlock
from resnet18_dilated import ResNet18Dilated
from resnet18_deformable import ResNet18Deformable


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AdienceFoldDataset(Dataset):
    def __init__(self, faces_dir, fold_files, transform=None):
        self.faces_dir = faces_dir
        self.transform = transform
        self.data = []

        for fold_file in fold_files:
            df = pd.read_csv(fold_file, sep="\t")
            for _, row in df.iterrows():
                user_id = str(row["user_id"])
                original_image = str(row["original_image"])
                gender = str(row["gender"]).strip().lower()
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

        print(f"Loaded {len(self.data)} samples | "
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
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    f1  = f1_score(all_labels, all_preds, average="weighted")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    return total_loss / total, acc, f1, f1_macro


def get_model(variant, num_classes=2):
    if variant == "vanilla":
        return ResNet(img_channels=3, num_layers=18,
                      block=BasicBlock, num_classes=num_classes)
    elif variant == "dilated":
        return ResNet18Dilated(img_channels=3, num_classes=num_classes)
    elif variant == "deformable":
        return ResNet18Deformable(img_channels=3, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      type=str, required=True)
    parser.add_argument("--checkpoint",    type=str, default=None)
    parser.add_argument("--variant",       type=str, required=True,
                        choices=["vanilla", "dilated", "deformable"])
    parser.add_argument("--phase1_epochs", type=int, default=5)
    parser.add_argument("--phase2_epochs", type=int, default=15)
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--lr_phase1",     type=float, default=1e-3)
    parser.add_argument("--lr_phase2",     type=float, default=1e-5)
    parser.add_argument("--output",        type=str, default=None)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--num_workers",   type=int, default=4)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"adience_{args.variant}_finetuned.pth"

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"Variant: {args.variant.upper()}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*50}\n")

    base_dir  = args.data_dir
    faces_dir = os.path.join(base_dir, "faces")
    fold_files = [os.path.join(base_dir, f"fold_{i}_data.txt") for i in range(5)]

    train_folds = [fold_files[0], fold_files[1], fold_files[2]]
    val_folds   = [fold_files[3]]
    test_folds  = [fold_files[4]]

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

    print("Loading train folds (0,1,2)...")
    train_ds = AdienceFoldDataset(faces_dir, train_folds, transform=train_tf)
    print("Loading val fold (3)...")
    val_ds   = AdienceFoldDataset(faces_dir, val_folds,   transform=val_tf)
    print("Loading test fold (4) - held out...")
    test_ds  = AdienceFoldDataset(faces_dir, test_folds,  transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = get_model(args.variant, num_classes=2)
    print(f"Parameters: {count_params(model):,}")

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Training from scratch")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    # ── Phase 1: Freeze backbone ──────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"PHASE 1: FC head only | LR={args.lr_phase1} | Epochs={args.phase1_epochs}")
    print(f"{'='*50}")

    for name, param in model.named_parameters():
        param.requires_grad = "fc" in name or "linear" in name

    optimizer1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_phase1, weight_decay=1e-4
    )
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.phase1_epochs)

    for epoch in range(1, args.phase1_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer1.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer1.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(dim=1) == labels).sum().item()
            total      += imgs.size(0)

        _, val_acc, val_f1, _ = evaluate(model, val_loader, device)
        scheduler1.step()
        print(f"[P1] Epoch [{epoch:02d}/{args.phase1_epochs}] "
              f"Train Acc: {correct/total:.4f} | "
              f"Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  Saved best (val_acc={val_acc:.4f})", flush=True)

    # ── Phase 2: Unfreeze all ─────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"PHASE 2: All layers | LR={args.lr_phase2} | Epochs={args.phase2_epochs}")
    print(f"{'='*50}")

    for param in model.parameters():
        param.requires_grad = True

    optimizer2 = optim.Adam(model.parameters(), lr=args.lr_phase2, weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.phase2_epochs)

    for epoch in range(1, args.phase2_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer2.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer2.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(dim=1) == labels).sum().item()
            total      += imgs.size(0)

        _, val_acc, val_f1, _ = evaluate(model, val_loader, device)
        scheduler2.step()
        print(f"[P2] Epoch [{epoch:02d}/{args.phase2_epochs}] "
              f"Train Acc: {correct/total:.4f} | "
              f"Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  Saved best (val_acc={val_acc:.4f})", flush=True)

    # ── Final test ────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Final Test on held-out fold 4...")
    model.load_state_dict(torch.load(args.output, map_location=device))
    test_loss, test_acc, test_f1, test_f1_macro = evaluate(model, test_loader, device)
    print(f"Test Accuracy:   {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test F1 (weighted): {test_f1:.4f}")
    print(f"Test F1 (macro):    {test_f1_macro:.4f}")
    print(f"Test Loss:       {test_loss:.4f}")
    print(f"Parameters:      {count_params(model):,}")
    print(f"Best Val Acc:    {best_val_acc:.4f}")
    print(f"Saved to:        {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()