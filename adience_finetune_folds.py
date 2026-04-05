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

from resnet18 import ResNet, BasicBlock


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
                        img_path = os.path.join(img_folder, fname)
                        label = 1 if gender == "m" else 0
                        self.data.append((img_path, label))
                        break

        print(f"Loaded {len(self.data)} samples from {len(fold_files)} fold(s) | "
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


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(dim=1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str, required=True)
    parser.add_argument("--checkpoint",  type=str, default=None)
    parser.add_argument("--phase1_epochs", type=int, default=5)
    parser.add_argument("--phase2_epochs", type=int, default=15)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--lr_phase1",   type=float, default=1e-3)
    parser.add_argument("--lr_phase2",   type=float, default=1e-5)
    parser.add_argument("--output",      type=str, default="adience_folds_finetuned.pth")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    base_dir = args.data_dir
    faces_dir = os.path.join(base_dir, "faces")
    fold_files = [os.path.join(base_dir, f"fold_{i}_data.txt") for i in range(5)]

    train_folds = [fold_files[0], fold_files[1], fold_files[2]]
    val_folds   = [fold_files[3]]
    test_folds  = [fold_files[4]]  # held out

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

    print("Loading train folds (0, 1, 2)...")
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

    # Load model
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=2)
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

    # ── Phase 1: Freeze backbone, train only FC head ──────────────────────────
    print(f"\n{'='*50}")
    print(f"PHASE 1: Freezing backbone, training FC head only")
    print(f"Epochs: {args.phase1_epochs} | LR: {args.lr_phase1}")
    print(f"{'='*50}")

    for name, param in model.named_parameters():
        if "fc" in name or "linear" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_phase1, weight_decay=1e-4
    )
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.phase1_epochs)

    for epoch in range(1, args.phase1_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer1, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler1.step()

        print(f"[Phase1] Epoch [{epoch:02d}/{args.phase1_epochs}] "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  Saved best model (val_acc={val_acc:.4f})", flush=True)

    # ── Phase 2: Unfreeze all, fine-tune full network ─────────────────────────
    print(f"\n{'='*50}")
    print(f"PHASE 2: Unfreezing all layers, fine-tuning full network")
    print(f"Epochs: {args.phase2_epochs} | LR: {args.lr_phase2}")
    print(f"{'='*50}")

    for param in model.parameters():
        param.requires_grad = True

    optimizer2 = optim.Adam(model.parameters(), lr=args.lr_phase2, weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.phase2_epochs)

    for epoch in range(1, args.phase2_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer2, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler2.step()

        print(f"[Phase2] Epoch [{epoch:02d}/{args.phase2_epochs}] "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  Saved best model (val_acc={val_acc:.4f})", flush=True)

    # ── Final test on held out fold 4 ─────────────────────────────────────────
    print("\nRunning final test on held-out fold 4...")
    model.load_state_dict(torch.load(args.output, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
    print(f"\nDone. Best val acc: {best_val_acc:.4f}. Saved to: {args.output}")


if __name__ == "__main__":
    main()