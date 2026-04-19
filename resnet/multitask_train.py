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


# ── Age group mapping ─────────────────────────────────────────────────────────
AGE_MAP = {
    "(0, 2)":   0,
    "(4, 6)":   1,
    "(8, 13)":  2,
    "(15, 20)": 3,
    "(25, 32)": 4,
    "(38, 43)": 5,
    "(48, 53)": 6,
    "(60, 100)": 7,
}
AGE_GROUP_NAMES = ["0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60+"]
NUM_AGE_CLASSES = 8


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_age(age_str):
    age_str = str(age_str).strip()
    if age_str in AGE_MAP:
        return AGE_MAP[age_str]
    return -1


# ── Dataset ───────────────────────────────────────────────────────────────────
class AdienceMultiTaskDataset(Dataset):
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
                age_label      = parse_age(row["age"])

                if gender not in ["m", "f"]:
                    continue
                if age_label == -1:
                    continue

                img_folder = os.path.join(faces_dir, user_id)
                if not os.path.exists(img_folder):
                    continue

                for fname in os.listdir(img_folder):
                    if original_image in fname:
                        gender_label = 1 if gender == "m" else 0
                        self.data.append((
                            os.path.join(img_folder, fname),
                            gender_label,
                            age_label
                        ))
                        break

        print(f"Loaded {len(self.data)} samples | "
              f"Female: {sum(1 for _,g,_ in self.data if g==0)} | "
              f"Male: {sum(1 for _,g,_ in self.data if g==1)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, gender_label, age_label = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, gender_label, age_label


# ── Multi-task model ──────────────────────────────────────────────────────────
class ResNet18MultiTask(nn.Module):
    def __init__(self, num_gender_classes=2, num_age_classes=8,
                 checkpoint=None):
        super(ResNet18MultiTask, self).__init__()
        # Load backbone
        backbone = ResNet(img_channels=3, num_layers=18,
                         block=BasicBlock, num_classes=2)
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            backbone.load_state_dict(state, strict=False)
            print(f"Loaded backbone: {checkpoint}")

        # Remove the original FC
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))

        # Two separate heads
        self.gender_head = nn.Linear(512, num_gender_classes)
        self.age_head    = nn.Linear(512, num_age_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        gender_out = self.gender_head(x)
        age_out    = self.age_head(x)
        return gender_out, age_out


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, lam):
    model.eval()
    gender_criterion = nn.CrossEntropyLoss()
    age_criterion    = nn.CrossEntropyLoss()

    total_loss = 0.0
    gender_correct, age_correct, total = 0, 0, 0
    all_gender_preds, all_gender_labels = [], []
    all_age_preds,    all_age_labels    = [], []

    # Per age group tracking
    age_group_correct = [0] * NUM_AGE_CLASSES
    age_group_total   = [0] * NUM_AGE_CLASSES

    with torch.no_grad():
        for imgs, gender_labels, age_labels in loader:
            imgs          = imgs.to(device, non_blocking=True)
            gender_labels = gender_labels.to(device, non_blocking=True)
            age_labels    = age_labels.to(device, non_blocking=True)

            gender_out, age_out = model(imgs)

            loss_gender = gender_criterion(gender_out, gender_labels)
            loss_age    = age_criterion(age_out, age_labels)
            loss        = loss_gender + lam * loss_age
            total_loss += loss.item() * imgs.size(0)

            gender_preds = gender_out.argmax(dim=1)
            age_preds    = age_out.argmax(dim=1)

            gender_correct += (gender_preds == gender_labels).sum().item()
            age_correct    += (age_preds    == age_labels).sum().item()
            total          += imgs.size(0)

            all_gender_preds.extend(gender_preds.cpu().numpy())
            all_gender_labels.extend(gender_labels.cpu().numpy())
            all_age_preds.extend(age_preds.cpu().numpy())
            all_age_labels.extend(age_labels.cpu().numpy())

            # Per age group gender accuracy
            for i in range(len(age_labels)):
                ag = age_labels[i].item()
                age_group_total[ag]   += 1
                if gender_preds[i] == gender_labels[i]:
                    age_group_correct[ag] += 1

    gender_acc = gender_correct / total
    age_acc    = age_correct / total
    gender_f1  = f1_score(all_gender_labels, all_gender_preds, average="weighted")

    per_age_gender_acc = []
    for i in range(NUM_AGE_CLASSES):
        if age_group_total[i] > 0:
            per_age_gender_acc.append(age_group_correct[i] / age_group_total[i])
        else:
            per_age_gender_acc.append(0.0)

    return (total_loss / total, gender_acc, age_acc,
            gender_f1, per_age_gender_acc)


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, lam):
    model.train()
    gender_criterion = nn.CrossEntropyLoss()
    age_criterion    = nn.CrossEntropyLoss()

    total_loss, gender_correct, total = 0.0, 0, 0

    for imgs, gender_labels, age_labels in loader:
        imgs          = imgs.to(device, non_blocking=True)
        gender_labels = gender_labels.to(device, non_blocking=True)
        age_labels    = age_labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        gender_out, age_out = model(imgs)

        loss_gender = gender_criterion(gender_out, gender_labels)
        loss_age    = age_criterion(age_out, age_labels)
        loss        = loss_gender + lam * loss_age
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * imgs.size(0)
        gender_correct += (gender_out.argmax(dim=1) == gender_labels).sum().item()
        total          += imgs.size(0)

    return total_loss / total, gender_correct / total


def run_experiment(args, lam, train_loader, val_loader, test_loader, device):
    print(f"\n{'='*55}")
    print(f"Lambda = {lam}")
    print(f"{'='*55}")

    model = ResNet18MultiTask(
        num_gender_classes=2,
        num_age_classes=NUM_AGE_CLASSES,
        checkpoint=args.checkpoint
    ).to(device)

    output_path = f"multitask_lam{str(lam).replace('.', '')}.pth"
    best_val_gender_acc = 0.0

    # Phase 1: freeze backbone
    for name, param in model.named_parameters():
        param.requires_grad = ("gender_head" in name or "age_head" in name)

    optimizer1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_phase1, weight_decay=1e-4
    )
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, T_max=args.phase1_epochs)

    for epoch in range(1, args.phase1_epochs + 1):
        train_loss, train_gender_acc = train_one_epoch(
            model, train_loader, optimizer1, device, lam)
        _, val_gender_acc, val_age_acc, val_f1, _ = evaluate(
            model, val_loader, device, lam)
        scheduler1.step()
        print(f"[P1][λ={lam}] Epoch [{epoch:02d}/{args.phase1_epochs}] "
              f"Train Gender Acc: {train_gender_acc:.4f} | "
              f"Val Gender Acc: {val_gender_acc:.4f} | "
              f"Val Age Acc: {val_age_acc:.4f}", flush=True)
        if val_gender_acc > best_val_gender_acc:
            best_val_gender_acc = val_gender_acc
            torch.save(model.state_dict(), output_path)

    # Phase 2: unfreeze all
    for param in model.parameters():
        param.requires_grad = True

    optimizer2 = optim.Adam(model.parameters(),
                            lr=args.lr_phase2, weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=args.phase2_epochs)

    for epoch in range(1, args.phase2_epochs + 1):
        train_loss, train_gender_acc = train_one_epoch(
            model, train_loader, optimizer2, device, lam)
        _, val_gender_acc, val_age_acc, val_f1, _ = evaluate(
            model, val_loader, device, lam)
        scheduler2.step()
        print(f"[P2][λ={lam}] Epoch [{epoch:02d}/{args.phase2_epochs}] "
              f"Train Gender Acc: {train_gender_acc:.4f} | "
              f"Val Gender Acc: {val_gender_acc:.4f} | "
              f"Val Age Acc: {val_age_acc:.4f}", flush=True)
        if val_gender_acc > best_val_gender_acc:
            best_val_gender_acc = val_gender_acc
            torch.save(model.state_dict(), output_path)

    # Final test
    model.load_state_dict(torch.load(output_path, map_location=device))
    _, test_gender_acc, test_age_acc, test_f1, per_age = evaluate(
        model, test_loader, device, lam)

    print(f"\n[RESULT][λ={lam}]")
    print(f"Test Gender Acc: {test_gender_acc:.4f} | "
          f"Test Age Acc: {test_age_acc:.4f} | "
          f"F1: {test_f1:.4f}")
    print(f"Per Age Group Gender Accuracy:")
    for i, (name, acc) in enumerate(zip(AGE_GROUP_NAMES, per_age)):
        print(f"  {name}: {acc:.4f}")

    return test_gender_acc, test_age_acc, test_f1, per_age


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      type=str, required=True)
    parser.add_argument("--checkpoint",    type=str, default=None)
    parser.add_argument("--phase1_epochs", type=int, default=5)
    parser.add_argument("--phase2_epochs", type=int, default=15)
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--lr_phase1",     type=float, default=1e-3)
    parser.add_argument("--lr_phase2",     type=float, default=1e-5)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--num_workers",   type=int, default=4)
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

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
    train_ds = AdienceMultiTaskDataset(faces_dir, train_folds, transform=train_tf)
    print("Loading val fold (3)...")
    val_ds   = AdienceMultiTaskDataset(faces_dir, val_folds,   transform=val_tf)
    print("Loading test fold (4)...")
    test_ds  = AdienceMultiTaskDataset(faces_dir, test_folds,  transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    # Run for all lambda values
    lambdas = [0.0, 0.1, 0.3, 0.5, 1.0]
    results = {}

    for lam in lambdas:
        gender_acc, age_acc, f1, per_age = run_experiment(
            args, lam, train_loader, val_loader, test_loader, device)
        results[lam] = {
            "gender_acc": gender_acc,
            "age_acc": age_acc,
            "f1": f1,
            "per_age": per_age
        }

    # Final summary table
    print(f"\n{'='*55}")
    print("FINAL SUMMARY TABLE 3")
    print(f"{'='*55}")
    print(f"{'Lambda':<10} {'Gender Acc':>12} {'Age Acc':>10} {'F1':>8}")
    print("-" * 45)
    for lam in lambdas:
        r = results[lam]
        print(f"{lam:<10} {r['gender_acc']:>12.4f} {r['age_acc']:>10.4f} {r['f1']:>8.4f}")

    print(f"\n{'='*55}")
    print("TABLE 4 — Per Age Group Gender Accuracy")
    print(f"{'='*55}")
    header = f"{'Age Group':<12}"
    for lam in lambdas:
        header += f"  λ={lam}"
    print(header)
    print("-" * 70)
    for i, name in enumerate(AGE_GROUP_NAMES):
        row = f"{name:<12}"
        for lam in lambdas:
            row += f"  {results[lam]['per_age'][i]:.4f}"
        print(row)


if __name__ == "__main__":
    main()