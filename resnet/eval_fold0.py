import os
import argparse
import random
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

from resnet18 import ResNet, BasicBlock
from resnet18_dilated import ResNet18Dilated
from resnet18_deformable import ResNet18Deformable
from utk_data_loader import UTKFaceDataset


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


def evaluate(model, loader, device, class_names=["Female", "Male"]):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0], batch[1]
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

    print(f"Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 (weighted): {f1:.4f}")
    print(f"F1 (macro):    {f1_macro:.4f}")
    print(f"Loss:          {total_loss/total:.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    return acc, f1, f1_macro


def load_model(checkpoint, device, use_torchvision=False, arch="vanilla"):
    if use_torchvision:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif arch == "dilated":
        model = ResNet18Dilated(img_channels=3, num_classes=2)
    elif arch == "deformable":
        model = ResNet18Deformable(img_channels=3, num_classes=2)
    else:
        model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=2)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adience_dir",  type=str, required=True)
    parser.add_argument("--utk_dir",      type=str, required=True)
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--label",        type=str, required=True)
    parser.add_argument("--torchvision",  action="store_true")
    parser.add_argument("--arch",         type=str, default="vanilla",
                        choices=["vanilla", "dilated", "deformable"])
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Adience fold 0 as test
    base_dir  = args.adience_dir
    faces_dir = os.path.join(base_dir, "faces")
    fold_files = [os.path.join(base_dir, f"fold_{i}_data.txt") for i in range(5)]

    print("\nLoading Adience fold 0 (test)...")
    adience_test_ds = AdienceFoldDataset(faces_dir, [fold_files[0]], transform=val_tf)
    adience_loader  = DataLoader(adience_test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)

    # UTKFace aligned test
    print("\nLoading UTKFace aligned (test)...")
    utk_full_ds = UTKFaceDataset(
        img_dir=args.utk_dir,
        transform=val_tf,
        align_with_adience=True,
        age_mode='continuous'
    )
    indices = list(range(len(utk_full_ds)))
    _, test_indices = train_test_split(indices, test_size=0.2, random_state=args.seed)
    utk_test_ds = Subset(utk_full_ds, test_indices)
    utk_loader  = DataLoader(utk_test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)
    print(f"UTKFace test: {len(utk_test_ds)} images")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device,
                       use_torchvision=args.torchvision, arch=args.arch)

    print(f"\n{'='*55}")
    print(f"MODEL: {args.label}")
    print(f"{'='*55}")

    print("\n--- Adience Test (Fold 0) ---")
    adience_acc, adience_f1, adience_f1_macro = evaluate(model, adience_loader, device)

    print("\n--- UTKFace Aligned Test ---")
    utk_acc, utk_f1, utk_f1_macro = evaluate(model, utk_loader, device)

    print(f"\n{'='*55}")
    print(f"SUMMARY: {args.label}")
    print(f"Adience Fold 0 Test Acc: {adience_acc:.4f}")
    print(f"UTKFace Aligned Test Acc: {utk_acc:.4f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()