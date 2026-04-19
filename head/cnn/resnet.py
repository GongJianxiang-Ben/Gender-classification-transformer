import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm
from torchvision import transforms

from resnet18 import BasicBlock, ResNet


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AdienceGenderDataset(Dataset):
    def __init__(self, img_dir, txt_file, transform=None):
        self.img_dir = Path(img_dir)
        self.txt_file = Path(txt_file)
        self.transform = transform
        self.gender_to_idx = {"f": 0, "m": 1}
        self.samples = []

        with self.txt_file.open("r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                user_id = parts[0]
                original_image = parts[1]
                face_id = parts[2]

                if parts[3] == "None":
                    continue

                gender = parts[5] if parts[4].endswith(")") and len(parts) > 5 else parts[4]
                gender = gender.strip().lower()
                if gender not in self.gender_to_idx:
                    continue

                img_name = f"landmark_aligned_face.{face_id}.{original_image}"
                img_path = self.img_dir / user_id / img_name
                if img_path.exists():
                    self.samples.append((str(img_path), self.gender_to_idx[gender]))

        female_count = sum(1 for _, label in self.samples if label == 0)
        male_count = len(self.samples) - female_count
        print(
            f"Loaded {len(self.samples)} samples from {self.txt_file.name} | "
            f"Female: {female_count} | Male: {male_count}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("Empty dataset cannot be indexed")


def build_concat_dataset(img_dir, fold_files, transform):
    dataset = EmptyDataset()
    for fold_file in fold_files:
        fold_dataset = AdienceGenderDataset(img_dir=img_dir, txt_file=fold_file, transform=transform)
        dataset = ConcatDataset([dataset, fold_dataset])
    return dataset


def compute_binary_f1(preds, labels):
    preds = preds.long()
    labels = labels.long()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def head_parameter_count(model):
    return sum(p.numel() for p in model.fc.parameters())


def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for imgs, labels in progress:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            if total > 0:
                progress.set_postfix(
                    loss=f"{total_loss / total:.4f}",
                    acc=f"{correct / total:.4f}",
                )

    if total == 0:
        raise RuntimeError("Validation/Test loader is empty. Please check your fold files and image directory.")

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "f1": compute_binary_f1(all_preds, all_labels),
        "preds": all_preds,
        "labels": all_labels,
    }


def train_one_epoch(model, loader, optimizer, criterion, device, desc="Train"):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    for imgs, labels in progress:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += imgs.size(0)

        if total > 0:
            progress.set_postfix(
                loss=f"{total_loss / total:.4f}",
                acc=f"{correct / total:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    if total == 0:
        raise RuntimeError("Training loader is empty. Please check your fold files and image directory.")
    return total_loss / total, correct / total


def infer_state_dict(state):
    if not isinstance(state, dict):
        raise TypeError("Checkpoint is not a dict-like object.")

    for key in ("state_dict", "model_state_dict", "model", "net"):
        nested = state.get(key)
        if isinstance(nested, dict):
            return nested
    return state


def normalize_state_dict_keys(state_dict):
    prefixes = ("module.", "model.", "backbone.")
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        normalized[new_key] = value
    return normalized


def load_checkpoint(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = normalize_state_dict_keys(infer_state_dict(state))

    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    missing = sorted(set(model_state.keys()) - set(compatible.keys()))
    model.load_state_dict(compatible, strict=False)

    print(f"Loaded {len(compatible)} tensor(s) from checkpoint: {checkpoint_path}")
    if skipped:
        print(f"Skipped {len(skipped)} incompatible tensor(s), e.g. {skipped[:5]}")
    if missing:
        print(f"Model kept {len(missing)} randomly initialized tensor(s), e.g. {missing[:5]}")


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_logging(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)
    return log_file


def resolve_default_paths(output_name, log_name):
    cnn_dir = Path(__file__).resolve().parent
    project_root = cnn_dir.parent
    return {
        "project_root": project_root,
        "img_dir": project_root / "Dataset" / "aligned",
        "label_dir": project_root / "Dataset",
        "checkpoint": project_root / "checkpoint" / "cnn" / "celeba_scratch.pth",
        "output": project_root / "checkpoint" / "cnn" / output_name,
        "log": project_root / "log" / "cnn" / log_name,
    }


def build_arg_parser(defaults):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(defaults["project_root"]))
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--label_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=str(defaults["checkpoint"]))
    parser.add_argument("--phase1_epochs", type=int, default=10)
    parser.add_argument("--phase2_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_phase1", type=float, default=1e-3)
    parser.add_argument("--lr_phase2", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default=str(defaults["output"]))
    parser.add_argument("--log_file", type=str, default=str(defaults["log"]))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser


def build_simple_model():
    return ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=2)


def run_training(model_builder, output_name, head_name, log_name):
    defaults = resolve_default_paths(output_name, log_name)
    parser = build_arg_parser(defaults)
    args = parser.parse_args()

    log_file = setup_logging(Path(args.log_file).resolve())
    seed_everything(args.seed)

    base_dir = Path(args.data_dir).resolve()
    img_dir = Path(args.img_dir).resolve() if args.img_dir else base_dir / "Dataset" / "aligned"
    label_dir = Path(args.label_dir).resolve() if args.label_dir else base_dir / "Dataset"
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    output_path = Path(args.output).resolve()

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    fold_files = [label_dir / f"fold_{i}_data.txt" for i in range(5)]
    missing_folds = [str(path) for path in fold_files if not path.exists()]
    if missing_folds:
        raise FileNotFoundError(f"Missing fold files: {missing_folds}")

    train_folds = fold_files[1:]
    val_folds = [fold_files[0]]
    test_folds = [fold_files[0]]

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print(f"Head type: {head_name}")
    print(f"Resolved image directory: {img_dir}")
    print(f"Resolved label directory: {label_dir}")
    print(f"Train folds: {[p.name for p in train_folds]}")
    print(f"Val folds: {[p.name for p in val_folds]}")
    print(f"Test folds: {[p.name for p in test_folds]}")

    print("Loading train folds...")
    train_ds = build_concat_dataset(img_dir, train_folds, train_tf)
    print("Loading val fold (same as test fold, following adience.py setup)...")
    val_ds = build_concat_dataset(img_dir, val_folds, eval_tf)
    print("Loading test fold...")
    test_ds = build_concat_dataset(img_dir, test_folds, eval_tf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = model_builder().to(device)
    if checkpoint_path:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        load_checkpoint(model, str(checkpoint_path), device="cpu")
    else:
        print("Training from scratch")

    print(f"Head parameters: {head_parameter_count(model)}")

    criterion = nn.CrossEntropyLoss()
    best_val_acc = -1.0

    print(f"\n{'=' * 50}")
    print(f"PHASE 1: Freeze backbone, train {head_name} head only")
    print(f"Epochs: {args.phase1_epochs} | LR: {args.lr_phase1}")
    print(f"{'=' * 50}")

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")

    optimizer1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_phase1,
        weight_decay=1e-4,
    )
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=max(args.phase1_epochs, 1))

    for epoch in range(1, args.phase1_epochs + 1):
        print(f"\n[Phase1] Epoch {epoch:02d}/{args.phase1_epochs}", flush=True)
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer1,
            criterion,
            device,
            desc=f"Phase1 Train {epoch:02d}/{args.phase1_epochs}",
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            desc=f"Phase1 Val   {epoch:02d}/{args.phase1_epochs}",
        )
        scheduler1.step()

        print(
            f"[Phase1] Epoch [{epoch:02d}/{args.phase1_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} F1: {val_metrics['f1']:.4f}",
            flush=True,
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"  Saved best model (val_acc={val_metrics['acc']:.4f})", flush=True)

    print(f"\n{'=' * 50}")
    print("PHASE 2: Unfreeze all layers, fine-tune full network")
    print(f"Epochs: {args.phase2_epochs} | LR: {args.lr_phase2}")
    print(f"{'=' * 50}")

    for param in model.parameters():
        param.requires_grad = True

    optimizer2 = optim.Adam(model.parameters(), lr=args.lr_phase2, weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=max(args.phase2_epochs, 1))

    for epoch in range(1, args.phase2_epochs + 1):
        print(f"\n[Phase2] Epoch {epoch:02d}/{args.phase2_epochs}", flush=True)
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer2,
            criterion,
            device,
            desc=f"Phase2 Train {epoch:02d}/{args.phase2_epochs}",
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            desc=f"Phase2 Val   {epoch:02d}/{args.phase2_epochs}",
        )
        scheduler2.step()

        print(
            f"[Phase2] Epoch [{epoch:02d}/{args.phase2_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} F1: {val_metrics['f1']:.4f}",
            flush=True,
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"  Saved best model (val_acc={val_metrics['acc']:.4f})", flush=True)

    print("\nRunning final test on fold 0...")
    model.load_state_dict(torch.load(output_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device, desc="Test")

    print(f"Best checkpoint: {output_path}")
    print(f"Gender Acc (%): {test_metrics['acc'] * 100:.2f}")
    print(f"Gender F1: {test_metrics['f1']:.4f}")
    print(f"# Head Params: {head_parameter_count(model)}")
    log_file.close()


def main():
    run_training(
        model_builder=build_simple_model,
        output_name="resnet_simple.pth",
        head_name="Simple (FC->output)",
        log_name="resnet_simple.log",
    )


if __name__ == "__main__":
    main()
