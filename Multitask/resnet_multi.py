import argparse
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm
from torchvision import transforms
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from head.cnn.resnet18 import BasicBlock, ResNet


AGE_GROUP_NAMES = ["0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60+"]
AGE_BUCKET_RANGES = [
    (0, 2),
    (4, 6),
    (8, 13),
    (15, 20),
    (25, 32),
    (38, 43),
    (48, 53),
    (60, 100),
]


def gaussian_soft_labels(labels, num_classes=8, sigma=1.0):
    classes = torch.arange(num_classes, device=labels.device).float()
    targets = labels.unsqueeze(1).float()
    dist = torch.exp(-0.5 * ((classes - targets) / sigma) ** 2)
    return dist / dist.sum(dim=1, keepdim=True)


def soft_ce_loss(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


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


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def map_age_to_bucket(age_value):
    for idx, (low, high) in enumerate(AGE_BUCKET_RANGES):
        if low <= age_value <= high:
            return idx

    bucket_centers = [(low + high) / 2 for low, high in AGE_BUCKET_RANGES]
    nearest_idx = min(
        range(len(bucket_centers)),
        key=lambda idx: abs(age_value - bucket_centers[idx]),
    )
    return nearest_idx


def parse_age_value(parts):
    if len(parts) < 5 or parts[3] == "None":
        return None

    age_tokens = []
    gender_idx = None
    for idx in range(3, len(parts)):
        token = parts[idx]
        if token.lower() in {"f", "m", "u"}:
            gender_idx = idx
            break
        age_tokens.append(token)

    age_str = " ".join(age_tokens).strip()
    if not age_str:
        return None

    matches = re.findall(r"\d+", age_str)
    if not matches:
        return None

    # Use the lower bound for ranged buckets like "(25, 32)".
    return int(matches[0])


class AdienceMultiTaskDataset(Dataset):
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

                age_value = parse_age_value(parts)
                if age_value is None:
                    continue

                gender = None
                for token in parts[3:]:
                    lowered = token.strip().lower()
                    if lowered in self.gender_to_idx:
                        gender = lowered
                        break
                if gender is None:
                    continue

                age_label = map_age_to_bucket(age_value)

                img_name = f"landmark_aligned_face.{face_id}.{original_image}"
                img_path = self.img_dir / user_id / img_name
                if img_path.exists():
                    self.samples.append((str(img_path), self.gender_to_idx[gender], age_label))

        print(f"Loaded {len(self.samples)} samples from {self.txt_file.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gender, age = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, gender, age


class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("Empty dataset cannot be indexed")


def build_concat_dataset(img_dir, fold_files, transform):
    dataset = EmptyDataset()
    for fold_file in fold_files:
        fold_dataset = AdienceMultiTaskDataset(img_dir=img_dir, txt_file=fold_file, transform=transform)
        dataset = ConcatDataset([dataset, fold_dataset])
    return dataset


class ResNetMultiTask(nn.Module):
    def __init__(self, lam=0.5, age_sigma=1.0):
        super().__init__()
        base = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=2)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        self.gender_head = base.fc
        self.age_head = nn.Linear(512, 8)
        self.ce = nn.CrossEntropyLoss()
        self.lam = lam
        self.age_sigma = age_sigma

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x, labels=None, age_labels=None):
        features = self.extract_features(x)
        gender_logits = self.gender_head(features)
        age_logits = self.age_head(features)

        loss = None
        if labels is not None and age_labels is not None:
            loss_gender = self.ce(gender_logits, labels)
            soft_age = gaussian_soft_labels(age_labels, num_classes=8, sigma=self.age_sigma)
            loss_age = soft_ce_loss(age_logits, soft_age)
            loss = loss_gender + self.lam * loss_age

        return loss, gender_logits, age_logits


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
    replacements = {
        "fc.": "gender_head.",
    }

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

        for old, new in replacements.items():
            if new_key.startswith(old):
                new_key = new + new_key[len(old):]
        normalized[new_key] = value
    return normalized


def load_checkpoint(model, checkpoint_path, device):
    try:
        state = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint: {checkpoint_path}. "
            f"If you want to train from scratch, pass --checkpoint none."
        ) from exc
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


def resolve_default_paths(lam):
    cnn_dir = Path(__file__).resolve().parent
    project_root = cnn_dir.parent
    lam_str = str(lam)
    return {
        "project_root": project_root,
        "img_dir": project_root / "Dataset" / "aligned",
        "label_dir": project_root / "Dataset",
        "checkpoint": project_root / "checkpoint" / "cnn" / "celeba_scratch.pth",
        "output": project_root / "checkpoint" / "cnn" / f"resnet_multitask-lam{lam_str}.pth",
        "log": project_root / "log" / "cnn" / f"resnet_multitask-lam{lam_str}.log",
    }


def head_parameter_count(model):
    return sum(p.numel() for p in model.gender_head.parameters()) + sum(
        p.numel() for p in model.age_head.parameters()
    )


def evaluate_model(model, loader, device, desc="Eval"):
    model.eval()
    total_loss = 0.0
    total = 0
    all_gender_preds, all_gender_labels = [], []
    all_age_preds, all_age_labels = [], []

    with torch.no_grad():
        progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for imgs, gender_labels, age_labels in progress:
            imgs = imgs.to(device, non_blocking=True)
            gender_labels = gender_labels.to(device, non_blocking=True)
            age_labels = age_labels.to(device, non_blocking=True)

            loss, gender_logits, age_logits = model(imgs, labels=gender_labels, age_labels=age_labels)
            gender_preds = gender_logits.argmax(dim=1)
            age_preds = age_logits.argmax(dim=1)

            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

            all_gender_preds.append(gender_preds.cpu())
            all_gender_labels.append(gender_labels.cpu())
            all_age_preds.append(age_preds.cpu())
            all_age_labels.append(age_labels.cpu())

            running_gender_preds = torch.cat(all_gender_preds)
            running_gender_labels = torch.cat(all_gender_labels)
            running_gender_acc = (running_gender_preds == running_gender_labels).float().mean().item()
            progress.set_postfix(loss=f"{total_loss / total:.4f}", gender_acc=f"{running_gender_acc:.4f}")

    if total == 0:
        raise RuntimeError("Evaluation loader is empty. Please check your fold files and image directory.")

    gender_preds = torch.cat(all_gender_preds)
    gender_labels = torch.cat(all_gender_labels)
    age_preds = torch.cat(all_age_preds)
    age_labels = torch.cat(all_age_labels)

    gender_acc = (gender_preds == gender_labels).float().mean().item()
    gender_f1 = compute_binary_f1(gender_preds, gender_labels)
    age_diff = (age_preds - age_labels).abs().float()
    age_acc = (age_diff == 0).float().mean().item()
    age_1off_acc = (age_diff <= 1).float().mean().item()
    age_mae = age_diff.mean().item()

    age_group_results = []
    for age_idx, age_name in enumerate(AGE_GROUP_NAMES):
        mask = age_labels == age_idx
        sample_count = int(mask.sum().item())
        if sample_count > 0:
            group_gender_acc = (gender_preds[mask] == gender_labels[mask]).float().mean().item()
        else:
            group_gender_acc = 0.0
        age_group_results.append(
            {
                "age_group": age_name,
                "samples": sample_count,
                "gender_acc": group_gender_acc,
            }
        )

    return {
        "loss": total_loss / total,
        "gender_acc": gender_acc,
        "age_acc": age_acc,
        "age_1off_acc": age_1off_acc,
        "age_mae": age_mae,
        "gender_f1": gender_f1,
        "samples": int(age_labels.numel()),
        "age_group_results": age_group_results,
    }


def train_one_epoch(model, loader, optimizer, device, desc="Train"):
    model.train()
    total_loss = 0.0
    total = 0
    all_gender_preds, all_gender_labels = [], []

    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for imgs, gender_labels, age_labels in progress:
        imgs = imgs.to(device, non_blocking=True)
        gender_labels = gender_labels.to(device, non_blocking=True)
        age_labels = age_labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        loss, gender_logits, _ = model(imgs, labels=gender_labels, age_labels=age_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
        all_gender_preds.append(gender_logits.argmax(dim=1).detach().cpu())
        all_gender_labels.append(gender_labels.cpu())

        running_gender_preds = torch.cat(all_gender_preds)
        running_gender_labels = torch.cat(all_gender_labels)
        running_gender_acc = (running_gender_preds == running_gender_labels).float().mean().item()
        progress.set_postfix(
            loss=f"{total_loss / total:.4f}",
            gender_acc=f"{running_gender_acc:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    if total == 0:
        raise RuntimeError("Training loader is empty. Please check your fold files and image directory.")

    return total_loss / total, (torch.cat(all_gender_preds) == torch.cat(all_gender_labels)).float().mean().item()


def print_metrics_block(metrics, lam, head_params):
    print(f"\n{'=' * 50}")
    print(f"Task Setup: Multi-task | Lambda: {lam}")
    print(f"{'=' * 50}")
    print(f"Gender Acc (%): {metrics['gender_acc'] * 100:.2f}")
    print(f"Age Acc (%): {metrics['age_acc'] * 100:.2f}")
    print(f"Age 1-off Acc (%): {metrics['age_1off_acc'] * 100:.2f}")
    print(f"Age MAE (buckets): {metrics['age_mae']:.3f}")
    print(f"Gender F1: {metrics['gender_f1']:.4f}")
    print(f"Samples: {metrics['samples']}")
    print(f"# Head Params: {head_params}")

    print("\nGender Accuracy Breakdown by Age Group:")
    for group in metrics["age_group_results"]:
        print(
            f"{group['age_group']:>5} | "
            f"samples={group['samples']:4d} | "
            f"gender_acc={group['gender_acc'] * 100:6.2f}%"
        )


def build_optimizer(model, backbone_lr, gender_head_lr, age_head_lr, train_backbone):
    param_groups = []
    if train_backbone:
        backbone_params = [
            param
            for name, param in model.named_parameters()
            if not name.startswith("gender_head.") and not name.startswith("age_head.")
        ]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr})

    gender_head_params = list(model.gender_head.parameters())
    age_head_params = list(model.age_head.parameters())
    if gender_head_params:
        param_groups.append({"params": gender_head_params, "lr": gender_head_lr})
    if age_head_params:
        param_groups.append({"params": age_head_params, "lr": age_head_lr})

    return optim.Adam(param_groups, weight_decay=1e-4)


def main():
    LAM = 0.5
    AGE_SIGMA = 1.0
    PHASE1_EPOCHS = 10
    PHASE2_EPOCHS = 40
    BATCH_SIZE = 32
    LR_PHASE1_BACKBONE = 1e-3
    LR_PHASE1_GENDER_HEAD = 1e-3
    LR_PHASE1_AGE_HEAD = 1e-3
    LR_PHASE2_BACKBONE = 1e-4
    LR_PHASE2_GENDER_HEAD = 3e-4
    LR_PHASE2_AGE_HEAD = 3e-4

    default_defaults = resolve_default_paths(LAM)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(default_defaults["project_root"]))
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--label_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=str(default_defaults["checkpoint"]))
    parser.add_argument("--output", type=str, default=str(default_defaults["output"]))
    parser.add_argument("--log_file", type=str, default=str(default_defaults["log"]))
    parser.add_argument("--phase1_epochs", type=int, default=PHASE1_EPOCHS)
    parser.add_argument("--phase2_epochs", type=int, default=PHASE2_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr_phase1_backbone", type=float, default=LR_PHASE1_BACKBONE)
    parser.add_argument("--lr_phase1_gender_head", type=float, default=LR_PHASE1_GENDER_HEAD)
    parser.add_argument("--lr_phase1_age_head", type=float, default=LR_PHASE1_AGE_HEAD)
    parser.add_argument("--lr_phase2_backbone", type=float, default=LR_PHASE2_BACKBONE)
    parser.add_argument("--lr_phase2_gender_head", type=float, default=LR_PHASE2_GENDER_HEAD)
    parser.add_argument("--lr_phase2_age_head", type=float, default=LR_PHASE2_AGE_HEAD)
    parser.add_argument("--lam", type=float, default=LAM)
    parser.add_argument("--age_sigma", type=float, default=AGE_SIGMA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    lam_defaults = resolve_default_paths(args.lam)
    if args.output == str(default_defaults["output"]):
        args.output = str(lam_defaults["output"])
    if args.log_file == str(default_defaults["log"]):
        args.log_file = str(lam_defaults["log"])

    log_file = setup_logging(Path(args.log_file).resolve())
    seed_everything(args.seed)

    base_dir = Path(args.data_dir).resolve()
    img_dir = Path(args.img_dir).resolve() if args.img_dir else base_dir / "Dataset" / "aligned"
    label_dir = Path(args.label_dir).resolve() if args.label_dir else base_dir / "Dataset"
    checkpoint_path = None
    if args.checkpoint and str(args.checkpoint).lower() != "none":
        checkpoint_path = Path(args.checkpoint).resolve()
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

    print(f"Resolved image directory: {img_dir}")
    print(f"Resolved label directory: {label_dir}")
    print(f"Train folds: {[p.name for p in train_folds]}")
    print(f"Val folds: {[p.name for p in val_folds]}")
    print(f"Test folds: {[p.name for p in test_folds]}")
    print(f"Lambda: {args.lam}")
    print(f"Age sigma: {args.age_sigma}")
    print(f"LR phase1 backbone: {args.lr_phase1_backbone}")
    print(f"LR phase1 gender head: {args.lr_phase1_gender_head}")
    print(f"LR phase1 age head: {args.lr_phase1_age_head}")
    print(f"LR phase2 backbone: {args.lr_phase2_backbone}")
    print(f"LR phase2 gender head: {args.lr_phase2_gender_head}")
    print(f"LR phase2 age head: {args.lr_phase2_age_head}")

    print("Loading train folds...")
    train_ds = build_concat_dataset(img_dir, train_folds, train_tf)
    print("Loading val fold (same as test fold, aligned with current single-task setup)...")
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

    model = ResNetMultiTask(lam=args.lam, age_sigma=args.age_sigma).to(device)
    if checkpoint_path:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        load_checkpoint(model, str(checkpoint_path), device="cpu")
    else:
        print("Training from scratch")

    print(f"Head parameters: {head_parameter_count(model)}")

    print(f"\n{'=' * 50}")
    print("PHASE 1: Freeze backbone, train multitask heads only")
    print(
        f"Epochs: {args.phase1_epochs} | "
        f"LR gender head: {args.lr_phase1_gender_head} | "
        f"LR age head: {args.lr_phase1_age_head}"
    )
    print(f"{'=' * 50}")

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("gender_head.") or name.startswith("age_head.")

    optimizer1 = build_optimizer(
        model,
        backbone_lr=args.lr_phase1_backbone,
        gender_head_lr=args.lr_phase1_gender_head,
        age_head_lr=args.lr_phase1_age_head,
        train_backbone=False,
    )
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=max(args.phase1_epochs, 1))

    best_val_gender_acc = -1.0
    for epoch in range(1, args.phase1_epochs + 1):
        print(f"\n[Phase1] Epoch {epoch:02d}/{args.phase1_epochs}", flush=True)
        train_loss, train_gender_acc = train_one_epoch(
            model, train_loader, optimizer1, device, desc=f"Phase1 Train {epoch:02d}/{args.phase1_epochs}"
        )
        val_metrics = evaluate_model(model, val_loader, device, desc=f"Phase1 Val   {epoch:02d}/{args.phase1_epochs}")
        scheduler1.step()

        print(
            f"[Phase1] Epoch [{epoch:02d}/{args.phase1_epochs}] "
            f"Train Loss: {train_loss:.4f} Gender Acc: {train_gender_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Gender Acc: {val_metrics['gender_acc']:.4f} "
            f"Age Acc: {val_metrics['age_acc']:.4f} Age 1-off: {val_metrics['age_1off_acc']:.4f} "
            f"Gender F1: {val_metrics['gender_f1']:.4f}",
            flush=True,
        )

        if val_metrics["gender_acc"] > best_val_gender_acc:
            best_val_gender_acc = val_metrics["gender_acc"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"  Saved best model (val_gender_acc={val_metrics['gender_acc']:.4f})", flush=True)

    print(f"\n{'=' * 50}")
    print("PHASE 2: Unfreeze all layers, fine-tune full network")
    print(
        f"Epochs: {args.phase2_epochs} | "
        f"LR backbone: {args.lr_phase2_backbone} | "
        f"LR gender head: {args.lr_phase2_gender_head} | "
        f"LR age head: {args.lr_phase2_age_head}"
    )
    print(f"{'=' * 50}")

    for param in model.parameters():
        param.requires_grad = True

    optimizer2 = build_optimizer(
        model,
        backbone_lr=args.lr_phase2_backbone,
        gender_head_lr=args.lr_phase2_gender_head,
        age_head_lr=args.lr_phase2_age_head,
        train_backbone=True,
    )
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=max(args.phase2_epochs, 1))

    for epoch in range(1, args.phase2_epochs + 1):
        print(f"\n[Phase2] Epoch {epoch:02d}/{args.phase2_epochs}", flush=True)
        train_loss, train_gender_acc = train_one_epoch(
            model, train_loader, optimizer2, device, desc=f"Phase2 Train {epoch:02d}/{args.phase2_epochs}"
        )
        val_metrics = evaluate_model(model, val_loader, device, desc=f"Phase2 Val   {epoch:02d}/{args.phase2_epochs}")
        scheduler2.step()

        print(
            f"[Phase2] Epoch [{epoch:02d}/{args.phase2_epochs}] "
            f"Train Loss: {train_loss:.4f} Gender Acc: {train_gender_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Gender Acc: {val_metrics['gender_acc']:.4f} "
            f"Age Acc: {val_metrics['age_acc']:.4f} Age 1-off: {val_metrics['age_1off_acc']:.4f} "
            f"Gender F1: {val_metrics['gender_f1']:.4f}",
            flush=True,
        )

        if val_metrics["gender_acc"] > best_val_gender_acc:
            best_val_gender_acc = val_metrics["gender_acc"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"  Saved best model (val_gender_acc={val_metrics['gender_acc']:.4f})", flush=True)

    print("\nRunning final test on fold 0...")
    model.load_state_dict(torch.load(output_path, map_location=device))
    test_metrics = evaluate_model(model, test_loader, device, desc="Test")

    print(f"Best checkpoint: {output_path}")
    print_metrics_block(test_metrics, args.lam, head_parameter_count(model))
    log_file.close()


if __name__ == "__main__":
    main()
