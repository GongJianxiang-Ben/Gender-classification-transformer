import argparse
import random
import re
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import ViTImageProcessor, ViTModel


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
    pl.seed_everything(seed, workers=True)


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
    for idx in range(3, len(parts)):
        token = parts[idx]
        if token.lower() in {"f", "m", "u"}:
            break
        age_tokens.append(token)

    age_str = " ".join(age_tokens).strip()
    if not age_str:
        return None

    matches = re.findall(r"\d+", age_str)
    if not matches:
        return None

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


def build_concat_dataset(img_dir, fold_files, transform=None):
    dataset = EmptyDataset()
    for fold_file in fold_files:
        fold_dataset = AdienceMultiTaskDataset(img_dir=img_dir, txt_file=fold_file, transform=transform)
        dataset = ConcatDataset([dataset, fold_dataset])
    return dataset


class ImageClassificationCollator:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, batch):
        images = [x[0] for x in batch]
        encodings = self.image_processor(images=images, return_tensors="pt")
        encodings["labels"] = torch.tensor([x[1] for x in batch], dtype=torch.long)
        encodings["age_labels"] = torch.tensor([x[2] for x in batch], dtype=torch.long)
        return encodings


class MultiTaskViT(nn.Module):
    def __init__(self, pretrained_name="rizvandwiki/gender-classification", lam=0.5, age_sigma=1.0):
        super().__init__()
        self.lam = lam
        self.age_sigma = age_sigma
        self.vit = ViTModel.from_pretrained(pretrained_name)
        hidden_size = self.vit.config.hidden_size

        self.gender_head = nn.Linear(hidden_size, 2)
        self.age_head = nn.Linear(hidden_size, 8)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None, age_labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]

        gender_logits = self.gender_head(cls_token)
        age_logits = self.age_head(cls_token)

        loss = None
        if labels is not None and age_labels is not None:
            loss_gender = self.ce(gender_logits, labels)
            soft_age = gaussian_soft_labels(age_labels, num_classes=8, sigma=self.age_sigma)
            loss_age = soft_ce_loss(age_logits, soft_age)
            loss = loss_gender + self.lam * loss_age

        return loss, gender_logits, age_logits


class MultiTaskClassifier(pl.LightningModule):
    def __init__(
        self,
        model,
        lr_backbone=2e-5,
        lr_gender_head=2e-5,
        lr_age_head=2e-5,
        max_epochs=5,
    ):
        super().__init__()
        self.model = model
        self.lr_backbone = lr_backbone
        self.lr_gender_head = lr_gender_head
        self.lr_age_head = lr_age_head
        self.max_epochs = max_epochs
        self.save_hyperparameters(ignore=["model"])
        self.gender_ce = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        _, gender_logits, age_logits = self.model(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
            age_labels=batch["age_labels"],
        )
        loss_gender = self.gender_ce(gender_logits, batch["labels"])
        soft_age = gaussian_soft_labels(
            batch["age_labels"],
            num_classes=8,
            sigma=self.model.age_sigma,
        )
        loss_age = soft_ce_loss(age_logits, soft_age)
        loss = loss_gender + self.model.lam * loss_age

        gender_acc = (gender_logits.argmax(dim=1) == batch["labels"]).float().mean()
        age_acc = (age_logits.argmax(dim=1) == batch["age_labels"]).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_gender_acc", gender_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_age_acc", age_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, gender_logits, age_logits = self.model(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
            age_labels=batch["age_labels"],
        )
        loss_gender = self.gender_ce(gender_logits, batch["labels"])
        soft_age = gaussian_soft_labels(
            batch["age_labels"],
            num_classes=8,
            sigma=self.model.age_sigma,
        )
        loss_age = soft_ce_loss(age_logits, soft_age)
        loss = loss_gender + self.model.lam * loss_age

        gender_acc = (gender_logits.argmax(dim=1) == batch["labels"]).float().mean()
        age_acc = (age_logits.argmax(dim=1) == batch["age_labels"]).float().mean()

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_gender_acc", gender_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_age_acc", age_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            [
                {"params": self.model.vit.parameters(), "lr": self.lr_backbone},
                {"params": self.model.gender_head.parameters(), "lr": self.lr_gender_head},
                {"params": self.model.age_head.parameters(), "lr": self.lr_age_head},
            ]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(self.max_epochs, 1),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def infer_state_dict(state):
    if not isinstance(state, dict):
        raise TypeError("Checkpoint is not a dict-like object.")

    for key in ("state_dict", "model_state_dict", "model", "net"):
        nested = state.get(key)
        if isinstance(nested, dict):
            return nested
    return state


def normalize_state_dict_keys(state_dict):
    prefixes = ("module.", "model.")
    replacements = {
        "classifier.": "gender_head.",
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
    vit_dir = Path(__file__).resolve().parent
    project_root = vit_dir.parent
    lam_str = str(lam)
    return {
        "project_root": project_root,
        "img_dir": project_root / "Dataset" / "aligned",
        "label_dir": project_root / "Dataset",
        "checkpoint": project_root / "checkpoint" / "vit" / "Vit-CelebA-pretrain.ckpt",
        "output": project_root / "checkpoint" / "vit" / f"vit_multitask-lam{lam_str}.ckpt",
        "log": project_root / "log" / "vit" / f"vit_multitask-lam{lam_str}.log",
    }


def evaluate_model(model, loader, device):
    model.eval()
    all_gender_preds, all_gender_labels = [], []
    all_age_preds, all_age_labels = [], []

    with torch.no_grad():
        progress = tqdm(loader, desc="Test", leave=False, dynamic_ncols=True)
        for batch in progress:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            _, gender_logits, age_logits = model(**batch)

            gender_preds = gender_logits.argmax(dim=1)
            age_preds = age_logits.argmax(dim=1)

            all_gender_preds.append(gender_preds.cpu())
            all_gender_labels.append(batch["labels"].cpu())
            all_age_preds.append(age_preds.cpu())
            all_age_labels.append(batch["age_labels"].cpu())

            running_gender_preds = torch.cat(all_gender_preds)
            running_gender_labels = torch.cat(all_gender_labels)
            running_gender_acc = (running_gender_preds == running_gender_labels).float().mean().item()
            progress.set_postfix(gender_acc=f"{running_gender_acc:.4f}")

    if not all_gender_preds:
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
        "gender_acc": gender_acc,
        "age_acc": age_acc,
        "age_1off_acc": age_1off_acc,
        "age_mae": age_mae,
        "gender_f1": gender_f1,
        "samples": int(age_labels.numel()),
        "age_group_results": age_group_results,
    }


def print_metrics_block(metrics, lam):
    print(f"\n{'=' * 50}")
    print(f"Task Setup: Multi-task | Lambda: {lam}")
    print(f"{'=' * 50}")
    print(f"Gender Acc (%): {metrics['gender_acc'] * 100:.2f}")
    print(f"Age Acc (%): {metrics['age_acc'] * 100:.2f}")
    print(f"Age 1-off Acc (%): {metrics['age_1off_acc'] * 100:.2f}")
    print(f"Age MAE (buckets): {metrics['age_mae']:.3f}")
    print(f"Gender F1: {metrics['gender_f1']:.4f}")
    print(f"Samples: {metrics['samples']}")

    print("\nGender Accuracy Breakdown by Age Group:")
    for group in metrics["age_group_results"]:
        print(
            f"{group['age_group']:>5} | "
            f"samples={group['samples']:4d} | "
            f"gender_acc={group['gender_acc'] * 100:6.2f}%"
        )


def evaluate_checkpoint(checkpoint_path, pretrained_name, lam, age_sigma, loader, device):
    model = MultiTaskViT(pretrained_name=pretrained_name, lam=lam, age_sigma=age_sigma)
    load_checkpoint(model, str(checkpoint_path), device="cpu")
    model = model.to(device)
    return evaluate_model(model, loader, device)


def main():
    PRETRAINED = "rizvandwiki/gender-classification"
    LR_BACKBONE = 1e-4
    LR_GENDER_HEAD = 1e-4
    LR_AGE_HEAD = 5e-4
    LAM = 0.5
    AGE_SIGMA = 1.0
    MAX_EPOCHS = 5
    BATCH_SIZE = 64

    default_defaults = resolve_default_paths(LAM)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(default_defaults["project_root"]))
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--label_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=str(default_defaults["checkpoint"]))
    parser.add_argument("--output", type=str, default=str(default_defaults["output"]))
    parser.add_argument("--log_file", type=str, default=str(default_defaults["log"]))
    parser.add_argument("--pretrained_name", type=str, default=PRETRAINED)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr_backbone", type=float, default=LR_BACKBONE)
    parser.add_argument("--lr_gender_head", type=float, default=LR_GENDER_HEAD)
    parser.add_argument("--lr_age_head", type=float, default=LR_AGE_HEAD)
    parser.add_argument("--lam", type=float, default=LAM)
    parser.add_argument("--age_sigma", type=float, default=AGE_SIGMA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="16-mixed")
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
    init_checkpoint_path = None
    if args.checkpoint and str(args.checkpoint).lower() != "none":
        init_checkpoint_path = Path(args.checkpoint).resolve()
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

    image_processor = ViTImageProcessor.from_pretrained(args.pretrained_name)
    collator = ImageClassificationCollator(image_processor)

    print(f"Resolved image directory: {img_dir}")
    print(f"Resolved label directory: {label_dir}")
    print(f"Train folds: {[p.name for p in train_folds]}")
    print(f"Val folds: {[p.name for p in val_folds]}")
    print(f"Test folds: {[p.name for p in test_folds]}")
    print(f"Lambda: {args.lam}")
    print(f"Age sigma: {args.age_sigma}")
    print(f"LR backbone: {args.lr_backbone}")
    print(f"LR gender head: {args.lr_gender_head}")
    print(f"LR age head: {args.lr_age_head}")

    print("Loading train folds...")
    train_ds = build_concat_dataset(img_dir, train_folds)
    print("Loading val fold (same as test fold, aligned with current single-task setup)...")
    val_ds = build_concat_dataset(img_dir, val_folds)
    print("Loading test fold...")
    test_ds = build_concat_dataset(img_dir, test_folds)

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"Device: {device}")
    if use_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_gpu,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_gpu,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_gpu,
        collate_fn=collator,
    )

    multitask_model = MultiTaskViT(
        pretrained_name=args.pretrained_name,
        lam=args.lam,
        age_sigma=args.age_sigma,
    )
    if init_checkpoint_path:
        if not init_checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {init_checkpoint_path}")
        load_checkpoint(multitask_model, str(init_checkpoint_path), device="cpu")
    else:
        print("Training from scratch")

    classifier = MultiTaskClassifier(
        multitask_model,
        lr_backbone=args.lr_backbone,
        lr_gender_head=args.lr_gender_head,
        lr_age_head=args.lr_age_head,
        max_epochs=args.epochs,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_gender_acc",
        mode="max",
        dirpath=str(output_path.parent),
        filename=output_path.stem,
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision=args.precision if use_gpu else "32-true",
        max_epochs=args.epochs,
        enable_progress_bar=True,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
    )

    print("Starting Trainer.fit()...", flush=True)
    trainer.fit(classifier, train_loader, val_loader)

    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        raise RuntimeError("Best checkpoint path is empty after training.")

    print(f"Best checkpoint: {best_model_path}")
    metrics = evaluate_checkpoint(
        checkpoint_path=best_model_path,
        pretrained_name=args.pretrained_name,
        lam=args.lam,
        age_sigma=args.age_sigma,
        loader=test_loader,
        device=device,
    )
    print_metrics_block(metrics, args.lam)

    log_file.close()


if __name__ == "__main__":
    main()
