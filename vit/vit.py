import argparse
import random
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


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


class ImageClassificationCollator:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, batch):
        images = [x[0] for x in batch]
        encodings = self.image_processor(images=images, return_tensors="pt")
        encodings["labels"] = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return encodings


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
    return sum(p.numel() for p in model.model.classifier.parameters())


def infer_state_dict(state):
    if not isinstance(state, dict):
        raise TypeError("Checkpoint is not a dict-like object.")

    for key in ("state_dict", "model_state_dict", "model", "net"):
        nested = state.get(key)
        if isinstance(nested, dict):
            return nested
    return state


def normalize_state_dict_keys(state_dict):
    prefixes = ("module.",)
    replacements = {
        "vit.": "model.vit.",
        "classifier.": "model.classifier.",
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


def build_vit_config():
    return ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        num_labels=2,
    )


def build_image_processor():
    return ViTImageProcessor(
        do_resize=True,
        size={"height": 224, "width": 224},
        do_rescale=True,
        do_normalize=True,
    )


class ViTGenderClassifier(pl.LightningModule):
    def __init__(self, model, epochs, lr=1e-4):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.epochs = epochs
        self.save_hyperparameters(ignore=["model"])

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits

    def training_step(self, batch, batch_idx):
        logits = self(batch["pixel_values"])
        labels = batch["labels"]
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["pixel_values"])
        labels = batch["labels"]
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(self.epochs, 1),
            eta_min=0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        progress = tqdm(loader, desc="Test", leave=False, dynamic_ncols=True)
        for batch in progress:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            logits = model(pixel_values)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            running_preds = torch.cat(all_preds)
            running_labels = torch.cat(all_labels)
            running_acc = (running_preds == running_labels).float().mean().item()
            progress.set_postfix(acc=f"{running_acc:.4f}")

    if not all_preds:
        raise RuntimeError("Test loader is empty. Please check your fold files and image directory.")

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return {
        "acc": (all_preds == all_labels).float().mean().item(),
        "f1": compute_binary_f1(all_preds, all_labels),
    }


def resolve_default_paths(output_name, log_name):
    vit_dir = Path(__file__).resolve().parent
    project_root = vit_dir.parent
    return {
        "project_root": project_root,
        "img_dir": project_root / "Dataset" / "aligned",
        "label_dir": project_root / "Dataset",
        "checkpoint": project_root / "checkpoint" / "vit" / "Vit-CelebA-pretrain.ckpt",
        "output": project_root / "checkpoint" / "vit" / output_name,
        "log": project_root / "log" / "vit" / log_name,
    }


def build_arg_parser(defaults):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(defaults["project_root"]))
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--label_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=str(defaults["checkpoint"]))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default=str(defaults["output"]))
    parser.add_argument("--log_file", type=str, default=str(defaults["log"]))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="16-mixed")
    return parser


def build_simple_model():
    config = build_vit_config()
    return ViTForImageClassification(config)


def run_training(model_builder, output_name, head_name, log_name):
    defaults = resolve_default_paths(output_name, log_name)
    parser = build_arg_parser(defaults)
    args = parser.parse_args()

    log_file = setup_logging(Path(args.log_file).resolve())
    seed_everything(args.seed)

    base_dir = Path(args.data_dir).resolve()
    img_dir = Path(args.img_dir).resolve() if args.img_dir else base_dir / "Dataset" / "aligned"
    label_dir = Path(args.label_dir).resolve() if args.label_dir else base_dir / "Dataset"
    init_checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else None
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

    image_processor = build_image_processor()
    collator = ImageClassificationCollator(image_processor)

    print(f"Head type: {head_name}")
    print(f"Resolved image directory: {img_dir}")
    print(f"Resolved label directory: {label_dir}")
    print(f"Train folds: {[p.name for p in train_folds]}")
    print(f"Val folds: {[p.name for p in val_folds]}")
    print(f"Test folds: {[p.name for p in test_folds]}")

    print("Loading train folds...")
    train_ds = build_concat_dataset(img_dir, train_folds, transform=None)
    print("Loading val fold (same as test fold, following train_vit/adience setup)...")
    val_ds = build_concat_dataset(img_dir, val_folds, transform=None)
    print("Loading test fold...")
    test_ds = build_concat_dataset(img_dir, test_folds, transform=None)

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

    model = ViTGenderClassifier(model_builder(), epochs=args.epochs, lr=args.lr)
    if init_checkpoint_path:
        if not init_checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {init_checkpoint_path}")
        load_checkpoint(model, str(init_checkpoint_path), device="cpu")
    else:
        print("Training from scratch")

    print(f"Head parameters: {head_parameter_count(model)}")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
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
    trainer.fit(model, train_loader, val_loader)

    best_model_path = checkpoint_callback.best_model_path or str(output_path.with_suffix(".ckpt"))
    if not best_model_path:
        raise RuntimeError("Best checkpoint path is empty after training.")

    print(f"Best checkpoint: {best_model_path}")
    load_checkpoint(model, best_model_path, device="cpu")
    model = model.to(device)

    test_metrics = evaluate(model, test_loader, device)
    print(f"Gender Acc (%): {test_metrics['acc'] * 100:.2f}")
    print(f"Gender F1: {test_metrics['f1']:.4f}")
    print(f"# Head Params: {head_parameter_count(model)}")

    log_file.close()


def main():
    run_training(
        model_builder=build_simple_model,
        output_name="vit_simple.ckpt",
        head_name="Simple (FC->output)",
        log_name="vit_simple.log",
    )


if __name__ == "__main__":
    main()
