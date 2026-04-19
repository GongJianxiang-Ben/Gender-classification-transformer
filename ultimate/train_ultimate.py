import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    ViT_B_16_Weights,
    resnet18,
    vit_b_16,
)
from torchvision.ops import DeformConv2d


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
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_enabled() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_enabled() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(args):
    use_ddp = args.ddp or int(os.environ.get("WORLD_SIZE", "1")) > 1
    if not use_ddp:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, device, 0

    if not torch.cuda.is_available():
        raise RuntimeError("DDP requested, but CUDA is not available.")

    if not is_dist_enabled():
        backend = "nccl"
        dist.init_process_group(backend=backend, init_method="env://")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return True, device, local_rank


def cleanup_distributed():
    if is_dist_enabled():
        dist.destroy_process_group()


def gaussian_soft_labels(labels: torch.Tensor, num_classes: int = 8, sigma: float = 1.0) -> torch.Tensor:
    classes = torch.arange(num_classes, device=labels.device).float()
    targets = labels.unsqueeze(1).float()
    dist = torch.exp(-0.5 * ((classes - targets) / sigma) ** 2)
    return dist / dist.sum(dim=1, keepdim=True)


def soft_ce_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def compute_binary_f1(preds: torch.Tensor, labels: torch.Tensor) -> float:
    preds = preds.long()
    labels = labels.long()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def safe_mean(values):
    return mean(values) if values else 0.0


def safe_std(values):
    return pstdev(values) if len(values) > 1 else 0.0


def map_age_to_bucket(age_value: int) -> int:
    for idx, (low, high) in enumerate(AGE_BUCKET_RANGES):
        if low <= age_value <= high:
            return idx
    centers = [(low + high) / 2 for low, high in AGE_BUCKET_RANGES]
    return min(range(len(centers)), key=lambda idx: abs(age_value - centers[idx]))


def parse_age_value(parts):
    if len(parts) < 5 or parts[3] == "None":
        return None
    age_tokens = []
    for token in parts[3:]:
        if token.lower() in {"f", "m", "u"}:
            break
        age_tokens.append(token)
    matches = re.findall(r"\d+", " ".join(age_tokens).strip())
    return int(matches[0]) if matches else None


class AdienceAgeGenderDataset(Dataset):
    def __init__(self, img_dir: Path, fold_file: Path, transform=None):
        self.transform = transform
        self.samples = []
        gender_to_idx = {"f": 0, "m": 1}

        with fold_file.open("r", encoding="utf-8") as handle:
            next(handle, None)
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 5 or parts[3] == "None":
                    continue

                user_id, original_image, face_id = parts[:3]
                age_value = parse_age_value(parts)
                if age_value is None:
                    continue

                gender = None
                for token in parts[3:]:
                    lowered = token.lower()
                    if lowered in gender_to_idx:
                        gender = lowered
                        break
                if gender is None:
                    continue

                img_name = f"landmark_aligned_face.{face_id}.{original_image}"
                img_path = img_dir / user_id / img_name
                if img_path.exists():
                    self.samples.append(
                        {
                            "path": img_path,
                            "gender": gender_to_idx[gender],
                            "age": map_age_to_bucket(age_value),
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample["gender"], sample["age"]


class AdienceCombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = [ds for ds in datasets if len(ds) > 0]
        self.cumulative = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self.cumulative.append(total)

    def __len__(self):
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        for ds_idx, bound in enumerate(self.cumulative):
            if index < bound:
                prev = 0 if ds_idx == 0 else self.cumulative[ds_idx - 1]
                return self.datasets[ds_idx][index - prev]
        raise IndexError(index)


def build_adience_dataset(img_dir: Path, fold_files, transform):
    return AdienceCombinedDataset(
        [AdienceAgeGenderDataset(img_dir=img_dir, fold_file=fold_file, transform=transform) for fold_file in fold_files]
    )


class DeformableConvAdapter(nn.Module):
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.offset = nn.Conv2d(
            conv.in_channels,
            2 * conv.kernel_size[0] * conv.kernel_size[1],
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=True,
        )
        self.deform = DeformConv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=conv.bias is not None,
        )
        nn.init.constant_(self.offset.weight, 0.0)
        nn.init.constant_(self.offset.bias, 0.0)
        with torch.no_grad():
            self.deform.weight.copy_(conv.weight)
            if conv.bias is not None and self.deform.bias is not None:
                self.deform.bias.copy_(conv.bias)

    def forward(self, x):
        return self.deform(x, self.offset(x))


def convert_resnet_layer_to_deformable(layer: nn.Sequential) -> None:
    for block in layer:
        block.conv2 = DeformableConvAdapter(block.conv2)


class MultiTaskResNet18(nn.Module):
    def __init__(self, lam: float = 0.5, age_sigma: float = 1.0):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)
        convert_resnet_layer_to_deformable(backbone.layer3)
        convert_resnet_layer_to_deformable(backbone.layer4)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.gender_head = nn.Linear(num_features, 2)
        self.age_head = nn.Linear(num_features, 8)
        self.ce = nn.CrossEntropyLoss()
        self.lam = lam
        self.age_sigma = age_sigma
        self.pretrain_source = "torchvision::IMAGENET1K_V1"

    def forward(self, x, gender_labels=None, age_labels=None):
        features = self.backbone(x)
        gender_logits = self.gender_head(features)
        age_logits = self.age_head(features)
        loss = None
        if gender_labels is not None and age_labels is not None:
            loss_gender = self.ce(gender_logits, gender_labels)
            loss_age = soft_ce_loss(
                age_logits,
                gaussian_soft_labels(age_labels, num_classes=8, sigma=self.age_sigma),
            )
            loss = loss_gender + self.lam * loss_age
        return loss, gender_logits, age_logits


class MultiTaskVPTDeepViT(nn.Module):
    def __init__(
        self,
        lam: float = 0.5,
        age_sigma: float = 1.0,
        num_prompts: int = 100,
        prompt_dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        backbone = vit_b_16(weights=weights)
        hidden_dim = backbone.hidden_dim
        num_layers = len(backbone.encoder.layers)

        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.num_prompts = num_prompts
        self.prompts = nn.Parameter(torch.randn(num_layers, num_prompts, hidden_dim) * 0.02)
        self.prompt_dropout = nn.Dropout(prompt_dropout)
        self.gender_head = nn.Linear(hidden_dim, 2)
        self.age_head = nn.Linear(hidden_dim, 8)
        self.ce = nn.CrossEntropyLoss()
        self.lam = lam
        self.age_sigma = age_sigma
        self.freeze_backbone = freeze_backbone
        self.pretrain_source = "torchvision::IMAGENET1K_V1"

        self.backbone.heads = nn.Identity()
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _process_tokens(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        x = self.backbone._process_input(x)
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = x + self.backbone.encoder.pos_embedding
        x = self.backbone.encoder.dropout(x)

        for layer_idx, block in enumerate(self.backbone.encoder.layers):
            prompts = self.prompt_dropout(self.prompts[layer_idx].unsqueeze(0).expand(n, -1, -1))
            if layer_idx == 0:
                x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)
            else:
                x = torch.cat([x[:, :1], prompts, x[:, 1 + self.num_prompts :]], dim=1)
            x = block(x)

        x = self.backbone.encoder.ln(x)
        return x[:, 0]

    def forward(self, x, gender_labels=None, age_labels=None):
        cls = self._process_tokens(x)
        gender_logits = self.gender_head(cls)
        age_logits = self.age_head(cls)
        loss = None
        if gender_labels is not None and age_labels is not None:
            loss_gender = self.ce(gender_logits, gender_labels)
            loss_age = soft_ce_loss(
                age_logits,
                gaussian_soft_labels(age_labels, num_classes=8, sigma=self.age_sigma),
            )
            loss = loss_gender + self.lam * loss_age
        return loss, gender_logits, age_logits


def evaluate_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    gender_preds_all, gender_labels_all = [], []
    age_preds_all, age_labels_all = [], []

    with torch.no_grad():
        for images, gender_labels, age_labels in loader:
            images = images.to(device, non_blocking=True)
            gender_labels = gender_labels.to(device, non_blocking=True)
            age_labels = age_labels.to(device, non_blocking=True)

            loss, gender_logits, age_logits = model(images, gender_labels=gender_labels, age_labels=age_labels)
            gender_preds = gender_logits.argmax(dim=1)
            age_preds = age_logits.argmax(dim=1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size
            gender_preds_all.append(gender_preds.cpu())
            gender_labels_all.append(gender_labels.cpu())
            age_preds_all.append(age_preds.cpu())
            age_labels_all.append(age_labels.cpu())

    gender_preds = torch.cat(gender_preds_all)
    gender_labels = torch.cat(gender_labels_all)
    age_preds = torch.cat(age_preds_all)
    age_labels = torch.cat(age_labels_all)
    age_diff = (age_preds - age_labels).abs().float()

    return {
        "loss": total_loss / total,
        "gender_acc": (gender_preds == gender_labels).float().mean().item(),
        "gender_f1": compute_binary_f1(gender_preds, gender_labels),
        "age_acc": (age_diff == 0).float().mean().item(),
        "age_1off_acc": (age_diff <= 1).float().mean().item(),
        "age_mae": age_diff.mean().item(),
        "samples": int(total),
    }


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0

    for images, gender_labels, age_labels in loader:
        images = images.to(device, non_blocking=True)
        gender_labels = gender_labels.to(device, non_blocking=True)
        age_labels = age_labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = model(images, gender_labels=gender_labels, age_labels=age_labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size

    return total_loss / total


def build_optimizer(model, arch: str, lr_backbone: float, lr_head: float):
    head_params = list(model.gender_head.parameters()) + list(model.age_head.parameters())
    if arch == "vit":
        groups = [
            {"params": [model.prompts], "lr": lr_head},
            {"params": head_params, "lr": lr_head},
        ]
        if not model.freeze_backbone:
            backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
            if backbone_params:
                groups.append({"params": backbone_params, "lr": lr_backbone})
        return torch.optim.AdamW(groups, weight_decay=1e-4)

    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("gender_head.") and not n.startswith("age_head.")]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=1e-4,
    )


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


@dataclass
class FoldResult:
    fold: int
    best_epoch: int
    checkpoint: str
    gender_acc: float
    gender_f1: float
    age_acc: float
    age_1off_acc: float
    age_mae: float


def create_model(args):
    if args.arch == "resnet":
        return MultiTaskResNet18(lam=args.lam, age_sigma=args.age_sigma)
    return MultiTaskVPTDeepViT(
        lam=args.lam,
        age_sigma=args.age_sigma,
        num_prompts=args.vpt_k,
        prompt_dropout=args.prompt_dropout,
        freeze_backbone=not args.vit_unfreeze_backbone,
    )


def run_fold(args, fold_idx: int, img_dir: Path, label_dir: Path, output_dir: Path, device: torch.device) -> FoldResult:
    fold_files = [label_dir / f"fold_{idx}_data.txt" for idx in range(5)]
    test_folds = [fold_files[fold_idx]]
    train_folds = [fold_files[idx] for idx in range(5) if idx != fold_idx]

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_ds = build_adience_dataset(img_dir=img_dir, fold_files=train_folds, transform=train_transform)
    test_ds = build_adience_dataset(img_dir=img_dir, fold_files=test_folds, transform=eval_transform)

    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(f"Fold {fold_idx} produced an empty dataset.")

    train_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(args).to(device)
    base_model = model
    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )
    optimizer = build_optimizer(base_model, args.arch, lr_backbone=args.lr_backbone, lr_head=args.lr_head)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_epoch = args.epochs
    checkpoint_path = output_dir / args.arch / f"fold_{fold_idx}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if is_main_process():
        print(f"\n===== Fold {fold_idx} / 4 | arch={args.arch} =====")
        print(f"Train folds: {[path.name for path in train_folds]}")
        print(f"Test fold: {[path.name for path in test_folds]}")
        print(f"Pretrain source: {base_model.pretrain_source}")
        print(f"Head choice: simple dual linear heads")
        if args.arch == "vit":
            print(f"ViT modification: VPT-Deep (K={args.vpt_k})")
        else:
            print("ResNet modification: deformable conv inserted into layer3/layer4")
        if args.distributed:
            print(f"DDP enabled | world_size={get_world_size()}")

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        if is_main_process():
            print(
                f"Fold {fold_idx} | Epoch {epoch:02d}/{args.epochs} | "
                f"train_loss={train_loss:.4f}"
            )

    if args.distributed:
        dist.barrier()

    final_metrics = None
    if is_main_process():
        final_metrics = evaluate_model(unwrap_model(model), test_loader, device)
        torch.save(
            {
                "model_state_dict": unwrap_model(model).state_dict(),
                "arch": args.arch,
                "fold": fold_idx,
                "epoch": args.epochs,
                "metrics": final_metrics,
                "config": vars(args),
            },
            checkpoint_path,
        )
        print(
            f"Fold {fold_idx} | Final Test | "
            f"gender_acc={final_metrics['gender_acc']:.4f} | "
            f"gender_f1={final_metrics['gender_f1']:.4f} | "
            f"age_acc={final_metrics['age_acc']:.4f} | "
            f"age_1off={final_metrics['age_1off_acc']:.4f}"
        )

    if args.distributed:
        payload = [final_metrics]
        dist.broadcast_object_list(payload, src=0)
        final_metrics = payload[0]
        dist.barrier()

    return FoldResult(
        fold=fold_idx,
        best_epoch=best_epoch,
        checkpoint=str(checkpoint_path),
        gender_acc=final_metrics["gender_acc"],
        gender_f1=final_metrics["gender_f1"],
        age_acc=final_metrics["age_acc"],
        age_1off_acc=final_metrics["age_1off_acc"],
        age_mae=final_metrics["age_mae"],
    )


def summarize_results(results):
    gender_accs = [item.gender_acc for item in results]
    gender_f1s = [item.gender_f1 for item in results]
    age_accs = [item.age_acc for item in results]
    age_1off_accs = [item.age_1off_acc for item in results]
    age_maes = [item.age_mae for item in results]
    return {
        "gender_acc_mean": safe_mean(gender_accs),
        "gender_acc_std": safe_std(gender_accs),
        "gender_f1_mean": safe_mean(gender_f1s),
        "gender_f1_std": safe_std(gender_f1s),
        "age_acc_mean": safe_mean(age_accs),
        "age_acc_std": safe_std(age_accs),
        "age_1off_acc_mean": safe_mean(age_1off_accs),
        "age_1off_acc_std": safe_std(age_1off_accs),
        "age_mae_mean": safe_mean(age_maes),
        "age_mae_std": safe_std(age_maes),
    }


def main():
    parser = argparse.ArgumentParser(description="Ultimate Adience multitask training for ResNet18-Deformable and ViT-VPTDeep.")
    parser.add_argument("--data_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "Dataset"))
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--arch", choices=["resnet", "vit"], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--age_sigma", type=float, default=1.0)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--vpt_k", type=int, default=100)
    parser.add_argument("--prompt_dropout", type=float, default=0.1)
    parser.add_argument("--vit_unfreeze_backbone", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    data_dir = Path(args.data_dir).resolve()
    img_dir = data_dir / "aligned"
    output_dir = Path(args.output_dir).resolve()
    label_dir = data_dir

    for fold_idx in range(5):
        fold_file = label_dir / f"fold_{fold_idx}_data.txt"
        if not fold_file.exists():
            raise FileNotFoundError(f"Missing fold file: {fold_file}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {img_dir}")

    args.distributed, device, args.local_rank = setup_distributed(args)

    try:
        if is_main_process():
            print(f"Device: {device}")
            print("Shared pretraining source for ViT and ResNet: torchvision ImageNet-1K weights")

        results = [run_fold(args, fold_idx, img_dir, label_dir, output_dir, device) for fold_idx in range(5)]
        summary = summarize_results(results)

        payload = {
            "arch": args.arch,
            "head": "simple dual linear heads",
            "multitask_lambda": args.lam,
            "pretraining_source": "torchvision::IMAGENET1K_V1",
            "folds": [result.__dict__ for result in results],
            "summary": summary,
        }

        if is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = output_dir / f"{args.arch}_5fold_metrics.json"
            metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            print("\n===== 5-Fold Summary =====")
            print(f"Gender Acc: {summary['gender_acc_mean'] * 100:.2f} +- {summary['gender_acc_std'] * 100:.2f}")
            print(f"Gender F1: {summary['gender_f1_mean']:.4f} +- {summary['gender_f1_std']:.4f}")
            print(f"Age Exact Acc: {summary['age_acc_mean'] * 100:.2f} +- {summary['age_acc_std'] * 100:.2f}")
            print(f"Age 1-off Acc: {summary['age_1off_acc_mean'] * 100:.2f} +- {summary['age_1off_acc_std'] * 100:.2f}")
            print(f"Age MAE: {summary['age_mae_mean']:.4f} +- {summary['age_mae_std']:.4f}")
            print(f"Saved metrics to: {metrics_path}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
