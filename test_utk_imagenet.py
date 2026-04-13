import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

from utk_data_loader import UTKFaceDataset


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",     type=str, required=True)
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--test_ratio",  type=float, default=0.2)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    full_ds = UTKFaceDataset(
        img_dir=args.img_dir,
        transform=val_tf,
        align_with_adience=True,
        age_mode='continuous'
    )

    indices = list(range(len(full_ds)))
    _, test_indices = train_test_split(
        indices, test_size=args.test_ratio, random_state=args.seed)
    test_ds = Subset(full_ds, test_indices)
    print(f"Test set: {len(test_ds)} images")

    loader = DataLoader(test_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)

    # Load torchvision ResNet18
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, genders, _ in loader:
            imgs    = imgs.to(device, non_blocking=True)
            genders = genders.to(device, non_blocking=True)
            outputs = model(imgs)
            loss    = criterion(outputs, genders)
            preds   = outputs.argmax(dim=1)
            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == genders).sum().item()
            total      += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(genders.cpu().numpy())

    acc      = correct / total
    f1       = f1_score(all_labels, all_preds, average="weighted")
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    print(f"\n{'='*45}")
    print(f"UTKFace Cross-Dataset Evaluation")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*45}")
    print(f"Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 (weighted): {f1:.4f}")
    print(f"F1 (macro):    {f1_macro:.4f}")
    print(f"Loss:          {total_loss/total:.4f}")
    print(classification_report(all_labels, all_preds,
                                target_names=["Female", "Male"]))
    print(f"{'='*45}")


if __name__ == "__main__":
    main()