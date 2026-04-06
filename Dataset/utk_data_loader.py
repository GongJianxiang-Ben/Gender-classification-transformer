"""
UTKFace Dataset Loader
======================
A standalone PyTorch Dataset for the UTKFace benchmark.

Dataset: https://susanqq.github.io/UTKFace/
Filename format: [age]_[gender]_[race]_[datetime].jpg
  - age:    integer 0-116
  - gender: 0=male, 1=female
  - race:   0=White, 1=Black, 2=Asian, 3=Indian, 4=Others

Usage
-----
    from utk_data_loader import UTKFaceDataset
    from torch.utils.data import DataLoader

    dataset = UTKFaceDataset(img_dir='/path/to/UTKFace')
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)

    for images, genders, ages in loader:
        ...

Cross-dataset evaluation with Adience
--------------------------------------
Adience gender encoding: female=0, male=1
UTKFace gender encoding: male=0,   female=1  ← opposite!

Set `align_with_adience=True` (default) to flip gender labels
so they match Adience convention (female=0, male=1).
"""

import os
from PIL import Image
from torch.utils.data import Dataset


# Adience-compatible age buckets (for cross-dataset evaluation)
ADIENCE_AGE_BUCKETS = {
    range(0,  3):  0,
    range(4,  7):  1,
    range(8,  14): 2,
    range(15, 21): 3,
    range(25, 33): 4,
    range(38, 44): 5,
    range(48, 54): 6,
    range(60, 101): 7,
}


def map_age_to_adience_bucket(age: int):
    """
    Map a continuous age value to the corresponding Adience age bucket (0-7).
    Returns None if the age falls in a gap between Adience buckets
    (e.g. age 3, 22-24, 34-37, 54-59).
    """
    for age_range, label in ADIENCE_AGE_BUCKETS.items():
        if age in age_range:
            return label
    return None


class UTKFaceDataset(Dataset):
    """
    PyTorch Dataset for UTKFace.

    Parameters
    ----------
    img_dir : str
        Path to the folder containing UTKFace images.
    transform : callable, optional
        Torchvision transform applied to each PIL image.
    align_with_adience : bool
        If True (default), flip gender labels so that:
            female = 0, male = 1  (matches Adience convention)
        If False, keep original UTKFace convention:
            male = 0, female = 1
    age_mode : str
        How to return age labels. One of:
            'continuous'  - raw integer age (0-116)
            'adience'     - Adience bucket index (0-7);
                            images whose age falls in a gap are skipped
    """

    def __init__(
        self,
        img_dir: str,
        transform=None,
        align_with_adience: bool = True,
        age_mode: str = 'continuous',
    ):
        assert age_mode in ('continuous', 'adience'), \
            "age_mode must be 'continuous' or 'adience'"

        self.img_dir = img_dir
        self.transform = transform
        self.align_with_adience = align_with_adience
        self.age_mode = age_mode

        self.samples = []   # list of (img_path, gender_label, age_label)
        skipped = 0

        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith('.jpg'):
                continue

            parts = fname.split('_')
            if len(parts) < 4:          # malformed filename
                skipped += 1
                continue

            try:
                age    = int(parts[0])
                gender = int(parts[1])  # 0=male, 1=female (UTK convention)
            except ValueError:
                skipped += 1
                continue

            # ── gender label ────────────────────────────────────────────────
            if align_with_adience:
                # flip: UTK male(0)→1, female(1)→0  →  female=0, male=1
                gender_label = 1 - gender
            else:
                gender_label = gender   # keep UTK convention

            # ── age label ───────────────────────────────────────────────────
            if age_mode == 'adience':
                age_label = map_age_to_adience_bucket(age)
                if age_label is None:   # age in a gap, skip
                    skipped += 1
                    continue
            else:
                age_label = age         # raw continuous age

            img_path = os.path.join(img_dir, fname)
            if not os.path.exists(img_path):
                skipped += 1
                continue

            self.samples.append((img_path, gender_label, age_label))

        print(
            f"UTKFace loaded | "
            f"images: {len(self.samples)} | "
            f"skipped: {skipped} | "
            f"gender convention: {'Adience (f=0,m=1)' if align_with_adience else 'UTK (m=0,f=1)'} | "
            f"age mode: {age_mode}"
        )

    # ── standard Dataset interface ───────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gender, age = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, gender, age


if __name__ == '__main__':
    """
    Cross-dataset gender evaluation:
    model trained on Adience → tested on full UTKFace
    """
    import torch
    from torch.utils.data import DataLoader

    IMG_DIR    = '/path/to/UTKFace'
    MODEL_PATH = '/path/to/final_model.pth'
    BATCH_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds     = UTKFaceDataset(img_dir=IMG_DIR, align_with_adience=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, genders, _ in loader:
            images  = images.to(device)
            genders = genders.to(device)
            _, gender_logits, _ = model(pixel_values=images)
            all_preds.append(gender_logits.argmax(dim=1).cpu())
            all_labels.append(genders.cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = (all_preds == all_labels).float().mean().item()
    print(f"UTKFace cross-dataset gender accuracy: {acc:.4f}")
