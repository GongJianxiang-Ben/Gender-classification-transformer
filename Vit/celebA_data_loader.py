import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class CelebAGenderDataset(Dataset):
    def __init__(self, img_dir, attr_file, split_file, split, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        attr_df = pd.read_csv(attr_file)
        split_df = pd.read_csv(split_file)

        split_files = split_df[split_df["partition"] == split]["image_id"].values
        merged = attr_df[attr_df["image_id"].isin(split_files)][["image_id", "Male"]].copy()
        merged["label"] = (merged["Male"] == 1).astype(int)
        self.data = merged.reset_index(drop=True)

        print(f"Split {split}: {len(self.data)} samples | "
              f"Female: {(self.data['label']==0).sum()} | "
              f"Male: {(self.data['label']==1).sum()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, row["label"]