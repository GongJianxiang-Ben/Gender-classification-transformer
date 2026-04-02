#CUDA_VISIBLE_DEVICES=0 nohup python train.py > train.log 2>&1 &

import os
import math
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import pytorch_lightning as pl
from torchmetrics import Accuracy
from transformers import ViTModel, ViTImageProcessor

# =============================================================================
# 1. Dataset
# =============================================================================
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, txt_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.ages = {
            range(0, 3): 0, range(4, 7): 1, range(8, 14): 2,
            range(15, 21): 3, range(25, 33): 4, range(38, 44): 5,
            range(48, 54): 6, range(60, 101): 7
        }
        self.genders = {'f': 0, 'm': 1}

        self.samples = []
        with open(txt_file, 'r') as f:
            next(f)  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                subdir_name = parts[0]
                img_name = "landmark_aligned_face." + parts[2] + "." + parts[1]

                if parts[3] == "None":
                    continue

                if parts[4][-1] == ')':
                    age = int(parts[4][:-1])
                    gender = parts[5]
                else:
                    age = int(parts[3])
                    gender = parts[4]

                age_label = None
                for k in self.ages:
                    if age in k:
                        age_label = self.ages[k]
                        break
                if age_label is None:
                    continue

                if gender not in ['f', 'm']:
                    continue

                img_path = os.path.join(img_dir, subdir_name, img_name)
                if os.path.exists(img_path):
                    self.samples.append((img_path, self.genders[gender], age_label))

        self.classes = [0, 1]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        print(f"Dataset loaded! {len(self.samples)} images, {len(self.classes)} gender classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gender, age = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, gender, age


class EmptyDataset(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return 0
    def __getitem__(self, index):
        raise IndexError("Empty dataset cannot be indexed")


# =============================================================================
# 2. Collator
# =============================================================================
class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        encodings = self.feature_extractor(
            [x[0] for x in batch], return_tensors='pt'
        )
        encodings['labels'] = torch.tensor(
            [x[1] for x in batch], dtype=torch.long
        )
        encodings['age_labels'] = torch.tensor(
            [x[2] for x in batch], dtype=torch.long
        )
        return encodings


# =============================================================================
# 3. Multi-Task Model
# =============================================================================
class MultiTaskViT(nn.Module):
    def __init__(self, pretrained_name='rizvandwiki/gender-classification', lam=0.5):
        super().__init__()
        self.lam = lam
        self.vit = ViTModel.from_pretrained(pretrained_name)
        hidden_size = self.vit.config.hidden_size

        self.gender_head = nn.Linear(hidden_size, 2)
        self.age_head    = nn.Linear(hidden_size, 8)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None, age_labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]

        gender_logits = self.gender_head(cls_token)
        age_logits    = self.age_head(cls_token)

        loss = None
        if labels is not None and age_labels is not None:
            loss_gender = self.ce(gender_logits, labels)
            loss_age    = self.ce(age_logits, age_labels)
            loss = loss_gender + self.lam * loss_age

        return loss, gender_logits, age_logits


# =============================================================================
# 4. Lightning Module
# =============================================================================
class MultiTaskClassifier(pl.LightningModule):
    def __init__(self, model, lr: float = 2e-5):
        super().__init__()
        self.model = model
        self.save_hyperparameters('lr')
        self.val_gender_acc = Accuracy(task='binary', num_classes=2)
        self.val_age_acc    = Accuracy(task='multiclass', num_classes=8)

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.model(**batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, gender_logits, age_logits = self.model(**batch)
        g_acc = self.val_gender_acc(gender_logits.argmax(dim=1), batch['labels'])
        a_acc = self.val_age_acc(age_logits.argmax(dim=1), batch['age_labels'])
        self.log('val_loss',       loss,  prog_bar=True)
        self.log('val_gender_acc', g_acc, prog_bar=True)
        self.log('val_age_acc',    a_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# =============================================================================
# 5. 5-Fold Cross Validation
# =============================================================================
if __name__ == '__main__':

    IMG_DIR    = "/export/home3/n2500485g/Code/SC4001/Gender/age-gender-data/aligned"
    TXT_DIR    = "/export/home3/n2500485g/Code/SC4001/Gender/label txt"
    PRETRAINED = 'rizvandwiki/gender-classification'

    LR         = 2e-5
    LAM        = 0.5
    MAX_EPOCHS = 5
    BATCH_SIZE = 64

    NUM_GPUS = torch.cuda.device_count()
    print(f"Using {NUM_GPUS} GPU(s): "
          + ", ".join([torch.cuda.get_device_name(i) for i in range(NUM_GPUS)]))

    USE_GPU   = NUM_GPUS > 0
    STRATEGY  = 'ddp' if NUM_GPUS > 1 else 'auto'
    PRECISION = 16 if USE_GPU else 32
    DEVICE    = torch.device('cuda' if USE_GPU else 'cpu')

    feature_extractor = ViTImageProcessor.from_pretrained(PRETRAINED)
    collator = ImageClassificationCollator(feature_extractor)

    folder = Path(TXT_DIR)
    paths  = sorted([str(p) for p in folder.iterdir() if p.is_file()])
    K      = len(paths)
    print(f"Found {K} fold files: {[Path(p).name for p in paths]}")

    fold_results = []

    for i in range(K):
        print(f"\n{'='*50}")
        print(f"Fold {i+1}/{K}  —  val: {Path(paths[i]).name}")
        print('='*50)

        val_ds   = CustomImageDataset(img_dir=IMG_DIR, txt_file=paths[i])
        train_ds = EmptyDataset()
        for p in paths[:i] + paths[i+1:]:
            train_ds = ConcatDataset([
                train_ds,
                CustomImageDataset(img_dir=IMG_DIR, txt_file=p)
            ])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  collate_fn=collator, num_workers=4,
                                  shuffle=True, pin_memory=USE_GPU)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                                  collate_fn=collator, num_workers=4,
                                  pin_memory=USE_GPU)

        model      = MultiTaskViT(pretrained_name=PRETRAINED, lam=LAM)
        classifier = MultiTaskClassifier(model, lr=LR)

        trainer = pl.Trainer(
            accelerator='gpu' if USE_GPU else 'cpu',
            devices=NUM_GPUS if USE_GPU else 1,
            strategy=STRATEGY,
            precision=PRECISION,
            max_epochs=MAX_EPOCHS,
            enable_progress_bar=True,
        )
        trainer.fit(classifier, train_loader, val_loader)

        if trainer.global_rank == 0:
            eval_model = classifier.model.to(DEVICE)
            eval_model.eval()
            all_preds, all_labels = [], []

            with torch.no_grad():
                for batch in val_loader:

                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    _, gender_logits, _ = eval_model(**batch)
                    all_preds.append(gender_logits.argmax(dim=1).cpu())
                    all_labels.append(batch['labels'].cpu())

            all_preds  = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            acc = (all_preds == all_labels).float().mean().item()
            fold_results.append(acc)
            print(f"Fold {i+1} gender accuracy (full val set): {acc:.4f}")

    if trainer.global_rank == 0:
        print(f"\n{'='*50}")
        print(f"Average gender accuracy across {K} folds: {sum(fold_results)/K:.4f}")