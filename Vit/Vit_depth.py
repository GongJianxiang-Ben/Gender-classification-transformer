
from transformers import AutoImageProcessor, AutoModelForImageClassification,AutoConfig
import lightning as pl
from torchmetrics import Accuracy
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from torchvision.transforms import CenterCrop, Resize, Compose
import inspect
from torch.utils.data import DataLoader
from transformers.models.vit.modeling_vit import ViTLayer
from lightning.pytorch.callbacks import ModelCheckpoint

import torch.nn as nn

from transformers import get_cosine_schedule_with_warmup
import os
from PIL import Image
from torch.utils.data import Dataset
from pathlib import  Path
from transformers.modeling_outputs import ImageClassifierOutput
from torch.utils.data import ConcatDataset
from adience_data_loader import CustomImageDataset,EmptyDataset,ImageClassificationCollator
from utils import trainer
from Vit import ViT
class DeepViT(pl.LightningModule):
    def __init__(self, num_classes,epochs,model_name=None, config=None, num_extra_layers=6, lr=1e-4):
        super().__init__()

        self.save_hyperparameters()

        if(config=="online"):
            self.base = AutoModelForImageClassification.from_pretrained(model_name)
        elif(config=="local"):
            self.base = self.base = ViT.load_from_checkpoint(model_name,num_classes=num_classes,epochs=epochs).model
        elif(config!=None):
            self.base=ViTForImageClassification(config)

        # freeze backbone
        for param in self.base.vit.parameters():
            param.requires_grad = False
        self.base.train()
        # extra layers
        self.extra_layers = nn.ModuleList([
            ViTLayer(self.base.config)
            for _ in range(num_extra_layers)
        ])

        # residual scaling
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pixel_values):
        outputs = self.base.vit(pixel_values)
        hidden_states = outputs.last_hidden_state

        for layer in self.extra_layers:
            residual = hidden_states
            hidden_states = layer(hidden_states)[0]
            hidden_states = residual + self.alpha * hidden_states

        cls_token = hidden_states[:, 0]
        logits = self.base.classifier(cls_token)
        return ImageClassifierOutput(
            
            logits=logits,
            # hidden_states=... 
        )

    def training_step(self, batch, batch_idx):
        if(batch_idx%40==0):
            print(f"DEBUG: Running training step {batch_idx}", flush=True)
        x = batch["pixel_values"]
        y = batch["labels"]

        logits = self(x).logits
        loss = self.criterion(logits, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if(batch_idx%40==0):
            print(f"DEBUG: Running validation step {batch_idx}", flush=True)
        x = batch["pixel_values"]
        y = batch["labels"]

        logits = self(x).logits
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def on_train_epoch_start(self):
        if self.current_epoch == 5:
            print("Unfreezing backbone...")
            for param in self.base.vit.encoder.layer[-4:].parameters():
                param.requires_grad = True

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {"params": self.extra_layers.parameters(), "lr": self.hparams.lr},
            {"params": self.base.classifier.parameters(), "lr": self.hparams.lr},
            {"params": self.base.vit.parameters(), "lr": self.hparams.lr * 0.1},
        ], weight_decay=0.01)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=self.trainer.estimated_stepping_batches
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }






if __name__ == "__main__":
    trainer("depth")
    