from transformers import  AutoModelForImageClassification
import lightning as pl

import torch
from transformers import ViTImageProcessor, ViTForImageClassification,ViTConfig

import torch.optim as optim

from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping
from utils import trainer,seed_everything
import torch.nn as nn
import json
from pathlib import  Path
from torchvision import transforms
from torch.utils.data import ConcatDataset
from adience_data_loader import CustomImageDataset,EmptyDataset,ImageClassificationCollator
from Vit import ViT
from celebA_data_loader import CelebAGenderDataset
import argparse
from transformers.models.vit.modeling_vit import ViTEmbeddings
class DilatedPatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, patch_size=16,
                 pretrained_weight: nn.Conv2d = None,
                 bias: bool = True,dilation: int = 2,):
        super().__init__()
        
        
        self.kernal_size=patch_size
        self.stride=patch_size
        self.padding=((self.kernal_size-1)*dilation- self.stride+1)/2.0
        if(self.padding%1!=0):
            self.pad=nn.ZeroPad2d((int(self.padding-0.5), int(self.padding+0.5),
                                    int(self.padding-0.5), int(self.padding+0.5)))
        else:
            self.pad=nn.ZeroPad2d((int(self.padding), int(self.padding),
                                    int(self.padding), int(self.padding)))
        
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.kernal_size,
            stride=self.stride,
            dilation=dilation,
            padding=0
        )
        # IMPORTANT: initialize offset to zero (stable start)
        nn.init.constant_(self.projection.weight, 0)
        nn.init.constant_(self.projection.bias, 0)
         
        if pretrained_weight is not None:
            with torch.no_grad():
                self.projection.weight.copy_(pretrained_weight.weight)
                if bias and pretrained_weight.bias is not None:
                    self.projection.bias.copy_(pretrained_weight.bias)

    def forward(self, pixel_values,interpolate_pos_encoding: bool = False):
        x = self.pad(pixel_values)
        x = self.projection(x)   # (B, 768, 14, 14)
        
        # flatten → token
        x = x.flatten(2).transpose(1, 2)  # (B, 196, 768)
        return x
class DilatedViTEmbeddings(ViTEmbeddings):
    def __init__(self, config, dilation: int = 2):
        super().__init__(config)
        original_projection = self.patch_embeddings.projection
        # Replace original patch embedding with dilated version
        self.patch_embeddings = DilatedPatchEmbedding(
            patch_size=config.patch_size,
            in_channels=config.num_channels,
            embed_dim=config.hidden_size,
            pretrained_weight=None,
            dilation=dilation,          # Try 2 or 3
            bias=True
        )
class CustomDilatedViT(pl.LightningModule):
    def __init__(self, num_classes,epochs,dilated_size=2,model_name=None, config=None,lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        if(config=="online"):
            self.base = AutoModelForImageClassification.from_pretrained(model_name)
        elif(config=="local"):
            self.base = self.base = ViT.load_from_checkpoint(model_name,num_classes=num_classes,epochs=epochs).model
        elif(config!=None):
            self.base=ViTForImageClassification(config)

        self.base.vit.embeddings = DilatedViTEmbeddings(
            self.base.config,
            dilated_size
        )
        self.base.train()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.epochs=epochs
    def freeze_transformer(self, model=None):
        for name, param in model.named_parameters():
            if "embeddings" in name:
                param.requires_grad = True
            else:
                # other all layers(Encoder, LayerNorm, Head ...)all forozen
                param.requires_grad = False
    def on_train_epoch_start(self):
        if self.current_epoch == 20:
            print("Unfreezing backbone...")
            for param in self.base.vit.encoder.layer[-4:].parameters():
                param.requires_grad = True
            for param in self.base.vit.encoder.layer[:4].parameters():
                param.requires_grad = True
            self.base.vit.layernorm.requires_grad = True #final layer norm
            for param in self.base.classifier.parameters():
                param.requires_grad = True
    def forward(self, x):
        return self.base(x)
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
    def configure_optimizers(self):
        
        optimizer=torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs-20,
            eta_min=0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
if __name__ == "__main__":
    trainer("dilatedConv")
    