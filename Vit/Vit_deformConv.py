from transformers import  AutoModelForImageClassification
import lightning as pl

import torch
from transformers import ViTImageProcessor, ViTForImageClassification,ViTConfig

import torch.optim as optim

from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping

import torch.nn as nn
import json
from torchvision import transforms
from celebA_data_loader import CelebAGenderDataset
from pathlib import  Path
import argparse
from torch.utils.data import ConcatDataset
from adience_data_loader import CustomImageDataset,EmptyDataset,ImageClassificationCollator
from torchvision.ops import DeformConv2d
from transformers.models.vit.modeling_vit import ViTEmbeddings
from utils import trainer,seed_everything
from Vit import ViT
class DeformablePatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, patch_size=16,
                 pretrained_weight: nn.Conv2d = None,
                 bias: bool = True,):
        super().__init__()
        
        self.patch_size = patch_size
        
        # offset: 2*k*k
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * patch_size * patch_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.projection = DeformConv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        # IMPORTANT: initialize offset to zero (stable start)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        nn.init.constant_(self.projection.weight, 0)
        nn.init.constant_(self.projection.bias, 0)
        if pretrained_weight is not None:
            with torch.no_grad():
                self.projection.weight.copy_(pretrained_weight.weight)
                if bias and pretrained_weight.bias is not None:
                    self.projection.bias.copy_(pretrained_weight.bias)

    def forward(self, pixel_values,interpolate_pos_encoding: bool = False):
        offset = self.offset_conv(pixel_values)
        x = self.projection(pixel_values, offset)   # (B, 768, 14, 14)
        
        # flatten → token
        x = x.flatten(2).transpose(1, 2)  # (B, 196, 768)
        return x
class DeformViTEmbeddings(ViTEmbeddings):
    
    def __init__(self, config,image_size=224):
        super().__init__(config)                       # 先初始化位置编码、dropout 等
        original_projection = self.patch_embeddings.projection
        self.patch_embeddings = DeformablePatchEmbedding(
            
            patch_size=config.patch_size,
            in_channels=config.num_channels,
            embed_dim=config.hidden_size,
            bias=True,
            pretrained_weight=None
        )
        num_patches = (image_size // config.patch_size)** 2
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self, pixel_values,**kwargs):
    
        # output [B, num_patches, hidden_size]
        embeddings = self.patch_embeddings(pixel_values)

        # connect with CLS token
        batch_size = embeddings.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        #add positional encoding
        embeddings = embeddings + self.pos_embed
        
        # 4. Dropout
        embeddings = self.dropout(embeddings)
        return embeddings
class CustomDeformViT(pl.LightningModule):
    def __init__(self, num_classes,epochs,model_name=None, config=None,lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        if(config=="online"):
            self.base = AutoModelForImageClassification.from_pretrained(model_name)
        elif(config=="local"):
            self.base = self.base = ViT.load_from_checkpoint(model_name,num_classes=num_classes,epochs=epochs).model
        elif(config!=None):
            self.base=ViTForImageClassification(config)
        
        self.base.vit.embeddings = DeformViTEmbeddings(
            self.base.config
        )
        self.epochs=epochs
        self.base.train()
        #self.freeze_transformer(self.base)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
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
            self.base.vit.layernorm.requires_grad = True
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
    trainer("deformConv")
    