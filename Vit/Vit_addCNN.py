from transformers import  AutoModelForImageClassification
import lightning as pl
import torch
from transformers import ViTImageProcessor, ViTForImageClassification,ViTConfig
from torchvision.transforms import CenterCrop, Resize, Compose
import inspect

from torch.utils.data import DataLoader
from transformers.models.vit.modeling_vit import ViTLayer
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping
import json
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import argparse
from Vit import ViT
from celebA_data_loader import CelebAGenderDataset
from torch.utils.data import Dataset
from pathlib import  Path

from torch.utils.data import ConcatDataset
from adience_data_loader import CustomImageDataset,EmptyDataset,ImageClassificationCollator
from torchvision.ops import DeformConv2d
from transformers.models.vit.modeling_vit import ViTEmbeddings
import torch.nn.functional as F
from torchvision.ops import  deform_conv2d
from transformers.modeling_outputs import ImageClassifierOutput
from utils import trainer,seed_everything
class LocalContextExtractor(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, img_size=224, 
                 dilation_rates=[1, 2, 3], deformable=False, 
                 reduction=4, use_residual=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_h = img_size // patch_size
        self.patch_w = img_size // patch_size
        self.use_residual = use_residual
        self.deformable = deformable
        # 1. projection：token -> 2D feature map (C=embed_dim)
        self.to_2d = nn.Linear(embed_dim, embed_dim)
        if deformable:
            # Deformable Conv 
            self.offset_conv = nn.Conv2d(embed_dim, 2 * 3 * 3, kernel_size=3, 
                                        padding=1, bias=False)  # add offset for 3*3 kernal
            self.deform_conv = DeformConv2d(
                in_channels=embed_dim, out_channels=embed_dim,
                kernel_size=3, padding=1, bias=False
            )
            self.modulation_conv = nn.Conv2d(embed_dim, 9, kernel_size=3, padding=1)  # optional mask
        else:
            # Dilated Conv 
            self.dilated_convs = nn.ModuleList()
            for rate in dilation_rates:
                self.dilated_convs.append(
                    nn.Conv2d(embed_dim, embed_dim // len(dilation_rates), 
                              kernel_size=3, padding=rate, dilation=rate, bias=False)
                )
            self.fuse = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.to_token = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        if reduction > 1:
            self.reduce = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // reduction, 1),
                nn.GELU(),
                nn.Conv2d(embed_dim // reduction, embed_dim, 1)
            )
        else:
            self.reduce = nn.Identity()
    def forward(self, x, height=None, width=None):
        # x: [B, num_patches + 1, embed_dim]  
        B, N, C = x.shape
        cls_token = x[:, :1]                    # [B, 1, C]   ← keep CLS token from local CNN
        patch_tokens = x[:, 1:]                 # [B, num_patches, C]

        # Calculate the spatial dimensions of the patch
        if height is None or width is None:
            height = width = int(patch_tokens.shape[1] ** 0.5)

        # 1. patch tokens → 2D feature map [B, C, H, W]
        feat_2d = self.to_2d(patch_tokens)                      # [B, num_patches, C]
        feat_2d = feat_2d.transpose(1, 2).reshape(B, C, height, width)

        # 2.Local context extraction (only performed on the patch, without affecting CLS)
        if self.deformable:
            offset = self.offset_conv(feat_2d)
            mask = torch.sigmoid(self.modulation_conv(feat_2d))
            local_feat = deform_conv2d(feat_2d, offset, self.deform_conv.weight,
                                       mask=mask, padding=1)
        else:
            dilated_outs = [conv(feat_2d) for conv in self.dilated_convs]
            local_feat = torch.cat(dilated_outs, dim=1)
            local_feat = self.fuse(local_feat)

        local_feat = self.reduce(local_feat)
        local_feat = F.gelu(local_feat)

        # 3. 2D → back to token sequence
        local_tokens = local_feat.flatten(2).transpose(1, 2)   # [B, num_patches, C]
        local_tokens = self.to_token(local_tokens)
        local_tokens = self.norm(local_tokens)
        local_tokens = self.dropout(local_tokens)

        # Residual connection
        if self.use_residual:
            local_tokens = patch_tokens + local_tokens

        # put back CLS token
        output = torch.cat([cls_token, local_tokens], dim=1)   # [B, num_patches + 1, C]
        return output
class EnhancedViTForImageClassification(pl.LightningModule):
    def __init__(self,  num_classes,epochs,model_name=None, config=None, 
                  insert_before=True, insert_after=True,
                 deformable=True, dilation_rates=[1,2,3],lr: float = 5e-5):
        super().__init__()
        self.save_hyperparameters()
        if(config=="online"):
            self.base = AutoModelForImageClassification.from_pretrained(model_name)
        elif(config=="local"):
            self.base = self.base = ViT.load_from_checkpoint(model_name,num_classes=num_classes,epochs=epochs).model
        elif(config!=None):
            self.base=ViTForImageClassification(config)
        self.epochs=epochs
        self.base.train()
        embed_dim = self.base.vit.config.hidden_size

        self.local_before = LocalContextExtractor(
            embed_dim=embed_dim, deformable=deformable, 
            dilation_rates=dilation_rates
        ) if insert_before else None

        self.local_after = LocalContextExtractor(
            embed_dim=embed_dim, deformable=deformable, 
            dilation_rates=dilation_rates
        ) if insert_after else None

        self.classifier = self.base.classifier
        #self.freeze_transformer(self.base)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    def freeze_transformer(self, model=None):
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
            else:
                # other all layers(Encoder, LayerNorm, Head ...)all forozen
                param.requires_grad = False
    def on_train_epoch_start(self):
        if self.current_epoch == 20:
            print("Unfreezing backbone...")
            for param in self.base.vit.encoder.layer[-4:].parameters():
                param.requires_grad = True
            
            self.base.vit.layernorm.requires_grad = True #final layer norm
            for param in self.base.classifier.parameters():
                param.requires_grad = True
    def forward(self, pixel_values, labels=None):
        # original ViT embeddings
        embeddings = self.base.vit.embeddings(pixel_values)

        #The Transformer Encoder processes layer by layer (here we hook each layer or only before/after the encoder).
        hidden_states = embeddings

        for i, layer in enumerate(self.base.vit.encoder.layer):
            if(i==6 or i==0 or i==11):
                # --- before transformer ：Local Context ---
                if self.local_before is not None:
                    hidden_states = self.local_before(hidden_states)

                # original ViT Layer（attention + intermediate + output）
                layer_outputs = layer(hidden_states)
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

                # ---after transformer ：Local Context ---
                if self.local_after is not None:
                    hidden_states = self.local_after(hidden_states)

        # last LayerNorm + CLS
        sequence_output = self.base.vit.layernorm(hidden_states)
        logits = self.classifier(sequence_output[:, 0, :])  # CLS token

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.base.num_labels), labels.view(-1))

        return ImageClassifierOutput(
            loss=loss,
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
                "interval": "step",      
                "frequency": 1
            }
        }
if __name__ == "__main__":
    trainer("addCNN")
    



    