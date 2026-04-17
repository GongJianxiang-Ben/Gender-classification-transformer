from transformers import  AutoModelForImageClassification
import lightning as pl
import torch
from transformers import ViTImageProcessor, ViTForImageClassification,ViTConfig
from torchvision.transforms import CenterCrop, Resize, Compose
import inspect

from torch.utils.data import DataLoader
from transformers.models.vit.modeling_vit import ViTLayer
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping

import torch.nn as nn
import json
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup
import torch.optim as optim
import argparse
from Vit import ViT
from celebA_data_loader import CelebAGenderDataset
from torch.utils.data import Dataset
from pathlib import  Path
import os
from torch.utils.data import ConcatDataset
from adience_data_loader import CustomImageDataset,EmptyDataset,ImageClassificationCollator
from torchvision.ops import DeformConv2d
from transformers.models.vit.modeling_vit import ViTEmbeddings
import torch.nn.functional as F
from torchvision.ops import  deform_conv2d
from transformers.modeling_outputs import ImageClassifierOutput
from utils import trainer,seed_everything

class DeepVPTViTForImageClassification(pl.LightningModule):
    def __init__(self,
        num_classes,epochs,model_name=None, config=None,
        num_prompts: int = 100,                    # number of prompts in VPT
        prompt_dropout: float = 0.2,
        freeze_backbone: bool = True,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if(config=="online"):
            self.base = AutoModelForImageClassification.from_pretrained(model_name)
        elif(config=="local"):
            self.base = ViT.load_from_checkpoint(model_name,num_classes=num_classes,epochs=epochs).model
        elif(config!=None):
            self.base=ViTForImageClassification(config)
        
        self.epochs=epochs
        self.base.train()
        self.embed_dim = self.base.config.hidden_size
        self.num_layers=self.base.config.num_hidden_layers
        self.prompts = nn.Parameter(
            torch.randn(self.num_layers, num_prompts, self.embed_dim)
        )
        self.prompt_dropout = nn.Dropout(p=prompt_dropout)
        self.classifier = nn.Linear(self.embed_dim, num_classes) 
        if freeze_backbone:
            for param in self.base.parameters():
                param.requires_grad = False
        self.base.vit.gradient_checkpointing_enable()
        self.base.vit.enable_input_require_grads()
        self.criterion=nn.CrossEntropyLoss()
    def forward(self, pixel_values):
        # x: (B, 3, 224, 224) → patch embeddings
        B = pixel_values.shape[0]
        embedding_output = self.base.vit.embeddings(pixel_values)

        x=embedding_output
        for i, layer in enumerate(self.base.vit.encoder.layer):
            # take out current layer prompt
            prompts = self.prompts[i].unsqueeze(0).expand(B, -1, -1)   # (B, num_prompts, embed_dim)
            prompts = self.prompt_dropout(prompts)

            if i == 0:
                #first layer; insert prompts between CLS and Patches
                x = torch.cat((x[:, :1], prompts, x[:, 1:]), dim=1)
            else:
                # subsequent layer replace the previous prompt part
                # x structure is still [CLS (1), Prompts (num_prompts), Patches (N)]
                x = torch.cat((x[:, :1], prompts, x[:, 1 + self.hparams.num_prompts :]), dim=1)
            # through current Transformer layer（backbone freeze）
            layer_output = layer(x)
            x = layer_output[0]
        cls_output = x[:, 0]                                      # (B, embed_dim)
        if hasattr(self.base.vit, 'layernorm'):
            cls_output = self.base.vit.layernorm(cls_output)

        # 4.classifier
        logits = self.classifier(cls_output)
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
    def configure_optimizers(self):
        # collect parameters
        param_groups = [
                   
           {
                "params": self.prompts,           
                "lr": self.hparams.lr
            },
            {
                "params": self.classifier.parameters(),
                "lr": self.hparams.lr * 0.5
            }   
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        
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
    trainer("VPTdeep")
    