from transformers import ViTConfig, ViTForImageClassification
from transformers import ViTImageProcessor
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader,WeightedRandomSampler
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping
import torch
from adience_data_loader import CustomImageDataset,EmptyDataset
from adience_data_loader import ImageClassificationCollator
from celebA_data_loader import CelebAGenderDataset
import argparse
from torchvision import transforms
import random
from pathlib import  Path
from torch.utils.data import ConcatDataset
import os,sys

class ViT(pl.LightningModule):
    def __init__(self, config,epochs, lr=1e-4):
        super().__init__()
        self.save_hyperparameters() 
        #initialize from designed structure 
        self.model = ViTForImageClassification(config)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.lr = lr
        self.epochs=epochs

    
    def forward(self, pixel_values):
        return self.model(pixel_values)

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
        optimizer=torch.optim.AdamW(self.parameters(), lr=self.lr,weight_decay=0.05)
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

def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",     type=str, required=True)
    parser.add_argument("--attr_file",   type=str, required=True)
    parser.add_argument("--split_file",  type=str, required=True)
    #for celebA 
    parser.add_argument("--checkpoint",  type=str, required=True)
    #place to store checkpoint
    parser.add_argument("--dataset",  type=str, required=True)
    #dataset to use for training
    parser.add_argument("--start",  type=str, required=True)  
    #the checkpoint to start from, or start from scratch with input "scratch"
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--dilated_size",type=int, default=2)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = ViTImageProcessor.from_pretrained('rizvandwiki/gender-classification')
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        
    ])
    collator = ImageClassificationCollator(feature_extractor)
    txt_dir="./label txt"
    folder = Path(txt_dir)
    paths = [str(p) for p in folder.iterdir()]
    if(args.dataset=="adience"):
        
        train_ds=EmptyDataset()
        val_ds=CustomImageDataset(img_dir="./aligned",
                            txt_file=paths[0],transform=val_tf)
        for p in paths[1:]:
            train_ds=ConcatDataset([train_ds,CustomImageDataset(img_dir="./aligned",
                            txt_file=p,transform=train_tf)])
        test_loader  = DataLoader(val_ds,  batch_size=args.batch_size,collate_fn=collator, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,  collate_fn=collator,shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
        paths1 = set()
        for ds in train_ds.datasets:
            # 确保跳过你那个 EmptyDataset (如果有的话)
            if hasattr(ds, 'samples'):
                for sample in ds.samples:
                    paths1.add(os.path.abspath(sample[0]))
        paths2 = set(os.path.abspath(sample[0]) for sample in val_ds.samples)
        
        # 寻找交集
        intersection = paths1.intersection(paths2)
        
        
        if len(intersection) == 0:
            print(f"no overlapping between train and valdation data")
        else:
            print(f"find {len(intersection)} overlapping samples")
            sys.exit()
    elif(args.dataset=="celebA"):
        train_ds = CelebAGenderDataset(args.img_dir, args.attr_file, args.split_file, split=0, transform=train_tf)
        val_ds   = CelebAGenderDataset(args.img_dir, args.attr_file, args.split_file, split=1, transform=val_tf)
        test_ds  = CustomImageDataset(img_dir="./aligned",
                            txt_file=paths[0],transform=val_tf)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,collate_fn=collator, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,  collate_fn=collator,shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
        train_ids = set(train_ds.data['image_id'])
        test_ids = set(val_ds.data['image_id'])

        intersection = train_ids.intersection(test_ids)
        if len(intersection) == 0:
            print(f"no overlapping between train and valdation data")
        else:
            print(f"find {len(intersection)} overlapping samples")
            sys.exit()
    elif(args.dataset=="both"):
        adience_subsets = []
        for p in paths[1:]:
            subset = CustomImageDataset(
                img_dir="./aligned",
                txt_file=p,
                transform=train_tf  
            )
            adience_subsets.append(subset)

        adience_ds = ConcatDataset(adience_subsets)
        size_adience = len(adience_ds)
        celebA_ds=CelebAGenderDataset(args.img_dir, args.attr_file, args.split_file, split=0, transform=train_tf)
        size_celeba = len(celebA_ds)
        train_ds=ConcatDataset([adience_ds,celebA_ds])
        val_ds=CustomImageDataset(img_dir="./aligned",
                            txt_file=paths[0],transform=val_tf)
        test_loader  = DataLoader(val_ds,  batch_size=args.batch_size,collate_fn=collator, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
        total_size = size_celeba + size_adience

        # give adience higher weight in ds
        weights = torch.DoubleTensor([1.0 / size_adience] * size_adience+[1.0 / size_celeba] * size_celeba )
        sampler = WeightedRandomSampler(weights, num_samples=total_size, replacement=True)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler, # use sampler is already random shuffle
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True
        )


    
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,collate_fn=collator, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    
    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        
        hidden_size=384,        
        num_hidden_layers=12,    
        num_attention_heads=12,  
        intermediate_size=3072, 
        
        num_labels=2,

        hidden_dropout_prob=0.1,         
        attention_probs_dropout_prob=0.1,  
        drop_path_rate=0.1,
    )
    if(args.start=="scratch"):
        model=ViT(config,args.epochs)
    else:

        model=ViT.load_from_checkpoint(args.start)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',     
        mode='max',                 
        dirpath=args.checkpoint,
        filename='vit-best-{epoch:02d}-{val_accuracy:.4f}',
        save_top_k=1,               # save best model
        save_last=True,             
        auto_insert_metric_name=False,
        verbose=True
    )
    early_stop_callback = EarlyStopping(
        monitor='val_acc',      
        min_delta=0.00,         
        patience=3,             
        verbose=True,           
        mode='max'              
    )
    model = model.to(device)
    trainer = pl.Trainer(accelerator='gpu', devices=1, precision=16, max_epochs=args.epochs,
    enable_progress_bar=False,log_every_n_steps=1,val_check_interval=1.0,
    check_val_every_n_epoch=3,callbacks=[checkpoint_callback,early_stop_callback])
    print("Starting Trainer.fit()...", flush=True)
    
    trainer.fit(model, train_loader, val_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():         
        for batch_idx, batch in enumerate(test_loader):
            if(batch_idx%40==0):
                print(f"DEBUG: Running validation step {batch_idx}", flush=True)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values).logits                   
            preds = outputs.argmax(dim=1)
            
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    avg_acc = total_correct / total_samples
    print("Average accuracy is",avg_acc) 