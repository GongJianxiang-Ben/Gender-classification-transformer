import argparse
import torch
from transformers import ViTConfig, ViTForImageClassification
from celebA_data_loader import CelebAGenderDataset
from torch.utils.data import ConcatDataset
from adience_data_loader import CustomImageDataset,EmptyDataset,ImageClassificationCollator
from torchvision.transforms import CenterCrop, Resize, Compose
from torch.utils.data import DataLoader,WeightedRandomSampler
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.metrics import classification_report, accuracy_score
import json
import random
import lightning as pl
from transformers import ViTImageProcessor
from torchvision import transforms
from pathlib import  Path
def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
def tester(model,test_loader):
    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    y_test=[]
    y_pred=[]
    with torch.no_grad():          
        for batch_idx, batch in enumerate(test_loader):
            if(batch_idx%40==0):
                print(f"DEBUG: Running validation step {batch_idx}", flush=True)
            pixel_values = batch['pixel_values'].to(device)  
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values).logits                    
            preds = outputs.argmax(dim=1)
            y_pred.extend(preds.cpu().numpy())     
            y_test.extend(labels.cpu().numpy())
            """total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)"""
    target_names = ['Female', 'Male']   # for label encoding 0=Female, 1=Male

    report = classification_report(
        y_test, 
        y_pred, 
        target_names=target_names, 
        digits=2,
        output_dict=True
    )

    df = pd.DataFrame(report).T

    df = df[['precision', 'recall', 'f1-score', 'support']]
    df.columns = ['Precision', 'Recall', 'F1-Score', 'Support']

    df = df.drop(['accuracy', 'macro avg'], errors='ignore')


    df = df.rename(index={'weighted avg': 'Weighted_avg'})
    

    print(df.round(4))
def trainer(model_name):
    from Vit import ViT
    from Vit_deformConv import CustomDeformViT
    from Vit_dilatedConv import CustomDilatedViT
    from Vit_addCNN import EnhancedViTForImageClassification
    from Vit_VPTshallow import ShallowVPTViTForImageClassification
    from Vit_VPTdeep import DeepVPTViTForImageClassification
    from Vit_depth import DeepViT
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
    parser.add_argument("--import_from",  type=str, default=None)  
    #the different model to load 
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--dilated_size",type=int, default=2)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))

    model_map = {
        "ViT":          (ViT,          
                {"num_classes": 2}),
        "addCNN":       (EnhancedViTForImageClassification, 
                {"num_classes": 2,"deformable":True,          
                "dilation_rates":[1, 2, 4], 
                "insert_before":True,       
                "insert_after":False}),
        "VPTshallow":   (ShallowVPTViTForImageClassification, 
                {"num_classes": 2, "num_prompts":100,                   
                "prompt_dropout": 0.1,
                "freeze_backbone" : True, }),
        "VPTdeep":      (DeepVPTViTForImageClassification, 
                { "num_classes": 2,"num_prompts":100,                   
                "prompt_dropout": 0.2,
                "freeze_backbone" : True, }),
        "depth":        (DeepViT,      
                { "num_classes": 2, "num_extra_layers":8}),
        "deformConv":   (CustomDeformViT, 
                { "num_classes": 2,}),
        "dilatedConv":  (CustomDilatedViT, 
                { "num_classes": 2,"dilated_size":args.dilated_size}),
    }
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
            
            if hasattr(ds, 'samples'):
                for sample in ds.samples:
                    paths1.add(os.path.abspath(sample[0]))
        paths2 = set(os.path.abspath(sample[0]) for sample in val_ds.samples)
        
        
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
    cls, extra_kwargs = model_map[model_name]
    if(args.start=="scratch"):
        model=cls(config=config,epochs=args.epochs,**extra_kwargs)
    elif(args.start=="local" or args.start=="online"):
        model = cls(
             config=args.start,model_name=args.import_from,epochs=args.epochs,
            **extra_kwargs     
        )
    else:
        model=cls.load_from_checkpoint(args.start)
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
    check_val_every_n_epoch=3,callbacks=[checkpoint_callback,early_stop_callback],
    accumulate_grad_batches=4)
    print("Starting Trainer.fit()...", flush=True)
    
    trainer.fit(model, train_loader, val_loader)
    tester(model,test_loader)
