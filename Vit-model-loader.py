from transformers import AutoImageProcessor
from train_vit import ViT
from adience_data_loader import CustomImageDataset,ImageClassificationCollator

import argparse


feature_extractor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification")

from pathlib import  Path
import math
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir",     type=str, required=True)
parser.add_argument("--attr_folder",   type=str, required=True)
parser.add_argument("--dataset",  type=str, required=True)
#which dataset to use for testing
parser.add_argument("--start",  type=str, required=True) 
#the checkpoint to start from
parser.add_argument("--epochs",      type=int, default=50)
parser.add_argument("--batch_size",  type=int, default=64)
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()



model=ViT.load_from_checkpoint(args.start)
train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        
    ])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    
])
txt_dir="./label txt"
folder = Path(txt_dir)
paths = [str(p) for p in folder.iterdir()]
K = len(paths)
avg_acc=0
if(args.dataset=="adience"):
    test_ds=CustomImageDataset(img_dir="./aligned",
                          txt_file=paths[0],transform=val_tf)
elif(args.dataset=="UTK"):
    pass
#test_ds=CelebAGenderDataset(args.img_dir, args.attr_file, args.split_file, split=2, transform=val_tf)
collator = ImageClassificationCollator(feature_extractor)

test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collator, 
                         num_workers=args.num_workers, pin_memory=True)
total_correct = 0
total_samples = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():          
    for batch_idx, batch in enumerate(test_loader):
        if(batch_idx%40==0):
            print(f"DEBUG: Running validation step {batch_idx}", flush=True)
        pixel_values = batch['pixel_values'].to(device)  
        labels = batch['labels'].to(device)
        
        outputs = model(pixel_values)                    
        preds = outputs.argmax(dim=1)
        
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
avg_acc = total_correct / total_samples
print("Average accuracy is",avg_acc) 