from transformers import AutoImageProcessor
from transformers import  AutoModelForImageClassification
from Vit import ViT
from model_deformConv import CustomDeformViT
from model_dilatedConv import CustomDilatedViT
from model_addCNN import EnhancedViTForImageClassification
from model_VPTshallow import ShallowVPTViTForImageClassification
from model_VPTdeep import DeepVPTViTForImageClassification
from model_depth import DeepViT
from adience_data_loader import CustomImageDataset,ImageClassificationCollator
from Dataset.utk_data_loader import UTKFaceDataset
import argparse
from sklearn.model_selection import train_test_split

import json
feature_extractor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification")

from pathlib import  Path
import math
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, Subset
import torch
from torchvision import transforms
from Dataset.utk_data_loader import UTKFaceDataset
from celebA_data_loader import CelebAGenderDataset
from utils import tester
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir",     type=str, required=True)
parser.add_argument("--attr_folder",   type=str, required=True)
parser.add_argument("--dataset",  type=str, required=True)
#which dataset to use for testing
parser.add_argument("--start",  type=str, required=True) 
#the checkpoint to start from
parser.add_argument("--model_name",  type=str, required=True) 
#the model type to use
parser.add_argument("--backbone",  type=str, required=True) 
#the backbone model location "online","local"
parser.add_argument("--address",  type=str, required=True) 
#the address of the backbone
parser.add_argument("--batch_size",  type=int, default=64)
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()
print(json.dumps(vars(args), indent=4))
TEST_RATIO = 0.2   # 20% as test set
SEED       = 42    # fixed seed for reproducibility

#use model.model to extract the ViTForImageClassification model from class
model_classes = {
    "ViT":          ViT,
    "deformConv":   CustomDeformViT,
    "dilatedConv":  CustomDilatedViT,
    "addCNN":       EnhancedViTForImageClassification,
    "VPTshallow":   ShallowVPTViTForImageClassification,
    "VPTdeep":      DeepVPTViTForImageClassification,
    "depth":        DeepViT,
    "online":       None,   
}

if args.model_name not in model_classes:
    raise ValueError(f"Unknown model name: {args.model_name}")

cls = model_classes[args.model_name]
if args.model_name == "online":
    model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")
else:
    if(args.backbone!="None"):
        model = cls.load_from_checkpoint(
            args.start,
            epochs=4,
            num_classes=2,
            model_name=args.address,
            config=args.backbone
        )
    else:
        model = cls.load_from_checkpoint(
            args.start,
            epochs=4,
            num_classes=2
        )
"""if(args.model_name=="ViT"): 
    model=ViT.load_from_checkpoint(args.start,epochs=4,num_classes=2,config=config)
elif(args.model_name=="deformConv"):
    model=CustomDeformViT.load_from_checkpoint(args.start)
elif(args.model_name=="dilatedConv"):
    model=CustomDilatedViT.load_from_checkpoint(args.start)
elif(args.model_name=="addCNN"):
    model=EnhancedViTForImageClassification.load_from_checkpoint(args.start,
    epochs=4,num_classes=2,config=config)
elif(args.model_name=="VPTshallow"):
    model=ShallowVPTViTForImageClassification.load_from_checkpoint(args.start,
    epochs=4,num_classes=2,config=config)
elif(args.model_name=="VPTdeep"):
    model=DeepVPTViTForImageClassification.load_from_checkpoint(args.start,
    epochs=4,num_classes=2,config=config)
elif(args.model_name=="depth"):
    model=DeepViT.load_from_checkpoint(args.start,epochs=4,num_classes=2,config=config)
elif(args.model_name=="online"):
    model=AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")"""
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
    test_ds=CustomImageDataset(img_dir=args.img_dir,
                          txt_file=paths[0],transform=val_tf)
elif(args.dataset=="UTK"):
    full_ds = UTKFaceDataset(img_dir=args.img_dir, align_with_adience=True)
    test_ds=full_ds
    # fixed 20% split — same split every run due to fixed seed
    """indices = list(range(len(full_ds)))
    _, test_indices = train_test_split(
        indices, test_size=TEST_RATIO, random_state=SEED
    )
    test_ds = Subset(full_ds, test_indices)"""
elif(args.dataset=="celebA"):
    test_ds=CelebAGenderDataset(args.img_dir, args.attr_file, args.split_file, split=2, transform=val_tf)
collator = ImageClassificationCollator(feature_extractor)

test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collator, 
                         num_workers=args.num_workers, pin_memory=True)


                         

tester(model,test_loader)
"""avg_acc = total_correct / total_samples
print("Average accuracy is",avg_acc) """