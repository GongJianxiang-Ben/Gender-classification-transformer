# Gender-classification-transformer
## Environment configuration - PyTorch 2.1.2 + CUDA 11.8

run the following code in terminal of the python environment

```bash

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118


pip install lightning


pip install torchinfo
pip install "transformers<4.40"
pip install torchmetrics
pip install pillow
pip install "numpy<2"
pip install scikit-learn```

## Vit experiment files
### test

run the following code

```python

python ./"gender classification"/Vit/Vit_model_loader.py \
  --img_dir ./crop_part1\ 
  --attr_folder ./"label txt" \
  --dataset UTK\
  --start ./checkpoints-768/addCNN/vit-best-08-0.0000.ckpt\
  --model_name addCNN\
  --backbone online\
  --address rizvandwiki/gender-classification\
  --batch_size 64 ```
or
```bash
sbatch train.sh```

need to change args

Required args:
  image_dir: the path to image folder of dataset
  attr_folder: the path to attribute folder of dataset (if no just enter ./)
  dataset: one of three dataset (UTK,adience,celebA)
  start: the path to the checkpoint to test on
  model_name: the model type to use (ViT, deformConv, dilatedConv, addCNN,VPTshallow, VPTdeep, depth, online)
  backbone: is the model based on some pretraining model(ViT only) (online, local, None)
  address:
    online: huggingface id
    local: path to checkpoint
    None: -
Choosible args:
  batch_size(default 64), seed(default 42), num_workers(for dataloader)(default 4)

###train

```python

python ./"gender classification"/Vit/Vit_deformConv.py \
  --img_dir ./celeba_data/img_align_celeba\
  --attr_file ./celeba_data/list_attr_celeba.csv \
  --split_file ./celeba_data/list_eval_partition.csv \
  --checkpoint ./checkpoints-768/addCNN \
  --dataset both\
  --start online \
  --import_from rizvandwiki/gender-classification\
  --epochs 50 \
  --dilated_size 3\
  --batch_size 64 \
  --lr 5e-4

or
```bash
sbatch loader.sh```

need to change args

Required args:
  image_dir: the path to image folder of dataset
  attr_folder: the path to attribute folder of dataset (if no just enter ./)
  split_file: the path to file that help split the dataset (if no just enter ./)
  checkpoint: the path to the folder to store checkpoints
  dataset: one of three types (both,adience,celebA)
  start:
    scratch: train from scratch
    online: use online model as backbone
    local: use local model as backbone (ViT only)
    path to checkpoints: continue training on previous checkpoints of same type
Choosible args:
  inport_from:
    start=="local": the path to local checkpoint
    start=="online": the id of huggingface model
  epochs(default 50),dilated_size(for dilatedConv only)(default 2), lr(default 1e-4), seed(default 42), num_workers(for dataloader)(default 4)
