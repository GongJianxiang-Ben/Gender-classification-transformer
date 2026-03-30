from torch.utils.data import Dataset
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, txt_file, transform=None):
        
        self.img_dir = img_dir
        self.transform = transform
        self.ages={range(0, 3):0,range(4, 7):1,range(8, 14):2,range(15, 21):3,range(25, 33):4,
              range(38, 44):5,range(48, 54):6,range(60, 101):7}
        self.genders={'f':0,'m':1}
        
        self.samples = []
        with open(txt_file, 'r') as f:
            next(f)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()          
                subdir_name=parts[0]
                img_name = "landmark_aligned_face."+parts[2]+"."+parts[1]
                if(parts[3]=="None"):
                    continue
                if(parts[4][-1]==')'):
                    age=int(parts[4][:-1])
                    gender=parts[5]
                else:
                    age=int(parts[3])
                    gender=parts[4]
                for k in list(self.ages.keys()):
                    if(age in k):
                        age=self.ages[k]
                        break
                if(gender not in ['f','m']):
                    continue
                img_path = os.path.join(img_dir,subdir_name, img_name)
                if os.path.exists(img_path):  # prevent file not exist
                    self.samples.append((img_path, age,self.genders[gender]))
        
       
        self.classes = [0,1]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"Dataset loaded! A total of {len(self.samples)} images, {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age,gender = self.samples[idx]
        
        # open image
        image = Image.open(img_path).convert('RGB')
        
        
        if self.transform:
            image = self.transform(image)
        
        return image,gender
class EmptyDataset(Dataset):
    def __init__(self):
        pass   
    
    def __len__(self):
        return 0
    
    def __getitem__(self, index):
        raise IndexError("Empty dataset cannot be indexed")
