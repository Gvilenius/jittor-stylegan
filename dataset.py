import os
import numpy as np
from PIL import Image
from jittor import transform as trans
from jittor.dataset.dataset import Dataset, dataset_root

## 由于不同数据集的目录组织不一样，就分开实现了

class ColorSymbol(Dataset):
    def __init__(self, data_root="color_symbol_7k/", train = True, batch_size=16, shuffle=False, transform=None):
        super().__init__()
        self.data_root = data_root
        self.is_train = train
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle =  False
        self.drop_last = False
        self.resolution = None
        self.data =  []
        for _, _, files in os.walk(data_root):
            for name in files:
                img_path = os.path.join(data_root, name)
                self.data.append(img_path);
        self.total_len = len(self.data)

    def __getitem__ (self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        img = img.resize((self.resolution, self.resolution))
        if self.transform:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.data)

class FFHQ(Dataset):
    def __init__(self, data_root="/data1/ffhq", train = True, batch_size=16, shuffle=False, transform=None):
        super().__init__()
        self.data_root = data_root
        self.is_train = train
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle =  False
        self.drop_last = False
        self.resolution = None
        self.data =  []
        for d in os.listdir(data_root):
            p = os.path.join(data_root, d)
            for _, _, files in os.walk(p):
                for name in files:
                    img_path = os.path.join(p, name)
                    self.data.append(img_path);
        self.total_len = len(self.data)

    def __getitem__ (self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        img = img.resize((self.resolution, self.resolution))
        if self.transform:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.data)

