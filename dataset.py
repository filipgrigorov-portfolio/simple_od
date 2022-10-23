import imageio
import glob
import numpy as np
import os
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DEBUG = True

class SimpleDataset(Dataset):
    def __init__(self, root_path, data_transforms=None):
        self.df = pd.read_csv(os.path.join(root_path, 'label.csv'))
        # Note: Load images in-memory (for this exercise)
        self.image_names = list(glob.glob(os.path.join(root_path, 'images/') + '*.png'))
        self.images = [ imageio.imread(img_name) for img_name in self.image_names ]
        if DEBUG:
            print(f'First 3 entries in labels.csv:\n\n{self.df.head(3)}')
        self.data_transforms = data_transforms
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img = self.images[idx]
        found_matches = self.df[self.df.img_name == img_name.split('/')[-1]]
        bboxes = []
        labels = []
        for row in found_matches.values:
            bboxes.append([ row[2], row[3], row[4], row[5] ])
            labels.append(row[1])
        img = transforms.ToPILImage()(img)
        if self.data_transforms:
            img_tensor = self.data_transforms(img)
        else:
            img_tensor = transforms.ToTensor()(img)
        lbl_tensor = torch.tensor(labels)
        bbox_tensor = torch.FloatTensor(bboxes)
        return img_tensor, lbl_tensor, bbox_tensor
    
    def __len__(self):
        return len(self.df.index)
