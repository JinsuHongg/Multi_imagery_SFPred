import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms

class SolarFlSets(Dataset):
    def __init__(self, annotations_df, img_dir, channel = False, num_sample = False, random_state=1004, transform=None, target_transform=None, normalization=False):
        
        if num_sample:
            self.img_labels = annotations_df.sample(n=num_sample, random_state=random_state) # random sample
        self.channel = channel
        self.img_dir = img_dir
        self.transform = transform
        # self.rgb = transforms.Grayscale(num_output_channels=3)
        self.target_transform = target_transform
        self.norm = normalization

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        
        # deploy channel if necessary
        if self.channel:
            image = read_image(img_path).float()[[self.channel],:,:] # 0:HMI, 1:GONG, 2:EUV304
            image = image.repeat(3,1,1) # create grayscale to 3 channel image with same values.
        else:
            image = read_image(img_path).float()

        label = self.img_labels.iloc[idx, 2] # 1: GOES class 2: magnitude of GOES class
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.norm:
            image = image / 255 # zero to one normalization
        return image, label