"""
.py created by: cesc
.py created on: 01/05/2020
.py created for: bytetrack => adaptation to work with 5 channels: rgb, d, i
in here we will create the dataset
"""

import os

import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


# create custom dataset pytorch, that loads the images from the folder. images are rgb, d and I
class AppleCrops(Dataset):
    def __init__(self, root_path, split, transform=None):
        self.root_path = root_path
        self.split = split
        self.transform = transform
        self.files = pd.read_csv(os.path.join(root_path, 'crops_info', 'crops_' + split + '.csv'), header=None)

        # get the last column, which is the label
        self.labels = self.files.iloc[:, -1].values
        # todo: labels vector should have consecutive numbers, starting from

        # get the three columns: the paths of the images rgb, d and I
        self.files = self.files.iloc[:, :-1].values

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # idx must be lower than __len__
        if idx >= len(self.files):
            raise IndexError('Index out of range')
        # load the image from the path
        img_names = self.files[idx]

        # imgs comes in pairs of 3: rgb, d and I => stored in img_collapsed as a list
        # create empty numpy array for the images (5 channels)
        # read rgb image
        img_path = os.path.join(self.root_path, 'crops_of_Apple_Tracking_db', self.split, img_names[0])
        img = Image.open(img_path)
        img = np.array(img)

        # stack the depth and infrared channels
        for i in [1, 2]:
            img_path = os.path.join(self.root_path, 'crops_of_Apple_Tracking_db', self.split, img_names[i])
            img_sensor = Image.open(img_path)
            img_sensor = np.array(img_sensor)
            img_sensor = img_sensor[:, :, np.newaxis]
            img = np.append(img, img_sensor, axis=-1)

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]


if __name__ == "__main__":
    # to test the datasets if they are working
    db = AppleCrops(root_path='../../data',
                       split='train',
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((32, 32)),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406, 0.5, 0.5],
                                std=[0.229, 0.224, 0.225, 0.5, 0.5]),
                       ]))
    db.__getitem__(543)
    print('finsihed')
