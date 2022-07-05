"""
.py created by: cesc
.py created on: 01/05/2020
.py created for: bytetrack => adaptation to work with 5 channels: rgb, d, i
in here we will create the dataset
"""
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
import numpy as np
import torch

# create custom dataset pytorch, that loads the images from the folder. images are rgb, d and I
class AppleCrops(Dataset):
    def __init__(self, root_path, split, transform=None):
        self.root_path = root_path
        self.split = split
        self.transform = transform

        # load pickle file it is a list of lists
        with open(os.path.join(root_path, 'crops_of_Apple_Tracking_db_numpy', f'{split}_crops.pkl'), 'rb') as f:
            self.files = pickle.load(f)

        for idx_img, (img, idx) in enumerate(self.files):
            # transpose: (a, b, c) => (c, b, a)
            img = img.transpose()
            self.files[idx_img] = (img, idx)

        # depth and ir => 0 to 1 (normalized) => 0 to 255 (to make it into the same dynamic range as the rgb images)
        for idx, file in enumerate(self.files):
            self.files[idx][0][:, :, 3] *= 255
            self.files[idx][0][:, :, 4] *= 255

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # idx must be lower than __len__
        if idx >= len(self.files):
            raise IndexError('Index out of range')

        # load the image and the label
        img, label = self.files[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def mean_and_std_calculator(root_path='../../data'):
    """
    This function computes the mean and std of the dataset for both splits, train and test
    It is used to normalize the images.
    """
    # create custom dataset pytorch, that loads the images from the folder. images are rgb, d and I. Only apply the
    # resize transformation.
    db_train = AppleCrops(root_path=root_path,
                          split='train',
                          transform=None)
    db_test = AppleCrops(root_path=root_path,
                          split='test',
                          transform=None)

    mean = 0
    std = 0
    print('computing mean and std for train set...')
    for i in tqdm(range(len(db_train))):
        img, _ = db_train[i]
        resized = cv2.resize(img, (32, 32))
        mean += resized.mean(axis=(0, 1))
        std += resized.std(axis=(0, 1))

    print('computing mean and std for test set...')
    for i in range(len(db_test)):
        img, _ = db_test[i]
        resized = cv2.resize(img, (32, 32))
        mean += resized.mean(axis=(0, 1))
        std += resized.std(axis=(0, 1))

    print('putting mean and std from splits together...')
    mean /= (len(db_train) + len(db_test))
    std /= (len(db_train) + len(db_test))

    return mean, std


def show_image(idx, dataset):
    """
    This function shows an image (item from the dataset)
    :param idx: index of the image to show
    :param dataset: dataset to show the image from
    function works if param transform of the dataset is None
    """
    img, _ = dataset.__getitem__(idx)
    # convert img to int8 to show it in matplotlib
    img = img.astype(np.uint8)
    # create a subplot of 3 image
    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    # show the images, not with blue axis
    img_c = img[:, :, :3]
    image = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)

    ax[0].imshow(image)
    ax[1].imshow(img[:, :, 3])
    ax[2].imshow(img[:, :, 4])
    # show the labels
    ax[0].set_title('rgb')
    ax[1].set_title('d')
    ax[2].set_title('i')
    plt.show()


# create custom dataset pytorch, that loads the images from the folder. images are rgb, d and I => Triplet
class AppleCropsTriplet(Dataset):
    def __init__(self, root_path, split, transform=None):
        self.root_path = root_path
        self.split = split
        self.transform = transform

        # load pickle file it is a list of lists
        with open(os.path.join(root_path, 'crops_of_Apple_Tracking_db_numpy', f'{split}_crops.pkl'), 'rb') as f:
            self.files = pickle.load(f)

        for idx_img, (img, idx) in enumerate(self.files):
            # transpose: (a, b, c) => (c, b, a)
            img = img.transpose()
            self.files[idx_img] = (img, idx)

        # depth and ir => 0 to 1 (normalized) => 0 to 255 (to make it into the same dynamic range as the rgb images)
        for idx, file in enumerate(self.files):
            self.files[idx][0][:, :, 3] *= 255
            self.files[idx][0][:, :, 4] *= 255

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # idx must be lower than __len__
        if idx >= len(self.files):
            raise IndexError('Index out of range')

        # load the anchor image and the label
        anchor_img, label_anchor = self.files[idx]
        # select an image in files that is not the anchor image but has the same label
        while True:
            """
            # acotate the search to make it faster!
            search_idx_max = idx + 100
            if search_idx_max >= len(self.files):
                search_idx_max = len(self.files) - 1
            search_idx_min = idx - 100
            if search_idx_min < 0:
                search_idx_min = 0

            idx_img = np.random.randint(search_idx_min, search_idx_max)
            """
            idx_img = np.random.randint(0, len(self.files))
            # if the apple has only one crop => problem
            # if idx_img != idx:
            positive_img, label_pos = self.files[idx_img]
            if label_pos == label_anchor:
                break

        # select an image in files that is not of the same label as the anchor and the positive image
        while True:
            idx_img = np.random.randint(0, len(self.files))
            if idx_img != idx:
                negative_img, label_neg = self.files[idx_img]
                if label_neg != label_anchor:
                    break

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        triplet_imgs = anchor_img, positive_img, negative_img

        return triplet_imgs, label_anchor


if __name__ == "__main__":
    # to test the datasets if they are working
    db = AppleCrops(root_path='../../data',
                    split='train',
                    transform=None)
    show_image(idx=15000, dataset=db)

    mean, std = mean_and_std_calculator(root_path='../../data')
    print(f'depth: mean:{mean[3]/255}, std:{std[3]/255}')
    print(f'ir: mean:{mean[4]/255}, std:{std[4]/255}')

    db_triplet = AppleCropsTriplet(root_path='../../data',
                                   split='train',
                                   transform=None)

    print('finsihed')
