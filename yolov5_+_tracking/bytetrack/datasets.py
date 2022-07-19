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
from tools.visualization import plot_hist
import torch


# create custom dataset pytorch, that loads the images from the folder. images are rgb, d and I
class AppleCrops(Dataset):
    """
    Custom class for the dataset, Apple crops are crops of the apple from the apple tracking dataset (rgb, d, i)
    """
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


class AppleCropsRGB(Dataset):
    """
    Custom class for the dataset, Apple crops are crops of the apple from the apple tracking dataset (rgb)
    """
    def __init__(self, root_path, split, transform=None):
        self.root_path = root_path
        self.split = split
        self.transform = transform

        # load pickle file it is a list of lists
        with open(os.path.join(root_path, 'crops_of_Apple_Tracking_db_numpy', f'{split}_crops_without_D_and_I.pkl'),
                  'rb') as f:
            self.files = pickle.load(f)

        for idx_img, (img, idx) in enumerate(self.files):
            # transpose: (a, b, c) => (c, b, a)
            img = img.transpose()
            self.files[idx_img] = (img, idx)

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
    :param root_path: path to the data
    :return: mean and std of the train and test dataset
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
    """
    Custom class for the dataset, Apple crops are crops of the apple from the apple tracking dataset (rgb, d, i), to
    use in the training of a triplet network.
    """
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

        # get the maximum id of the files
        ids = [idx for _, idx in self.files]
        self.max_id = max(ids)

        # regorganize crops by id
        self.files_by_id = []
        for idx in range(self.max_id + 1):
            self.files_by_id.append([])
        for idx, file in enumerate(self.files):
            self.files_by_id[file[1]].append(file[0])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # idx must be lower than __len__
        if idx >= len(self.files):
            raise IndexError('Index out of range')

        # load the anchor image and the label
        anchor_img, label_anchor = self.files[idx]
        # get the list of images with the same label as the anchor
        positive_imgs = self.files_by_id[label_anchor]
        # get the location of the anchor image in the list of images with the same label
        for idx, positive_img in enumerate(positive_imgs):
            if np.array_equal(positive_img, anchor_img):
                idx_anchor = idx
                break

        # -------------- to show an example, experiment -----------------
        plot_hist(idx_anchor, positive_imgs, plot_histogram_example=False)
        # -------------- to show an example, experiment -----------------

        # POSITIVE LOOP
        # select an image following a gaussian distribution with mean idx_anchor and std 1. if idx_anchor is chosen,
        # the next image is selected.
        while True:
            idx_img = int(np.random.normal(idx_anchor, 3))  # std of gaussian is 3 frames
            if 0 <= idx_img < len(positive_imgs) and idx_img != idx_anchor:
                break
        positive_img = positive_imgs[idx_img]

        # NEGATIVE LOOP
        while True:
            idx_negative = np.random.randint(0, len(self.files))
            negative_img, label_neg = self.files[idx_negative]
            if label_anchor != label_neg:
                break

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        triplet_imgs = anchor_img, positive_img, negative_img

        return triplet_imgs, label_anchor


# create custom dataset pytorch, that loads the images from the folder. images are rgb, d and I => Triplet
class AppleCropsTripletRGB(Dataset):
    """
    Custom class for the dataset, Apple crops are crops of the apple from the apple tracking dataset (rgb, d, i), to
    use in the training of a triplet network.
    """
    def __init__(self, root_path, split, transform=None):
        self.root_path = root_path
        self.split = split
        self.transform = transform

        # load pickle file it is a list of lists
        with open(os.path.join(root_path, 'crops_of_Apple_Tracking_db_numpy', f'{split}_crops_without_D_and_I.pkl'), 'rb') as f:
            self.files = pickle.load(f)

        for idx_img, (img, idx) in enumerate(self.files):
            # transpose: (a, b, c) => (c, b, a)
            img = img.transpose()
            self.files[idx_img] = (img, idx)

        # get the maximum id of the files
        ids = [idx for _, idx in self.files]
        self.max_id = max(ids)

        # regorganize crops by id
        self.files_by_id = []
        for idx in range(self.max_id + 1):
            self.files_by_id.append([])
        for idx, file in enumerate(self.files):
            self.files_by_id[file[1]].append(file[0])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # idx must be lower than __len__
        if idx >= len(self.files):
            raise IndexError('Index out of range')

        # load the anchor image and the label
        anchor_img, label_anchor = self.files[idx]
        # get the list of images with the same label as the anchor
        positive_imgs = self.files_by_id[label_anchor]
        # get the location of the anchor image in the list of images with the same label
        for idx, positive_img in enumerate(positive_imgs):
            if np.array_equal(positive_img, anchor_img):
                idx_anchor = idx
                break

        # -------------- to show an example, experiment -----------------
        plot_hist(idx_anchor, positive_imgs, plot_histogram_example=False)
        # -------------- to show an example, experiment -----------------

        # POSITIVE LOOP
        # select an image following a gaussian distribution with mean idx_anchor and std 1. if idx_anchor is chosen,
        # the next image is selected.
        while True:
            idx_img = int(np.random.normal(idx_anchor, 3))  # std of gaussian is 3 frames
            if 0 <= idx_img < len(positive_imgs) and idx_img != idx_anchor:
                break
        positive_img = positive_imgs[idx_img]

        # NEGATIVE LOOP
        while True:
            idx_negative = np.random.randint(0, len(self.files))
            negative_img, label_neg = self.files[idx_negative]
            if label_anchor != label_neg:
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
    db_triplet.__getitem__(idx=2000)
    db_rgb = AppleCropsRGB(root_path='../../data',
                                   split='train',
                                   transform=None)
    db_triplet_rgb = AppleCropsTripletRGB(root_path='../../data',
                                          split='train',
                                          transform=None)
    print('finsihed')
