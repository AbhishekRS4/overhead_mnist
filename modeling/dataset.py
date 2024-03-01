import os
import torch
import torchvision
import numpy as np
from PIL import Image
import torch.nn as nn
from skimage.io import imread
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class OverheadMNISTDataset(Dataset):
    def __init__(self, image_ids, labels, dir_images, is_train_set=True):
        """
        ---------
        Arguments
        ---------
        image_ids: list
            a list of strings indicating image files
        labels: list
            a list of labels corresponding to the list of image files
        dir_images: str
            full path to directory containing images
        is_train_set: bool
            indicating whether the instance is for train set or not (default: True)
        """
        self.image_ids = image_ids
        self.labels = labels
        self.dir_images = dir_images
        self.transform = None
        self.is_train_set = is_train_set

        if self.is_train_set:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation((90, 90)),
                    transforms.RandomAffine(
                        degrees=(-5, 5),
                        translate=(0, 0.05),
                        scale=(0.9, 1.05),
                        shear=(-5, 5),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        fill=170,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(0, 1),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(0, 1),
                ]
            )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        file_image = os.path.join(self.dir_images, self.image_ids[idx])
        image = imread(file_image)
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


def split_dataset(train_x, train_y, random_state=4):
    """
    ---------
    Arguments
    ---------
    train_x: list
        a list of train image files
    train_y: list
        a list of labels corresponding to train image files
    random_state: int
        random state to be used for split (default: 4)

    -------
    Returns
    -------
    (train_x, validation_x, train_y, validation_y) : n-tuple
        a n-tuple of training and validation image files and their corresponding labels
    """
    train_x, validation_x, train_y, validation_y = train_test_split(
        train_x, train_y, test_size=0.05, random_state=random_state
    )
    return train_x, validation_x, train_y, validation_y


def get_dataloaders_for_training(
    train_x, train_y, validation_x, validation_y, dir_images, batch_size=64
):
    """
    ---------
    Arguments
    ---------
    train_x: list
        a list of train image files
    train_y: list
        a list of labels corresponding to train image files
    validation_x: list
        a list of validation image files
    validation_y: list
        a list of labels corresponding to validation image files
    dir_images: str
        full path to directory containing the images
    batch_size: int
        batch size to be used for training and validation (default: 64)

    -------
    Returns
    -------
    (train_loader, validation_loader) : tuple
        a tuple of objects for training and validation dataset loaders
    """
    train_dataset = OverheadMNISTDataset(
        train_x, train_y, dir_images=dir_images, is_train_set=True
    )
    validation_dataset = OverheadMNISTDataset(
        validation_x, validation_y, dir_images=dir_images, is_train_set=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    return train_loader, validation_loader


def get_dataloader_for_testing(test_x, test_y, dir_images, batch_size=1):
    """
    ---------
    Arguments
    ---------
    test_x: list
        a list of test image files
    test_y: list
        a list of labels corresponding to test image files
    dir_images: str
        full path to directory containing the images
    batch_size: int
        batch size to be used for testing (default: 1)

    -------
    Returns
    -------
    test_loader: object
        an object for test dataset loader
    """
    test_dataset = OverheadMNISTDataset(
        test_x, test_y, dir_images=dir_images, is_train_set=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    return test_loader
