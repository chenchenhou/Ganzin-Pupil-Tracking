import os
import cv2
import glob
import torch
import matplotlib
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from natsort import natsorted

def read_masks(file_list):
    '''
    Read masks from directory and tranform to categorical
    '''
    n_masks = len(file_list)
    masks = np.empty((n_masks, 480, 640))

    for i, file in enumerate(file_list):
        mask = Image.open(file).convert('L')
        mask = np.array(mask)
        mask = (mask > 0).astype(int)
        masks[i] = mask

    return masks

class PupilDataSetwithGT(Dataset):
    def __init__(self, data, transform=None, transform_label=None, mode="train"):
        self.data = natsorted(data)
        self.transform = transform
        self.transform_label = transform_label
        self.mode = mode
        self.labels = [im for im in self.data if im.endswith('png')]
        self.images = [im for im in self.data if im.endswith('jpg')]
        self.labels = read_masks(self.labels)
        # This is only a naive way to separate training images and validation images, feel free to modify it.
        sep = 5
        if self.mode == "train":
            self.labels = [self.labels[i] for i in range(len(self.labels)) if i % sep != 0]
            self.images = [self.images[i] for i in range(len(self.images)) if i % sep != 0]
        elif self.mode == "val":
            self.labels = [self.labels[i] for i in range(len(self.labels)) if i % sep == 0]
            self.images = [self.images[i] for i in range(len(self.images)) if i % sep == 0]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = self.transform(img)
        if self.mode == "test":
            label = -1
        else:
            label = Image.open(self.labels[idx]).convert('L')
            label = self.transform_label(label)
        return img, label
