import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from natsort import natsorted
import glob
import cv2
from tqdm import tqdm
import matplotlib



class PupilDataSetwithGT(Dataset):
    def __init__(self, data, transform = None, transform_label = None, mode = 'train'):
        self.data = natsorted(data)
        self.transform = transform
        self.transform_label = transform_label
        self.mode = mode
        self.labels = [im for im in self.data if im.endswith('png')]
        self.images = [im for im in self.data if im.endswith('jpg')]

        # This is only a naive way to separate training images and validation images, feel free to modify it.
        sep = 5
        if self.mode == 'train':
            self.labels = [self.labels[i] for i in range(len(self.labels)) if i % sep != 0]
            self.images = [self.images[i] for i in range(len(self.images)) if i % sep != 0]
        elif self.mode == 'val':
            self.labels = [self.labels[i] for i in range(len(self.labels)) if i % sep == 0]
            self.images = [self.images[i] for i in range(len(self.images)) if i % sep == 0]


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        img = self.transform(img)
        if self.mode == 'test':
            label = -1
        else:
            label = Image.open(self.labels[idx]).convert('RGB')
            label = self.transform_label(label)
        return img, label



