import os
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
from postprocess import remove_eyebrow
import cv2


class PupilDataSet(Dataset):
    def __init__(self, data, transform=None, transform_label=None, mode="train"):
        self.data = data
        self.transform = transform
        self.transform_label = transform_label
        self.mode = mode
        self.labels = [im for im in self.data if im.endswith("png")]
        self.labels = natsorted(self.labels)
        self.images = [im for im in self.data if im.endswith("jpg")]
        self.images = natsorted(self.images)
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
        img = Image.open(self.images[idx]).convert("L")
        img = self.transform(img)
        if self.mode == "test":
            # img_name = self.images[idx].split("/")[-1]
            return img

        label = Image.open(self.labels[idx]).convert("L")
        label = np.array(label)
        label = (label > 0).astype(np.uint8)
        label = self.transform_label(label)
        label = (label > 0).to(torch.int)
        return img, label
