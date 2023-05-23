import os
import cv2
import glob
import torch
import wandb
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
from dataset import PupilDataSetwithGT
import argparse
from Unet import Unet

myseed = 777
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Choose which model to use (deeplabv3 or unet).", type=str, default="unet")
    parser.add_argument("ckpt_path", help="Path to checkpoint.", type=str, required=True)
    parser.add_argument("img_path", help="Path to testing img (only supports single image for now).", type=str, required=True)
    return parser


parser = get_parser()
args = parser.parse_args()

model_name = args.model
ckpt_path = args.ckpt_path
img_path = args.img_path

if model_name == "unet":
    model = Unet()
    model.load_state_dict(torch.load(ckpt_path))
    model = model.to(device)
elif model_name == "deeplabv3":
    model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True)
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load(ckpt_path))
    model = model.to(device)

valid_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

img = Image.open(img_path).convert("RGB")
img = valid_transform(img)
img = img.to(device)
img = img.unsqueeze(0)

if model_name == "unet":
    output = model(img)
elif model_name == "deeplabv3":
    output = model(img)["out"]

label_map = {0: [0, 0, 0], 1: [255, 255, 255]}


def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze().cpu(), dim=0).numpy()
    print(labels)
    # Create 3 Numpy arrays containing zeros.
    # Later each pixel will be filled with respective red, green, and blue pixels
    # depending on the predicted class.

    R_map = np.zeros_like(labels).astype(np.uint8)
    G_map = np.zeros_like(labels).astype(np.uint8)
    B_map = np.zeros_like(labels).astype(np.uint8)
    for label_num in range(0, len(label_map.keys())):
        index = labels == label_num
        R_map[index] = label_map[label_num][0]
        G_map[index] = label_map[label_num][1]
        B_map[index] = label_map[label_num][2]

    segmentation_map = np.stack([R_map, G_map, B_map], axis=2)
    return segmentation_map


mask = draw_segmentation_map(output)

fig = Image.fromarray(mask)
fig.save("./test.png")
