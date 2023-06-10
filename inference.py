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
from dataset import PupilDataSet
import argparse
from Unet import Unet
import torch.nn.functional as F
from postprocess import draw_segmentation_map, connected_components


myseed = 777
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Choose which model to use (deeplabv3 or unet).", type=str, default="deeplabv3")
    parser.add_argument("--result_dir", help="Path to result directory.", type=str, default="./solution/")
    parser.add_argument("--ckpt_path", help="Path to checkpoint.", type=str, required=True)
    parser.add_argument("--img_dir", help="Path to testing images directory (i.e., S5).", type=str, required=True)
    return parser


parser = get_parser()
args = parser.parse_args()

model_name = args.model
ckpt_path = args.ckpt_path
img_dir = args.img_dir
result_dir = args.result_dir

if not os.path.exists(result_dir):
    print("Creating result directory...")
    os.mkdir(result_dir)
subject = img_dir.split("/")[-1]
if not os.path.exists(os.path.join(result_dir, subject)):
    os.mkdir(os.path.join(result_dir, subject))

subfolders = os.listdir(img_dir)
subfolders = natsorted(subfolders)

label_map = {0: [0, 0, 0], 1: [255, 255, 255]}

if model_name == "unet":
    model = Unet(n_channels=1, n_classes=2)
    model.load_state_dict(torch.load(ckpt_path))
    model = model.to(device)
elif model_name == "deeplabv3":
    model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True)
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
    model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load(ckpt_path))
    model = model.to(device)

model.eval()
valid_transform = transforms.Compose(
    [
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

for sub in subfolders:
    if not os.path.exists(os.path.join(result_dir, subject, sub)):
        print(f"Creating subfolder {sub} in result directory...")
        os.mkdir(os.path.join(result_dir, subject, sub))
    data = glob.glob(os.path.join(img_dir, sub, "*.jpg"), recursive=True)
    test_dataset = PupilDataSet(data, transform=valid_transform, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for i, img in enumerate(test_loader):
            print(f"Processing {i}.jpg ...")
            img = img.to(device)

            if model_name == "unet":
                output = model(img)
            elif model_name == "deeplabv3":
                output = model(img)["out"]

            output = F.softmax(output, dim=1).float()
            pred = torch.argmax(output.squeeze().cpu(), dim=0).numpy()
            cleaned_pred = connected_components(pred, threshold=1500)
            mask = draw_segmentation_map(cleaned_pred, label_map)

            if 1 in pred:
                conf = 1
                # mask = find_pupil(mask)
            else:
                conf = 0

            conf_path = os.path.join(result_dir, subject, sub, "conf.txt")
            with open(conf_path, "a") as file:
                file.write(str(conf) + "\n")
            fig = Image.fromarray(mask)
            # fig = fig.resize((640, 480))
            fig_save_path = os.path.join(result_dir, subject, sub, str(i) + ".png")
            fig.save(fig_save_path)
