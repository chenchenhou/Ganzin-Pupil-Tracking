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
from postprocess import draw_segmentation_map, connected_components, gamma_correction


myseed = 777
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", help="Path to result directory.", type=str, default="./solution/")
    parser.add_argument("--use_gamma", help="Whether to use gamma correction on input image", action="store_true")
    parser.add_argument("--area_ratio", help="ratio of average area of a sub-directory used as the threshold for connected component", type=float, default=0.3)
    parser.add_argument("--ckpt_path", help="Path to checkpoint.", type=str, required=True)
    parser.add_argument("--img_dir", help="Path to testing images directory (i.e., S5).", type=str, required=True)
    return parser


parser = get_parser()
args = parser.parse_args()


ckpt_path = args.ckpt_path
img_dir = args.img_dir
result_dir = args.result_dir
use_gamma = args.use_gamma
area_ratio = args.area_ratio

if not os.path.exists(result_dir):
    print("Creating result directory...")
    os.mkdir(result_dir)
subject = img_dir.split("/")[-1]
if not os.path.exists(os.path.join(result_dir, subject)):
    os.mkdir(os.path.join(result_dir, subject))

subfolders = os.listdir(img_dir)
subfolders = natsorted(subfolders)

label_map = {0: [0, 0, 0], 1: [255, 255, 255]}


model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet101", pretrained=True)
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
    areas = []
    preds = []
    with torch.no_grad():
        for i, img in enumerate(test_loader):
            print(f"Processing {i}.jpg ...")

            # enhance image
            if use_gamma:
                print("Enhancing input image with gamma correction...")
                img = img.detach().cpu().numpy()
                img = gamma_correction(img)
                img = torch.from_numpy(img)

            img = img.to(device)

            output = model(img)["out"]

            output = F.softmax(output, dim=1).float()
            pred = torch.argmax(output.squeeze().cpu(), dim=0).numpy()
            area = np.sum(pred)
            areas.append(area)
            preds.append(pred)
    avg_area = sum(areas) / len(areas)
    avg_area = area_ratio * avg_area
    for i, p in enumerate(preds):
        print(f"Finetuning {i}.jpg with threshold area = {avg_area}")
        cleaned_pred = connected_components(p, threshold=int(avg_area))
        mask = draw_segmentation_map(cleaned_pred, label_map)

        if 1 in cleaned_pred:
            conf = 1
        else:
            conf = 0

        conf_path = os.path.join(result_dir, subject, sub, "conf.txt")
        with open(conf_path, "a") as file:
            file.write(str(conf) + "\n")
        fig = Image.fromarray(mask)
        fig_save_path = os.path.join(result_dir, subject, sub, str(i) + ".png")
        fig.save(fig_save_path)
