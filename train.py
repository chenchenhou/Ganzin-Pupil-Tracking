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
    parser.add_argument("model", help="Choose which model to use (deeplabv3 or unet).", type=str, required=True)
    parser.add_argument("save_dir", help="Path to checkpoint directory.", type=str, required=True)
    return parser


parser = get_parser()
args = parser.parse_args()
model_name = args.model
save_path = args.save_dir

S1 = glob.glob("dataset/S1/**/*.png", recursive=True) + glob.glob("dataset/S1/**/*.jpg", recursive=True)
S2 = glob.glob("dataset/S2/**/*.png", recursive=True) + glob.glob("dataset/S2/**/*.jpg", recursive=True)
S3 = glob.glob("dataset/S3/**/*.png", recursive=True) + glob.glob("dataset/S3/**/*.jpg", recursive=True)
S4 = glob.glob("dataset/S4/**/*.png", recursive=True) + glob.glob("dataset/S4/**/*.jpg", recursive=True)
dataWithGT = S1 + S2 + S3 + S4

# These transforms are meant for deeplabv3 in torchvision.models, feel free to modify them.
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
valid_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
transform_label = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
pupil_train_data = PupilDataSetwithGT(dataWithGT, transform=train_transform, transform_label=transform_label)
pupil_valid_data = PupilDataSetwithGT(dataWithGT, transform=valid_transform, transform_label=transform_label, mode="val")

"""# Configuration"""

config = {"num_epochs": 30, "lr": 0.001, "batch_size": 4, "save_path": save_path}

pupil_trainloader = DataLoader(pupil_train_data, batch_size=config["batch_size"], shuffle=True, drop_last=True)
pupil_validloader = DataLoader(pupil_valid_data, batch_size=config["batch_size"], shuffle=False, drop_last=True)

if model_name == "unet":
    model = Unet()
    model = model.to(device)
elif model_name == "deeplabv3":
    model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True)
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.2, 0.8]).to(device))

wandb.init(project="Ganzin Pupil Tracking")

for epoch in tqdm(range(1, config["num_epochs"] + 1)):
    model.train()
    print(f"Epoch {epoch}/{config['num_epochs']}")
    train_loss = []
    val_loss = []
    for img, label in tqdm(pupil_trainloader, desc="Training"):
        img = img.to(device)
        label = label.to(device)
        if model_name == "unet":
            output = model(img)
        elif model_name == "deeplabv3":
            output = model(img)["out"]
        loss = criterion(output, torch.squeeze(label.long()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    train_avg_loss = sum(train_loss) / len(train_loss)
    print(f"Training Loss = {train_avg_loss}")
    wandb.log({"Training Loss": train_avg_loss})

    model.eval()
    with torch.no_grad():
        for img, label in tqdm(pupil_trainloader, desc="Validation"):
            img = img.to(device)
            label = label.to(device)
            if model_name == "unet":
                output = model(img)
            elif model_name == "deeplabv3":
                output = model(img)["out"]
            loss = criterion(output, torch.squeeze(label.long()))
            val_loss.append(loss)
    val_avg_loss = sum(val_loss) / len(val_loss)
    print(f"Validation Loss = {val_avg_loss}")
    wandb.log({"Validation Loss": val_avg_loss})
    path_name = f"epoch{epoch}.pth"
    if os.path.exists(config["save_path"]) == False:
        print("Creating checkpoints directory...")
        os.mkdir(config["save_path"])
    if epoch % 10 == 0:
        print(f"Saving {path_name}...")
        torch.save(model.state_dict(), os.path.join(config["save_path"], path_name))
        if os.path.exists(os.path.join(config["save_path"], path_name)):
            print(f"Checkpoint successfully saved!")
        else:
            print(f"Failed to save the model...")

wandb.finish()
