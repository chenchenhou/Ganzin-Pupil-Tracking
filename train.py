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
from dice_loss import dice_loss
import torch.nn.functional as F
from postprocess import remove_eyebrow

myseed = 777
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Choose which model to use (deeplabv3 or unet).", type=str, default="unet")
    parser.add_argument("--save_dir", help="Path to checkpoint directory.", type=str, default="./checkpoints/")
    parser.add_argument("--num_epochs", help="Number of training epochs.", type=int, default=30)
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=16)
    parser.add_argument("--lr", help="Initial learning rate.", type=float, default=0.0001)
    return parser


parser = get_parser()
args = parser.parse_args()
model_name = args.model

S1 = glob.glob("dataset/S1/**/*.png", recursive=True) + glob.glob("dataset/S1/**/*.jpg", recursive=True)
S2 = glob.glob("dataset/S2/**/*.png", recursive=True) + glob.glob("dataset/S2/**/*.jpg", recursive=True)
S3 = glob.glob("dataset/S3/**/*.png", recursive=True) + glob.glob("dataset/S3/**/*.jpg", recursive=True)
S4 = glob.glob("dataset/S4/**/*.png", recursive=True) + glob.glob("dataset/S4/**/*.jpg", recursive=True)
dataWithGT = S1 + S2 + S3 + S4

# These transforms are meant for deeplabv3 in torchvision.models, feel free to modify them.
train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
valid_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
transform_label = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]
)
pupil_train_data = PupilDataSet(dataWithGT, transform=train_transform, transform_label=transform_label)
pupil_valid_data = PupilDataSet(dataWithGT, transform=valid_transform, transform_label=transform_label, mode="val")

"""# Configuration"""

config = {"num_epochs": args.num_epochs, "lr": args.lr, "batch_size": args.batch_size, "save_path": args.save_dir}

pupil_trainloader = DataLoader(pupil_train_data, batch_size=config["batch_size"], shuffle=True, drop_last=True)
pupil_validloader = DataLoader(pupil_valid_data, batch_size=config["batch_size"], shuffle=False, drop_last=True)

if model_name == "unet":
    model = Unet(n_channels=1, n_classes=2)
    model = model.to(device)
elif model_name == "deeplabv3":
    model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True)
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
    model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device)

optimizer = torch.optim.RMSprop(model.parameters(), lr=config["lr"])
criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.3, 0.7]).to(device))

wandb.init(project="Ganzin Pupil Tracking")

for epoch in tqdm(range(1, config["num_epochs"] + 1)):
    model.train()
    print(f"Epoch {epoch}/{config['num_epochs']}")
    train_loss = []
    val_loss = []
    for img, label in tqdm(pupil_trainloader, desc="Training"):
        img = img.to(device, memory_format=torch.channels_last)
        label = label.to(device, dtype=torch.long)
        if model_name == "unet":
            output = model(img)
        elif model_name == "deeplabv3":
            output = model(img)["out"]
        loss = criterion(output, torch.squeeze(label))
        loss = dice_loss(F.softmax(output, dim=1).float(), F.one_hot(label, 2).squeeze().permute(0, 3, 1, 2).float(), multiclass=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    train_avg_loss = sum(train_loss) / len(train_loss)
    print(f"Training Loss = {train_avg_loss}")
    wandb.log({"Training Loss": train_avg_loss})

    model.eval()
    with torch.no_grad():
        for img, label in tqdm(pupil_validloader, desc="Validation"):
            img = img.to(device, memory_format=torch.channels_last)
            label = label.to(device, dtype=torch.long)
            if model_name == "unet":
                output = model(img)
            elif model_name == "deeplabv3":
                output = model(img)["out"]
            loss = criterion(output, torch.squeeze(label))
            loss += dice_loss(F.softmax(output, dim=1).float(), F.one_hot(label, 2).squeeze().permute(0, 3, 1, 2).float(), multiclass=True)
            val_loss.append(loss.item())
    val_avg_loss = sum(val_loss) / len(val_loss)
    print(f"Validation Loss = {val_avg_loss}")
    wandb.log({"Validation Loss": val_avg_loss})

    path_name = f"epoch{epoch}.pth"
    if os.path.exists(config["save_path"]) == False:
        print("Creating checkpoints directory...")
        os.mkdir(config["save_path"])
    if epoch % 5 == 0:
        print(f"Saving {path_name}...")
        torch.save(model.state_dict(), os.path.join(config["save_path"], path_name))
        if os.path.exists(os.path.join(config["save_path"], path_name)):
            print(f"Checkpoint successfully saved!")
        else:
            print(f"Failed to save the model...")

wandb.finish()
