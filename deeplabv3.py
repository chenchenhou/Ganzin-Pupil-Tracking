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


myseed = 777
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""# Extracting data and Preprocessing
## Note that only S1 ~ S4 are provided with ground truth. Ground truth images are files that end with *.png*.
"""

# Note that only S1 ~ S4 are provided with ground truth. Ground truth images are files that end with '.png'.
# Maybe we can think about how to make use of unlabeled data (i.e., pseudo label).
# For now, I only use those with gt.
# Replace the path with your own path
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
pupil_train_data = PupilDataSetwithGT(dataWithGT, transform=train_transform, transform_label=transform_label)
pupil_valid_data = PupilDataSetwithGT(dataWithGT, transform=valid_transform, transform_label=transform_label, mode="val")

"""# Configuration"""

config = {"num_epochs": 30, "lr": 0.001, "batch_size": 4, "save_path": "./checkpoints/"}

pupil_trainloader = DataLoader(pupil_train_data, batch_size=config["batch_size"], shuffle=True)
pupil_validloader = DataLoader(pupil_valid_data, batch_size=config["batch_size"], shuffle=False)

"""# Training and Validation"""
# We can use different deeplabv3 architecture
deeplabv3 = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True)
deeplabv3.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
deeplabv3 = deeplabv3.to(device)

# print(deeplabv3)

optimizer = torch.optim.Adam(deeplabv3.parameters(), lr=config["lr"])
criterion = torch.nn.CrossEntropyLoss()

wandb.init(project="Ganzin Pupil Tracking")

for epoch in range(1, config["num_epochs"] + 1):
    deeplabv3.train()
    print(f"Epoch {epoch}/{config['num_epochs']}")
    train_loss = []
    val_loss = []
    for img, label in pupil_trainloader:
        img = img.to(device)
        label = label.to(device)
        output = deeplabv3(img)["out"]
        loss = criterion(output, torch.squeeze(label.long()))
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    train_avg_loss = sum(train_loss) / len(train_loss)
    print(f"Training Loss = {train_avg_loss}")
    wandb.log({"Training Loss": train_avg_loss})

    deeplabv3.eval()
    with torch.no_grad():
        for img, label in pupil_validloader:
            img = img.to(device)
            label = label.to(device)
            output = deeplabv3(img)["out"]
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
        torch.save(deeplabv3.state_dict(), os.path.join(config["save_path"], path_name))

wandb.finish()
