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
print(len(dataWithGT))

# These transforms are meant for deeplabv3 in torchvision.models, feel free to modify them.
train_transform = transforms.Compose(
    [
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)
valid_transform = transforms.Compose(
    [
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)
transform_label = transforms.Compose(
    [
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
    ]
)
pupil_train_data = PupilDataSetwithGT(dataWithGT, transform=train_transform, transform_label=transform_label)
pupil_valid_data = PupilDataSetwithGT(dataWithGT, transform=valid_transform, transform_label=transform_label, mode="val")

"""# Configuration

"""

config = {"num_epochs": 1, "lr": 0.001, "batch_size": 4, "save_path": "./checkpoints/"}

pupil_trainloder = DataLoader(pupil_train_data, batch_size=config["batch_size"], shuffle=True)
pupil_validloader = DataLoader(pupil_valid_data, batch_size=config["batch_size"], shuffle=False)

"""# Training and Validation"""

deeplabv3 = models.segmentation.deeplabv3_resnet50(weights="DEFAULT", weights_backbone="ResNet50_Weights.DEFAULT")
deeplabv3.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
deeplabv3.classifier[-1] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
deeplabv3 = deeplabv3.to(device)

print(deeplabv3)

optimizer = torch.optim.Adam(deeplabv3.parameters(), lr=config["lr"])
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, config["num_epochs"] + 1):
    deeplabv3.train()
    print(f"Epoch {epoch}...")
    train_loss = []
    val_loss = []
    for img, label in pupil_trainloder:
        img = img.to(device)
        label = label.to(device)
        output = deeplabv3(img)["out"]
        pred = torch.argmax(output, dim=1)
        loss = criterion(label, pred)
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    train_avg_loss = sum(train_loss) / len(train_loss)
    print(f"Training loss = {train_avg_loss}")

    deeplabv3.eval()
    with torch.no_grad():
        for img, label in pupil_validloader:
            img = img.to(device)
            label = label.to(device)
            output = deeplabv3(img)["out"]
            pred = torch.argmax(output, dim=1)
            loss = criterion(label, pred)
            val_loss.append(loss)
    val_avg_loss = sum(val_loss) / len(val_loss)
    print(f"Validation loss = {val_avg_loss}")
    path_name = f"epoch{epoch}.pth"
    torch.save(deeplabv3.state_dict(), os.path.join(config["save_path"], path_name))
