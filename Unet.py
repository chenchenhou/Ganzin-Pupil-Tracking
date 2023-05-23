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
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
pupil_train_data = PupilDataSetwithGT(dataWithGT, transform=train_transform, transform_label=transform_label)
pupil_valid_data = PupilDataSetwithGT(dataWithGT, transform=valid_transform, transform_label=transform_label, mode="val")

"""# Configuration"""

config = {"num_epochs": 30, "lr": 0.001, "batch_size": 4, "save_path": "./checkpoints_unet/"}

pupil_trainloader = DataLoader(pupil_train_data, batch_size=config["batch_size"], shuffle=True)
pupil_validloader = DataLoader(pupil_valid_data, batch_size=config["batch_size"], shuffle=False)

"""# Training and Validation"""
# We can use different deeplabv3 architecture
class decoder_Unet(torch.nn.Module):
    def __init__(self,in_ch,middle_ch,out_ch):
        super(decoder_Unet,self).__init__()
        self.up = torch.nn.ConvTranspose2d(in_ch,out_ch,kernel_size=2,stride=2)
        self.conv_relu = torch.nn.Sequential(
        torch.nn.Conv2d(middle_ch,out_ch,kernel_size=1,stride=1),
        torch.nn.ReLU(),
    )
    def forward(self,x1,x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1,x2),dim=1)
        x1 = self.conv_relu(x1)
        return x1

class Unet(torch.nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.pretrain_model = models.vgg16(pretrained=True)
        features = list(self.pretrain_model.features.children())
        # 1st stage
        self.ConvEn1 = torch.nn.Sequential(*features[:4])
        self.maxpool1 = torch.nn.Sequential(features[4])
    
        # 2nd stage
        self.ConvEn2 = torch.nn.Sequential(*features[5:9])
        self.maxpool2 = torch.nn.Sequential(features[9])

        # 3rd stage
        self.ConvEn3 = torch.nn.Sequential(*features[10:16])
        self.maxpool3 = torch.nn.Sequential(features[16])

        # 4th stage
        self.ConvEn4 = torch.nn.Sequential(*features[17:23])
        self.maxpool4 = torch.nn.Sequential(features[23])

        # 5th stage
        self.ConvEn5 = torch.nn.Sequential(*features[24:30])
        self.maxpool5 = torch.nn.Sequential(features[30])
        # decoder
        self.decoder1 = decoder_Unet(512,512+512,512)
        self.decoder2 = decoder_Unet(512,256+256,256)
        self.decoder3 = decoder_Unet(256,128+128,128)
        self.decoder4 = decoder_Unet(128,64+64,64)
        self.decoder5 = torch.nn.Sequential(
        torch.nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
        torch.nn.Conv2d(64,32,3,padding=1),
        torch.nn.Conv2d(32,64,3,padding=1),

        )
        self.classifier = torch.nn.Conv2d(64,2,1)
    def forward(self,x):
        # 1st encode stage
        encoder1 = self.ConvEn1(x)
        maxpool1= self.maxpool1(encoder1)
       # print(maxpool1.shape)

        # 2nd encode stage
        encoder2 = self.ConvEn2(maxpool1)
        maxpool2 = self.maxpool2(encoder2)
        #print(maxpool2.shape)

        # 3rd encode stage
        encoder3 = self.ConvEn3(maxpool2)
        maxpool3 = self.maxpool3(encoder3)
        #print(maxpool3.shape)

        # 4th encode stage
        encoder4 = self.ConvEn4(maxpool3)
        maxpool4 = self.maxpool4(encoder4)
        #print(maxpool4.shape)

        # 5th encode stage
        encoder5 = self.ConvEn5(maxpool4)
        maxpool5 = self.maxpool5(encoder5)
        #print(maxpool5.shape)

        # decode
        decode1 = self.decoder1(maxpool5,maxpool4)
        #print(decode1.shape)
        decode2 = self.decoder2(decode1,maxpool3)
        #print(decode2.shape)
        decode3 = self.decoder3(decode2,maxpool2)
        #print(decode3.shape)
        decode4 = self.decoder4(decode3,maxpool1)
        #print(decode4.shape)
        decode5 = self.decoder5(decode4)
        #print(decode5.shape)
        result = self.classifier(decode5)
        #print(result.shape)
        return result
unet = Unet()
optimizer = torch.optim.Adam(unet.parameters(), lr=config["lr"])
criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.2, 0.8]).to(device))

wandb.init(project="Ganzin Pupil Tracking")

for epoch in tqdm(range(1, config["num_epochs"] + 1)):
    unet.train()
    print(f"Epoch {epoch}/{config['num_epochs']}")
    train_loss = []
    val_loss = []
    for img, label in tqdm(pupil_trainloader, desc="Training"):
        img = img.to(device)
        label = label.to(device)
        output = unet(img)
        loss = criterion(output, torch.squeeze(label.long()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    train_avg_loss = sum(train_loss) / len(train_loss)
    print(f"Training Loss = {train_avg_loss}")
    wandb.log({"Training Loss": train_avg_loss})

    unet.eval()
    with torch.no_grad():
        for img, label in tqdm(pupil_trainloader, desc="Validation"):
            img = img.to(device)
            label = label.to(device)
            output = unet(img)
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
        torch.save(unet.state_dict(), os.path.join(config["save_path"], path_name))

wandb.finish()