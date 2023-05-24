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
import torch.nn.functional as F


myseed = 777
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class decoder_Unet(torch.nn.Module):
#     def __init__(self, in_ch, middle_ch, out_ch):
#         super(decoder_Unet, self).__init__()
#         self.up = torch.nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
#         self.conv_relu = torch.nn.Sequential(
#             torch.nn.Conv2d(middle_ch, out_ch, kernel_size=1, stride=1),
#             torch.nn.ReLU(),
#         )

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x1 = torch.cat((x1, x2), dim=1)
#         x1 = self.conv_relu(x1)
#         return x1


# class Unet(torch.nn.Module):
#     def __init__(self):
#         super(Unet, self).__init__()
#         self.pretrain_model = models.vgg16(pretrained=False)
#         features = list(self.pretrain_model.features.children())
#         # 1st stage
#         self.ConvEn1 = torch.nn.Sequential(*features[:4])
#         self.maxpool1 = torch.nn.Sequential(features[4])

#         # 2nd stage
#         self.ConvEn2 = torch.nn.Sequential(*features[5:9])
#         self.maxpool2 = torch.nn.Sequential(features[9])

#         # 3rd stage
#         self.ConvEn3 = torch.nn.Sequential(*features[10:16])
#         self.maxpool3 = torch.nn.Sequential(features[16])

#         # 4th stage
#         self.ConvEn4 = torch.nn.Sequential(*features[17:23])
#         self.maxpool4 = torch.nn.Sequential(features[23])

#         # 5th stage
#         self.ConvEn5 = torch.nn.Sequential(*features[24:30])
#         self.maxpool5 = torch.nn.Sequential(features[30])
#         # decoder
#         self.decoder1 = decoder_Unet(512, 512 + 512, 512)
#         self.decoder2 = decoder_Unet(512, 256 + 256, 256)
#         self.decoder3 = decoder_Unet(256, 128 + 128, 128)
#         self.decoder4 = decoder_Unet(128, 64 + 64, 64)
#         self.decoder5 = torch.nn.Sequential(
#             torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#             torch.nn.Conv2d(64, 32, 3, padding=1),
#             torch.nn.Conv2d(32, 64, 3, padding=1),
#         )
#         self.classifier = torch.nn.Conv2d(64, 2, 1)

#     def forward(self, x):
#         # 1st encode stage
#         encoder1 = self.ConvEn1(x)
#         maxpool1 = self.maxpool1(encoder1)
#         # print(maxpool1.shape)

#         # 2nd encode stage
#         encoder2 = self.ConvEn2(maxpool1)
#         maxpool2 = self.maxpool2(encoder2)
#         # print(maxpool2.shape)

#         # 3rd encode stage
#         encoder3 = self.ConvEn3(maxpool2)
#         maxpool3 = self.maxpool3(encoder3)
#         # print(maxpool3.shape)

#         # 4th encode stage
#         encoder4 = self.ConvEn4(maxpool3)
#         maxpool4 = self.maxpool4(encoder4)
#         # print(maxpool4.shape)

#         # 5th encode stage
#         encoder5 = self.ConvEn5(maxpool4)
#         maxpool5 = self.maxpool5(encoder5)
#         # print(maxpool5.shape)


#         # decode
#         decode1 = self.decoder1(maxpool5, maxpool4)
#         # print(decode1.shape)
#         decode2 = self.decoder2(decode1, maxpool3)
#         # print(decode2.shape)
#         decode3 = self.decoder3(decode2, maxpool2)
#         # print(decode3.shape)
#         decode4 = self.decoder4(decode3, maxpool1)
#         # print(decode4.shape)
#         decode5 = self.decoder5(decode4)
#         # print(decode5.shape)
#         result = self.classifier(decode5)
#         # print(result.shape)
#         return result
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True), nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
