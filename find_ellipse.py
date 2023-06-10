import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import glob, os
from natsort import natsorted
import argparse
import torch.nn.functional as F


def find_ellipse(image, output_path):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            ellipses.append(ellipse)
            
    # Determine the dimensions of the original image
    height, width, _ = image.shape

    # Create a blank image of the same dimensions as the original image
    ellipse_mask = np.zeros((height, width), dtype=np.uint8)

    # Draw the filled ellipse on the mask image
    for ellipse in ellipses:
        cv2.ellipse(ellipse_mask, ellipse, color=255, thickness=-1)

    # Save the ellipse mask image
    cv2.imwrite(output_path, ellipse_mask)
    
    return ellipse_mask

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--MaskPath", help="Path to imcomplete output.", type=str, required=True)
    parser.add_argument("--SavePath", help="Path to the saving directory", type=str, default='./solution')
    return parser


parser = get_parser()
args = parser.parse_args()


MaskPath = args.MaskPath
SavePath = args.SavePath
subject = ['S5', 'S6', 'S7', 'S8']


for sub in subject:

  mask_folder_path = natsorted(glob.glob(os.path.join(MaskPath, sub, '*')))
  num_of_folder = len(mask_folder_path)

  for i in range(num_of_folder):

    if not os.path.exists(os.path.join(SavePath, sub, str(i+1).zfill(2))):
        print(f"Creating subfolder {os.path.join(SavePath, sub, str(i+1).zfill(2))} in result directory...")
        os.makedirs(os.path.join(SavePath, sub, str(i+1).zfill(2)))
        
    all_mask_path = natsorted(glob.glob(os.path.join(mask_folder_path[i], '*')))
    num_of_image = len(all_mask_path)
    
    conf_list = open(all_mask_path[-1], "r").readlines()

    
    for j in range(num_of_image-1):
      
        print(all_mask_path[j])
        mask = cv2.imread(all_mask_path[j])
        
        outputPath = os.path.join(SavePath, sub, str(i+1).zfill(2), str(j)+'.png')
        conf_path = os.path.join(SavePath, sub, str(i+1).zfill(2), "conf.txt")
        
        ellipse_mask = find_ellipse(mask, outputPath)
        if np.sum(ellipse_mask) > 0:
            conf = 1
        else:
            conf = 0
        
        with open(conf_path, "a") as file:
            file.write(str(conf) + "\n")
    
