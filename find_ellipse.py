import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import glob, os
from natsort import natsorted
import argparse
import torch.nn.functional as F
import math

def check_for_nan(element):
    for item in element:
        if isinstance(item, tuple):
            if (np.isnan(item[0]) or np.isnan(item[1])):
                #print(item[0], item[1])
                return True
        elif np.isnan(item):
            return True
    return False

def ellipse_complete(image, output_path=None, main=None):
    if main == None:
        gray = (image*255).astype(np.uint8)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    ellipses = []
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            ellipses.append(ellipse)

    # Create a blank image of the same dimensions as the original image
    ellipse_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Draw the filled ellipse on the mask image
    num_of_pts = {}
    
    for i, ellipse in enumerate(ellipses):
        temp_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        if check_for_nan(ellipse):
            continue
        else:
            cv2.ellipse(temp_mask, ellipse, 255, -1)
            num_of_pts[i] = np.sum( temp_mask == 255)
            
    
    if len(num_of_pts) != 0:
        max_value = max(num_of_pts.values())
        max_key = max(num_of_pts, key=num_of_pts.get)
        cv2.ellipse(ellipse_mask, ellipses[max_key], 255, -1)

            
    # Save the ellipse mask image
    if output_path is not None:
        cv2.imwrite(output_path, ellipse_mask)

    return ellipse_mask


if __name__ == '__main__':

    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--MaskPath", help="Path to imcomplete output.", type=str, default="./solution_merge")
        parser.add_argument("--SavePath", help="Path to the saving directory", type=str, default="./solution/")
        return parser


    parser = get_parser()
    args = parser.parse_args()


    MaskPath = args.MaskPath
    SavePath = args.SavePath
    subject = ["S5", "S6", "S7", "S8"]

    main = 1

    for sub in subject:
        mask_folder_path = natsorted(glob.glob(os.path.join(MaskPath, sub, "*")))
        num_of_folder = len(mask_folder_path)

        for i in range(num_of_folder):
            if not os.path.exists(os.path.join(SavePath, sub, str(i + 1).zfill(2))):
                print(f"Creating subfolder {os.path.join(SavePath, sub, str(i+1).zfill(2))} in result directory...")
                os.makedirs(os.path.join(SavePath, sub, str(i + 1).zfill(2)))

            all_mask_path = natsorted(glob.glob(os.path.join(mask_folder_path[i], "*")))
            num_of_image = len(all_mask_path)

            conf_list = open(all_mask_path[-1], "r").readlines()

            for j in range(num_of_image - 1):
                print(all_mask_path[j])
                mask = cv2.imread(all_mask_path[j])

                outputPath = os.path.join(SavePath, sub, str(i + 1).zfill(2), str(j) + ".png")
                conf_path = os.path.join(SavePath, sub, str(i + 1).zfill(2), "conf.txt")

                ellipse_mask = ellipse_complete(mask, outputPath, main)
                
                if np.sum(ellipse_mask) > 0:
                    conf = 1
                else:
                    conf = 0

                with open(conf_path, "a") as file:
                    file.write(str(conf) + "\n")
