import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import glob, os
from natsort import natsorted
from PIL import Image
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_1", help="Path to first directory.", type=str, required=True)
    parser.add_argument("--candidate_2", help="Path to second directory.", type=str, required=True)
    parser.add_argument("--SavePath", help="Path to the saving directory", type=str, default="./solution_merge/")
    return parser


parser = get_parser()
args = parser.parse_args()


candidate_1 = args.candidate_1
candidate_2 = args.candidate_2
SavePath = args.SavePath
subject = ["S5", "S6", "S7", "S8"]


for sub in subject:
    SAM_folder_path = natsorted(glob.glob(os.path.join(candidate_1, sub, "*")))  # /home/yuchien/Ganzin-Pupil-Tracking/dataset/S5/01~..
    deep_folder_path = natsorted(glob.glob(os.path.join(candidate_2, sub, "*")))
    num_of_folder = len(SAM_folder_path)

    for i in range(num_of_folder):
        if not os.path.exists(os.path.join(SavePath, sub, str(i + 1).zfill(2))):
            print(f"Creating subfolder {os.path.join(SavePath, sub, str(i+1).zfill(2))} in result directory...")
            os.makedirs(os.path.join(SavePath, sub, str(i + 1).zfill(2)))

        SAM_all_image_path = natsorted(glob.glob(os.path.join(SAM_folder_path[i], "*")))  # /home/yuchien/Ganzin-Pupil-Tracking/dataset/S5/01~../1.jpg~..
        deep_all_mask_path = natsorted(glob.glob(os.path.join(deep_folder_path[i], "*")))
        num_of_image = len(SAM_all_image_path)
        # print(SAM_all_image_path)

        conf_of_sam = open(SAM_all_image_path[-1], "r").readlines()
        conf_of_deep = open(deep_all_mask_path[-1], "r").readlines()

        # print(conf_of_sam)

        for j in range(num_of_image - 1):
            outputPath = os.path.join(SavePath, sub, str(i + 1).zfill(2), str(j) + ".png")
            print(outputPath)

            sam = cv2.imread(SAM_all_image_path[j], cv2.IMREAD_GRAYSCALE)
            deep = cv2.imread(deep_all_mask_path[j], cv2.IMREAD_GRAYSCALE)

            num_of_pupil_sam = np.sum(sam)
            num_of_pupil_deep = np.sum(deep)

            if num_of_pupil_sam >= num_of_pupil_deep:
                # print("num_of_pupil_sam:", num_of_pupil_sam)
                conf = conf_of_sam[j]
                fig = Image.fromarray(sam)
                fig.save(outputPath)
            else:
                fig = Image.fromarray(deep)
                fig.save(outputPath)
                conf = conf_of_deep[j]

            conf_path = os.path.join(SavePath, sub, str(i + 1).zfill(2), "conf.txt")

            with open(conf_path, "a") as file:
                file.write(str(conf))
