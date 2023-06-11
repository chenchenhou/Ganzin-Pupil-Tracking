import cv2
import numpy as np
import os
import glob
from PIL import Image
from natsort import natsorted
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution_dir", help="Path to solution directory.", type=str, default="./solution/")
    return parser


parser = get_parser()
args = parser.parse_args()
solution_dir = args.solution_dir
masks = glob.glob(os.path.join(solution_dir, "**/*.txt"), recursive=True)
masks = natsorted(masks)
# print(masks)


for m in masks:
    file = open(m, "r")
    conf = file.read()
    conf = conf.split("\n")
    l = len(conf)
    # print(conf)
    file.close()
    os.remove(m)

    # modify conf by sequence
    zeros_count = 0
    fisrt_zero_idx = []
    last_zero_idx = []
    for j in range(l - 1):
        if conf[j] == str(0) and j == 0:
            fisrt_zero_idx.append(j)
        elif conf[j] == str(0) and conf[j - 1] == str(1):
            fisrt_zero_idx.append(j)
        if conf[j] == 0 and j == l - 2:
            last_zero_idx.append(j)
        elif conf[j] == str(0) and conf[j + 1] == str(1):
            last_zero_idx.append(j)

    for idx in fisrt_zero_idx:
        if idx != 0:
            conf[idx - 1] = str(0)
            conf[idx - 2] = str(0)
    for idx in last_zero_idx:
        if idx != (l - 2):
            conf[idx + 1] = str(0)
            conf[idx + 2] = str(0)
    # print(conf)
    for i in range(l):
        with open(m, "a") as file:
            file.write(conf[i] + "\n")
