import numpy as np
import torch


def draw_segmentation_map(outputs, label_map):
    labels = torch.argmax(outputs.squeeze().cpu(), dim=0).numpy()
    # Create 3 Numpy arrays containing zeros.
    # Later each pixel will be filled with respective red, green, and blue pixels
    # depending on the predicted class.

    R_map = np.zeros_like(labels).astype(np.uint8)
    G_map = np.zeros_like(labels).astype(np.uint8)
    B_map = np.zeros_like(labels).astype(np.uint8)
    for label_num in range(0, len(label_map.keys())):
        index = labels == label_num
        R_map[index] = label_map[label_num][0]
        G_map[index] = label_map[label_num][1]
        B_map[index] = label_map[label_num][2]

    segmentation_map = np.stack([R_map, G_map, B_map], axis=2)
    return segmentation_map
