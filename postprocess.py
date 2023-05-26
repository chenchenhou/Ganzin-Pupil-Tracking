import cv2
import torch
import numpy as np


def draw_segmentation_map(pred: np.ndarray, label_map: dict) -> np.ndarray:
    # labels = torch.argmax(outputs.squeeze().cpu(), dim=0).numpy()
    # Create 3 Numpy arrays containing zeros.
    # Later each pixel will be filled with respective red, green, and blue pixels
    # depending on the predicted class.

    R_map = np.zeros_like(pred).astype(np.uint8)
    G_map = np.zeros_like(pred).astype(np.uint8)
    B_map = np.zeros_like(pred).astype(np.uint8)
    for label_num in range(0, len(label_map.keys())):
        index = pred == label_num
        R_map[index] = label_map[label_num][0]
        G_map[index] = label_map[label_num][1]
        B_map[index] = label_map[label_num][2]

    segmentation_map = np.stack([R_map, G_map, B_map], axis=2)
    return segmentation_map


def connected_components(pred: np.ndarray, threshold: int) -> np.ndarray:
    pred = pred.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred, connectivity=8, ltype=cv2.CV_32S)

    max_area, max_label = 0, -1
    background_color = 0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if (area > max_area) and (area > threshold):
            max_area = area
            max_label = label

    labels = labels.astype(np.uint8)
    labels[labels != max_label] = background_color
    cleaned_pred = labels

    return cleaned_pred
