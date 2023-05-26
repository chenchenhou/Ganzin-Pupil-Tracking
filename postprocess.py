import numpy as np
import torch
import cv2


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


def is_ellipse(contour, threshold=0.55):
    # Fit an ellipse to the contour
    if contour.shape[0] < 5:
        return False
    ellipse = cv2.fitEllipse(contour)
    _, (axes), _ = ellipse

    minor_axis, major_axis = min(axes), max(axes)
    # Define threshold for eccentricity
    eccentricity_threshold = threshold

    # Calculate eccentricity
    eccentricity = minor_axis / major_axis
    print(eccentricity)
    # Check if eccentricity suggests an ellipse
    if eccentricity > eccentricity_threshold:
        return True
    else:
        return False


def find_pupil(mask):
    edges = cv2.Canny(mask, 30, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    not_pupil_masks = []
    for m in contours:
        candidate = np.squeeze(m)
        result = is_ellipse(candidate)
        if not result:
            not_pupil_masks.append(candidate)

    for c in not_pupil_masks:
        if len(c.shape) < 2:
            c = np.expand_dims(c, axis=0)
        c = np.transpose(c, (1, 0))
        xx, yy = np.meshgrid(c[0], c[1])
        mask[yy, xx] = 0

    return mask
