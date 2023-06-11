from segment_anything import sam_model_registry, SamPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob, os
from natsort import natsorted
import argparse
from PIL import Image


def gamma_correction(img: np.array, gamma=0.5):
    res = np.power(img, gamma)
    return res


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))


def find_ans(image, indices, outputWithCoorPath):
    masks = np.zeros_like(image)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    predictor.set_image(image)

    if indices.shape[0] == 0:
        return masks

    masks, scores, logits = predictor.predict(
        point_coords=np.array([[sum(indices[:, 1]) / indices.shape[0], sum(indices[:, 0]) / indices.shape[0]]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(np.array([[sum(indices[:, 1]) / indices.shape[0], sum(indices[:, 0]) / indices.shape[0]]]), np.array([1]), plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.savefig(outputWithCoorPath)
        plt.close("all")
        break

    return masks[0]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_of_SAM", help="Path to checkpoint of SAM.", type=str, default="sam_vit_h_4b8939.pth")
    parser.add_argument("--DataSetPath", help="Path to dataset.", type=str, required=True)
    parser.add_argument("--MaskPath", help="Path to masks predicted by our model", type=str, required=True)
    parser.add_argument("--SavePath", help="Path to saving directory", type=str, default="./solution/")
    parser.add_argument("--MaskWithCoorPath", help="Path to saving image with coordinate directory", type=str, default="./maskwithcoor/")
    return parser


parser = get_parser()
args = parser.parse_args()

ckpt_of_SAM = args.ckpt_of_SAM
DataSetPath = args.DataSetPath
MaskPath = args.MaskPath
SavePath = args.SavePath
MaskWithCoorPath = args.MaskWithCoorPath

sam = sam_model_registry["vit_h"](checkpoint=ckpt_of_SAM)
sam.to(device="cuda")
predictor = SamPredictor(sam)

subject = ["S5", "S6", "S7", "S8"]

for sub in subject:
    img_folder_path = natsorted(glob.glob(os.path.join(DataSetPath, sub, "*")))  # /home/yuchien/Ganzin-Pupil-Tracking/dataset/S5/01~..
    mask_folder_path = natsorted(glob.glob(os.path.join(MaskPath, sub, "*")))
    num_of_folder = len(img_folder_path)

    for i in range(num_of_folder):
        if not os.path.exists(os.path.join(SavePath, sub, str(i + 1).zfill(2))):
            print(f"Creating subfolder {os.path.join(SavePath, sub, str(i+1).zfill(2))} in result directory...")
            os.makedirs(os.path.join(SavePath, sub, str(i + 1).zfill(2)))
            os.makedirs(os.path.join(MaskWithCoorPath, sub, str(i + 1).zfill(2)))

        all_image_path = natsorted(glob.glob(os.path.join(img_folder_path[i], "*")))  # /home/yuchien/Ganzin-Pupil-Tracking/dataset/S5/01~../1.jpg~..
        all_mask_path = natsorted(glob.glob(os.path.join(mask_folder_path[i], "*")))
        num_of_image = len(all_image_path)

        for j in range(num_of_image):
            mask = cv2.imread(all_mask_path[j], cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(all_image_path[j], cv2.IMREAD_GRAYSCALE)

            index_of_pupil = mask == 255
            indices = np.argwhere(index_of_pupil)

            outputWithCoorPath = os.path.join(MaskWithCoorPath, sub, str(i + 1).zfill(2), str(j) + ".png")
            output = find_ans(image, indices, outputWithCoorPath)

            if np.sum(output) > 0:
                conf = 1
            else:
                conf = 0

            outputPath = os.path.join(SavePath, sub, str(i + 1).zfill(2), str(j) + ".png")
            conf_path = os.path.join(SavePath, sub, str(i + 1).zfill(2), "conf.txt")

            fig = Image.fromarray(output)
            fig.save(outputPath)

            with open(conf_path, "a") as file:
                file.write(str(conf) + "\n")
