import os
from glob import glob
from sklearn.model_selection import train_test_split
import scipy.io 
import numpy as np
import cv2
from img_proc import grayscale_to_rgb
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_dataset(path, split=0.2):
    train_x = sorted(glob(os.path.join(path, "Training", "Images", "*")))[:100]
    train_y = sorted(glob(os.path.join(path, "Training", "Categories", "*")))[:100]

    split_size = int(split * len(train_x))

    train_x, valid_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def get_colormap(path):
    mat_path = os.path.join(path, "human_colormap.mat")
    colormap = scipy.io.loadmat(mat_path)["colormap"]
    colormap = colormap * 256
    colormap = colormap.astype(np.uint8)
    colormap = [[c[2], c[1], c[0]] for c in colormap]
    classes = ["Background","Hat","Hair","Glove","Sunglasses",
                "UpperClothes","Dress","Coat","Socks","Pants",
                "Torso-skin","Scarf","Skirt","Face","Left-arm",
                "Right-arm","Left-leg","Right-leg","Left-shoe",
                "Right-shoe"]

    return classes, colormap

def save_results(image, mask, pred, save_image_path, dataset_path):
    CLASSES, COLORMAP = get_colormap(dataset_path)
    h, w, _ = image.shape
    line = np.ones((h, 10, 3)) * 255

    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, CLASSES, COLORMAP)

    cat_images = np.concatenate([image, line, mask, line, pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)


def predict_and_save(model, image, mask, name):
    pred = model.predict(image, verbose=0)[0]
    pred = np.argmax(pred, axis=-1).astype(np.float32)

    save_image_path = os.path.join("results", f"{name}.png")
    save_results(image, mask, pred, save_image_path)

    return pred

def calculate_metrics(onehot_mask, pred, num_classes):
    labels = [i for i in range(num_classes)]
    f1_value = f1_score(onehot_mask.flatten(), pred.flatten(), labels=labels, average=None, zero_division=0)
    jac_value = jaccard_score(onehot_mask.flatten(), pred.flatten(), labels=labels, average=None, zero_division=0)
    return f1_value, jac_value