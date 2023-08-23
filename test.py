import os
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from train import load_dataset, create_dir, get_colormap
from utils import save_results, predict_and_save,  calculate_metrics
from img_proc import preprocess_image , preprocess_mask
import yaml


if __name__ == "__main__":

    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("results")
    dataset_path = "Multiclass-Segmentation/data"
    model_path = "Multiclass-Segmentation/output"

    with open('Multiclass-Segmentation/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Access the hyperparameters
    IMG_HEIGHT = config['parameters']['IMG_HEIGHT']
    IMG_WIDTH = config['parameters']['IMG_WIDTH']
    NUM_CLASSES = config['parameters']['NUM_CLASSES']
    INPUT_SHAPE = config['parameters']['INPUT_SHAPE']
    BATCH_SIZE = config['parameters']['BATCH_SIZE']
    LEARNING_RATE = config['parameters']['LEARNING_RATE']
    NUM_EPOCHS = config['parameters']['NUM_EPOCHS']
    CLASSES, COLORMAP = get_colormap(dataset_path)

    model = tf.keras.models.load_model(model_path)
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
  

    SCORE = []

    for x, y in tqdm(zip(test_x, test_y), total=len(test_x), desc="Processing images"):
        name = os.path.splitext(os.path.basename(x))[0]

        image, image_x = preprocess_image(x, image_shape = [IMG_WIDTH, IMG_HEIGHT])
        onehot_mask, mask_x = preprocess_mask(y, image_shape = [IMG_WIDTH, IMG_HEIGHT])

        pred = predict_and_save(model, image, mask_x, name)

        f1_value, jac_value = calculate_metrics(onehot_mask, pred, NUM_CLASSES)
        SCORE.append([f1_value, jac_value])

    score = np.mean(np.array(SCORE), axis=0)
    
