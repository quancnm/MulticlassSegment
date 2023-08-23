
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import yaml
from unet import Unet
from custom_net import CustomSegmentationNet
from utils import create_dir, load_dataset, get_colormap
from img_proc import tf_dataset



if __name__ == "__main__":
    
    np.random.seed(42)
    tf.random.set_seed(42)

    dataset_path = "Multiclass-Segmentation/data"
    model_path = "Multiclass-Segmentation/output"
    create_dir("Multiclass-Segmentation/output")

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
    
    #Loading the dataset
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    #Create Dataset
    train_dataset = tf_dataset(train_x, train_y, batch=BATCH_SIZE)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=BATCH_SIZE)

    #Model
    model = Unet(INPUT_SHAPE, NUM_CLASSES)
    # model.load_weights(model_path)
    model.compile(loss="categorical_crossentropy",optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
    # model.summary()

    
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(train_dataset,validation_data=valid_dataset,epochs=NUM_EPOCHS,callbacks=callbacks)
