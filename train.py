import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import yaml
from model.unet import Unet
from model.deeplab.deeplab_v3 import DeeplabV3Plus
# from custom_net import CustomSegmentationNet
from utils import create_dir, load_dataset, get_colormap
from img_proc import tf_dataset
import argparse

class MyTrainer():
    def __init__(self,config):
      
      # Access the hyperparameters
      self.model = config['MODEL']

      self.random_seed = config['parameters']['RANDOM_SEED']
      self.num_classes = config['parameters']['NUM_CLASSES']
      self.input_shape = config['parameters']['INPUT_SHAPE']
      self.batch_size = config['parameters']['BATCH_SIZE']
      self.learning_rate = config['parameters']['LEARNING_RATE']
      self.num_epochs = config['parameters']['NUM_EPOCHS']
      self.backbone = config['parameters']['DL_BACKBONE']

      self.dataset_path = config['paths']['dataset_path']
      self.classes, self.colormap = get_colormap(self.dataset_path)
      self.output_path = config['paths']['output']
      

      self.checkpoint_vb = config['checkpoint']['verbose']
      self.checkpoint_save_best_only = config['checkpoint']['save_best_only']

      self.LROn_monitor = config['LROn']['monitor']
      self.LROn_factor = config['LROn']['factor']
      self.LROn_patience = config['LROn']['patience']
      self.LROn_min_lr = config['LROn']['min_lr']
      self.LROn_vb = config['LROn']['verbose']

      self.ES_monitor = config['EarlyStopping']['monitor']
      self.ES_patience = config['EarlyStopping']['patience']
      self.ES_rbw = config['EarlyStopping']['restore_best_weights']


    def build_loader(self):
      np.random.seed(self.random_seed)
      tf.random.set_seed(self.random_seed)
      (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(self.dataset_path)

      print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
      
      train_dataset = tf_dataset(train_x, train_y, batch=self.batch_size)
      valid_dataset = tf_dataset(valid_x, valid_y, batch=self.batch_size)
      return train_dataset, valid_dataset



    def build_model(self, model):
      if model == "unet":
        model = Unet(self.input_shape, self.num_classes)
        # model.load_weights(model_path)
        model.compile(loss="categorical_crossentropy",optimizer=tf.keras.optimizers.Adam(self.learning_rate))
        # model.summary()
        callbacks = [
          ModelCheckpoint(self.output_path, verbose=self.checkpoint_vb, save_best_only=self.checkpoint_save_best_only),
          ReduceLROnPlateau(monitor=self.LROn_monitor, factor=self.LROn_factor, patience=self.LROn_patience, min_lr=self.LROn_min_lr, verbose=self.LROn_vb),
          EarlyStopping(monitor=self.ES_monitor, patience=self.ES_patience, restore_best_weights=self.ES_rbw)
          ]
      elif model == "deeplab":
        model = DeeplabV3Plus(
                num_classes=self.num_classes,
                backbone=self.backbone
                )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy']
                )

        callbacks = [ModelCheckpoint(filepath=self.output_path,monitor='val_loss',save_best_only=True,mode='min',save_weights_only=True),
                     EarlyStopping(monitor=self.ES_monitor, patience=self.ES_patience, restore_best_weights=self.ES_rbw)
                     ]
      
      return model, callbacks


    def train(self):
      train_dataset, valid_dataset = self.build_loader()
      model, callbacks = self.build_model(self.model)
      model.fit(train_dataset,validation_data=valid_dataset,epochs=self.num_epochs,callbacks=callbacks)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default="", metavar="FILE", help="path to config file")
  parser.add_argument("--output", default="", metavar="FILE", help="path to config file")
  args = parser.parse_args()
  
  with open(args.config, 'r') as file:
    config = yaml.safe_load(file)
  Mytrainer = MyTrainer(config)
  Mytrainer.train()
