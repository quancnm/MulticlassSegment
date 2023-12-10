import os
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import argparse
from utils import save_results, predict_and_save,  calculate_metrics, load_dataset
from img_proc import preprocess_image , preprocess_mask
import yaml
class MyPredictor():
    def __init__(self,config):
      
      # Access the hyperparameters
      self.random_seed = config['parameters']['RANDOM_SEED']
      self.num_classes = config['parameters']['NUM_CLASSES']
      self.input_shape = config['parameters']['INPUT_SHAPE']
      self.batch_size = config['parameters']['BATCH_SIZE']
      self.learning_rate = config['parameters']['LEARNING_RATE']
      self.num_epochs = config['parameters']['NUM_EPOCHS']

      self.dataset_path = config['paths']['dataset_path']
      # self.classes, self.colormap = get_colormap(self.dataset_path)
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
      return test_x, test_y



    def load_model(self):
      model = tf.keras.models.load_model(self.output_path)
      return model
    def predict(self):
      SCORE = []
      test_x, test_y = self.build_loader()
      model = self.load_model()
      for x, y in tqdm(zip(test_x, test_y), total=len(test_x), desc="Processing images"):
          name = os.path.splitext(os.path.basename(x))[0]

          image, image_x = preprocess_image(x, image_shape = [self.input_shape[1], self.input_shape[0]])
          onehot_mask, mask_x = preprocess_mask(y, image_shape = [self.input_shape[1], self.input_shape[0]])

          pred = predict_and_save(model, image, mask_x, name, self.dataset_path)

          f1_value, jac_value = calculate_metrics(onehot_mask, pred, self.num_classes)
          SCORE.append([f1_value, jac_value])

      score = np.mean(np.array(SCORE), axis=0)
      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--output", default="", metavar="FILE", help="path to config file")
    args = parser.parse_args()
  
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    Mypredictor = MyPredictor(config)
    Mypredictor.predict()


