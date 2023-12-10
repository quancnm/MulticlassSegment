import numpy as np
import cv2
from utils import get_colormap
import tensorflow as tf

dataset_path = "/content/drive/MyDrive/MulticlassSegment-main/data"
CLASSES, COLORMAP = get_colormap(dataset_path)

def read_image_mask(x, y, IMG_SHAPE =[320, 416] ):
  x = cv2.imread(x, cv2.IMREAD_COLOR)
  y = cv2.imread(y, cv2.IMREAD_COLOR)
  assert x.shape == y.shape

  x = cv2.resize(x, (IMG_SHAPE[1], IMG_SHAPE[0]))
  y = cv2.resize(y, (IMG_SHAPE[1], IMG_SHAPE[0]))

  x = x / 255.0
  x = x.astype(np.float32)

  output = []
  for color in COLORMAP:
      cmap = np.all(np.equal(y, color), axis=-1)
      output.append(cmap)
  output = np.stack(output, axis=-1)
  output = output.astype(np.uint8)

  return x, output

def preprocess(x, y):
  def f(x, y):
      x = x.decode()
      y = y.decode()
      image, mask = read_image_mask(x, y)
      return image, mask

  image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.uint8])
  image.set_shape([320, 416, 3])
  mask.set_shape([320, 416, 20])
  
  return image, mask

def tf_dataset(x, y, batch):
  
  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  dataset = dataset.shuffle(buffer_size=5000)
  dataset = dataset.map(preprocess)
  dataset = dataset.batch(batch)
  dataset = dataset.prefetch(2)
  return dataset



def preprocess_image(image_path, image_shape ):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (image_shape[0], image_shape[1]))
    image_normalized = image / 255.0
    return np.expand_dims(image_normalized, axis=0), image

def preprocess_mask(mask_path, image_shape):
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    mask = cv2.resize(mask, (image_shape[0], image_shape[1]))
    onehot_mask = np.stack([np.all(np.equal(mask, color), axis=-1) for color in COLORMAP], axis=-1)
    onehot_mask = np.argmax(onehot_mask, axis=-1).astype(np.int32)
    return onehot_mask, mask
