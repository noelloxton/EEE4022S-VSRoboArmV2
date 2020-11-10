import tensorflow as tf
import keras
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skimage.transform as sktransform

from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling
from keras.models import load_model

from matplotlib import pyplot

np.set_printoptions(precision=4)

path = '/home/ghostlini/panda_arm_sim/src/panda_simulation/data_for_nn'

target_root = pathlib.Path(path)
print(target_root)
for item in target_root.glob("*"):
  print(item.name)

list_ds = tf.data.Dataset.list_files(str(target_root/'*/*'), shuffle=True)

for f in list_ds.take(5):
  print(f.numpy())

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def parse_image(filename):
  parts = tf.strings.split(filename, os.sep)
  label = parts[-2]
  label = float(label)

  image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(image,channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [800, 800])
  return image, label

file_path = next(iter(list_ds))
image, label = parse_image(file_path)

def show(image, label):
  plt.figure()
  plt.imshow(image)
  plt.title(label.numpy().decode('utf-8'))
  plt.axis('off')
  plt.show()

#show(image, label)

images_ds = list_ds.map(parse_image)

#for image, label in images_ds.take(2):
 # show(image, label)

train_ds = images_ds.take(900)
test_ds = images_ds.skip(900)

train_ds = train_ds.batch(10)
test_ds = test_ds.batch(10)

print(train_ds)
print(test_ds)

model = load_model('best_model.hdf5')
print(model.summary())

"""
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(optimizer=optimizers.Adam(lr=1e-04),loss='mean_squared_error', metrics=['mse','mae','mape'])

score = model.evaluate(test_ds)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
"""

path2 = '/home/ghostlini/panda_arm_sim/src/panda_simulation/visual_servo'

target_root2 = pathlib.Path(path2)
print(target_root2)
for item in target_root2.glob("*"):
  print(item.name)

list_ds2 = tf.data.Dataset.list_files(str(target_root2/'*/*'), shuffle=False)

for f in list_ds2.take(1):
  print(f.numpy())

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def parse_image2(filename):
  #parts = tf.strings.split(filename, os.sep)
  #label = parts[-2]
  #label = float(label)

  image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(image,channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [800, 800])
  return image#, label

file_path2 = next(iter(list_ds2))
image2 = parse_image2(file_path2)

def show2(image):
  plt.figure()
  plt.imshow(image)
  #plt.title(label.numpy().decode('utf-8'))
  plt.axis('off')
  plt.show()

#show2(image2)

images_ds2 = list_ds2.map(parse_image2)

#for image, label in images_ds.take(2):
 # show(image, label)

predict_ds = images_ds2.take(1)

predict_ds = predict_ds.batch(1)

print(model.predict(predict_ds))

