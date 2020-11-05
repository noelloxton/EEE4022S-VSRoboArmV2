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
from keras.callbacks import History
history = History()

np.set_printoptions(precision=4)

path2 = '/home/ghostlini/arm_testing_ws/src/panda_simulation/visual_servo'

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

  image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(image,channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [800, 800])
  return image#, label

def show(image):
  plt.figure()
  plt.imshow(image)
  #plt.title(label.numpy().decode('utf-8'))
  plt.axis('off')
  plt.show()

images_ds2 = list_ds2.map(parse_image2)

for image in images_ds2.take(1):
  show(image)

print(images_ds2)

ds = images_ds2.take(1)
print(ds)



