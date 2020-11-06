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

train_ds = images_ds.take(80)
test_ds = images_ds.skip(80)

train_ds = train_ds.batch(5)
test_ds = test_ds.batch(5)

print(train_ds)
print(test_ds)




model = models.Sequential()
#model.add(keras.Input(shape =(800,800,3)))
model.add(convolutional.Convolution2D(16, (3, 3),input_shape=(800, 800,3), activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(32, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(64, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(core.Flatten())
model.add(core.Dense(500, activation='relu'))
model.add(core.Dropout(.5))
model.add(core.Dense(100, activation='relu'))
model.add(core.Dropout(.25))
model.add(core.Dense(20, activation='relu'))
model.add(core.Dense(1))
model.compile(optimizer=optimizers.Adam(lr=1e-04),loss='mean_squared_error', metrics=['mse','mae','mape'])

print(model.summary())



#print(example[1])

#model.fit(train_ds, epochs=3)

history1 = model.fit(train_ds, epochs=3)
print(history1.history)

pyplot.plot(history1.history['mse'])
pyplot.plot(history1.history['mae'])
pyplot.plot(history1.history['mape'])
#pyplot.show()

print("evaluate:")
#result = model.evaluate(test_ds)
#dict(zip(model.metrics_names, result))

history2 = model.evaluate(test_ds)
print(history2)


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

show2(image2)

images_ds2 = list_ds2.map(parse_image2)

#for image, label in images_ds.take(2):
 # show(image, label)

predict_ds = images_ds2.take(1)

predict_ds = predict_ds.batch(1)

print(model.predict(predict_ds))


