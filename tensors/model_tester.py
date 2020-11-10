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

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from matplotlib import pyplot

np.set_printoptions(precision=4)
i=1
path = '/home/ghostlini/panda_arm_sim/src/panda_simulation/data_for_nn'

target_root = pathlib.Path(path)
print(target_root)
for item in target_root.glob("*"):
  print(item.name)

list_ds = tf.data.Dataset.list_files(str(target_root/'*/*'), shuffle=True)

for f in list_ds.take(5):
  print(f.numpy())
"""
list_ds = list_ds.shuffle(1000, seed=10,reshuffle_each_iteration=False)
for f in list_ds.take(5):
  print(f.numpy())
"""
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
"""
def show(image, label):
  plt.figure()
  plt.imshow(image)
  plt.title(label.numpy().decode('utf-8'))
  plt.axis('off')
  plt.show()

#show(image, label)
"""
images_ds = list_ds.map(parse_image)

#train_validate_ds = images_ds.take(900)
#test_ds = images_ds.skip(900)
train_ds = images_ds.take(800)
validate_ds = images_ds.skip(800)
#train_ds = images_ds.take(80)
#validate_ds = images_ds.skip(80)

#train_validate_ds = train_validate_ds.shuffle(900,reshuffle_each_iteration=False)

#train_ds = train_validate_ds.take(720)
#validate_ds = train_validate_ds.skip(720)

print(train_ds)
print(validate_ds)
#print(test_ds)

model = models.Sequential()
model.add(keras.Input(shape =(800,800,3)))
model.add(convolutional.Convolution2D(16, (3, 3), activation='relu'))
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
model.compile(optimizer=optimizers.Adam(),loss='mean_squared_error', metrics=['mae','mape'])

print(model.summary())

filename = "best_model_y{}.hdf5".format(i)
checkpoint = ModelCheckpoint(filename, monitor='loss', verbose=1, save_best_only=True, mode='auto')
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

train_ds = train_ds.batch(10)
validate_ds = validate_ds.batch(10)
#train_ds = train_ds.batch(2)
#validate_ds = validate_ds.batch(2)

history1 = model.fit(train_ds, epochs=50,validation_data=validate_ds)#, callbacks=[checkpoint])

#loss
pyplot.plot(history1.history['loss'],label='train_loss')
pyplot.plot(history1.history['val_loss'],label='val_loss')
pyplot.xlabel('Number of Epochs')
pyplot.ylabel('Error')
pyplot.legend()
pyplot.show()
pyplot.clf()

"""
#mae
pyplot.plot(history1.history['mae'],label='train_MAE')
pyplot.plot(history1.history['val_mae'],label='val_MAE')
pyplot.xlabel('Number of Epochs')
pyplot.ylabel('Error')
pyplot.legend()
pyplot.show()
pyplot.clf()
#mape
pyplot.plot(history1.history['mape'],label='train_MAPE')
pyplot.plot(history1.history['val_mape'],label='val_MAPE')
pyplot.xlabel('Number of  Epochs')
pyplot.ylabel('Error')
pyplot.legend()
pyplot.show()
pyplot.clf()
"""
"""
test_ds = test_ds.batch(10)

model = load_model(filename)
history2 = model.evaluate(test_ds)
f = open("losses.txt", "a+")
f.write(str(i)+" - loss:" + str(history2[0])+"\n")
f.close()

 
#print(history1.history)

#loss
pyplot.plot(history1.history['loss'],label='train_loss')
pyplot.plot(history1.history['val_loss'],label='val_loss')
pyplot.xlabel('Number of Epochs')
pyplot.ylabel('Loss')
pyplot.legend()
pyplot.show()
pyplot.clf()
#mse and mae
#pyplot.plot(history1.history['mse'],label='train_MSE')
pyplot.plot(history1.history['mae'],label='train_MAE')
#pyplot.plot(history1.history['val_mse'],label='val_MSE')
pyplot.plot(history1.history['val_mae'],label='val_MAE')
pyplot.xlabel('Number of Epochs')
pyplot.ylabel('Error')
pyplot.legend()
pyplot.show()
pyplot.clf()
#mape
pyplot.plot(history1.history['mape'],label='train_MAPE')
pyplot.plot(history1.history['val_mape'],label='val_MAPE')
pyplot.xlabel('Number of  Epochs')
pyplot.ylabel('Error')
pyplot.legend()
pyplot.show()
pyplot.clf()

print("evaluate:")
#result = model.evaluate(test_ds)
#dict(zip(model.metrics_names, result))

#history2 = model.evaluate(test_ds)
#pyplot.plot(history2.history['mse'])
#pyplot.show()
#print(history2)
"""


