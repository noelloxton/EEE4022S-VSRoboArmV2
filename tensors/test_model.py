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

from keras.callbacks import ModelCheckpoint

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

train_ds = images_ds.take(800)
test_ds = images_ds.skip(800)

train_ds = train_ds.batch(10)
test_ds = test_ds.batch(10)

print(train_ds)
print(test_ds)



#,input_shape=(800, 800,3)
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
model.compile(optimizer=optimizers.Adam(),loss='mean_squared_error', metrics=['accuracy','mae','mape'])

print(model.summary())

#checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)

#print(example[1])
"""
model.fit(train_ds, epochs=5)
scores = model.evaluate(test_ds)
model_json = model.to_json()
with open("model.json","w") as json_file:
  json_file.write(model_json)
model.save_weights("model.h5")
print("saved model to disk!")

"""
history1 = model.fit(train_ds, epochs=5,validation_data=test_ds)#, callbacks=[checkpoint])

print(history1.history)

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



