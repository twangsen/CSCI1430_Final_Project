import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
import numpy as np
import cv2
from PIL import Image

img_size = 50
train_size = 900
test_size = 300
class_num = 10

#load training images
trainImage = []
for j in range(0, class_num):
    num = str(j)
    for i in range(0, train_size):
        img = Image.open('./data/train/'+  num + "/" + num + '_' + str(i) + '.jpg')
        img = img.resize((img_size, img_size))
        img = np.array(img, dtype=np.float32)
        img /= 255.
        trainImage.append(img)

trainImage = np.array(trainImage)
trainImage = np.expand_dims(trainImage, axis=3)

trainLable = np.zeros((train_size * class_num, class_num))
for j in range(0, class_num):
    trainLable[j*train_size: j*train_size + train_size, j] = 1


# load testing images
testImages = []

for j in range(0, class_num):
    num = str(j)
    for i in range(0, test_size):
        img = Image.open('./data/test/'+  num + "/" + num + '_' + str(i) + '.jpg')
        img = img.resize((img_size, img_size))
        img = np.array(img, dtype=np.float32)
        img /= 255.
        testImages.append(img)

testImages = np.array(testImages)
testImages = np.expand_dims(testImages, axis=3)


testLabels = np.zeros((test_size * class_num, class_num))
for j in range(0, class_num):
    testLabels[j*test_size: j*test_size + test_size, j] = 1

# Define the CNN Model
model = Sequential([
              tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
              tf.keras.layers.MaxPool2D(pool_size=(2,2)),
              tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
              tf.keras.layers.MaxPool2D(pool_size=(2,2)),
              tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
              tf.keras.layers.MaxPool2D(pool_size=(2,2)),
              tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
              #tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
              tf.keras.layers.MaxPool2D(pool_size=(2,2)),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dropout(0.3),
              tf.keras.layers.Dense(units=256, activation='relu'),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(units=64, activation='relu'),
              tf.keras.layers.Dense(units=10, activation='softmax')])

model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])


data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=90,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range = 0.2,
                shear_range = 0.2,
                horizontal_flip=True)
it = data_gen.flow(trainImage, trainLable, batch_size=30, shuffle = True)

model.fit(it, batch_size=30, epochs=30,
           validation_data=(testImages, testLabels))

model.save_weights("my_checkpoint")

