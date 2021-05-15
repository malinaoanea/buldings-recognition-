from keras.preprocessing.image import ImageDataGenerator 
# ImageDataGenerator printeaza ceva de genul: "Found 71 images belonging to 5 classes.", adica impartirea in foldere
import cv2
import numpy as np
import pandas as pd
from keras import backend as K
import matplotlib.pyplot as pb
import os
import random
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image
from tensorflow.keras import regularizers
import tensorflow as tf
import gc # garbage colector
from tensorflow.keras import utils as np_utils


img_witdh, img_height = 150, 150

train_data_dir = "data/train"
validation_data_sir = "data/validation"
nb_train_samples = 10
nb_validation_samples = 10
epochs = 64
batch_size = 5 #cate poze vrem sa proceseze inainte de a face un update la loss function


# aici face toate pozele din fisiere de acelasi tip
if K.image_data_format() == 'chanel_first':
    input_shape = (3, img_witdh, img_height)
else:
    input_shape = (img_witdh, img_height, 3) # 3 arata ca este color imaginea

# aici face augumentarea pentru a mari numarul de poze de trainuire si validare
# https://www.analyticsvidhya.com/blog/2020/08/image-augmentation-on-the-fly-using-keras-imagedatagenerator/
train_data_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. /255)

train_generator = train_data_gen.flow_from_directory(
    train_data_dir, 
    target_size=(img_witdh, img_height),
    batch_size=batch_size,
    class_mode="categorical"
)

print(train_generator.classes)

validation_generator = test_datagen.flow_from_directory(
    validation_data_sir,
    target_size=(img_witdh, img_height),
    batch_size=batch_size,
    class_mode="categorical"
)


model = Sequential() # name of the neuronal network
model.add(Conv2D(54, (3, 3), activation="relu", input_shape=input_shape))
model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3),  activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())


# model.add(Conv2D(64, (3, 3), activation="relu", input_shape=input_shape))


# model.add(Conv2D(128, (3, 3), activation="relu", input_shape=input_shape))
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(128, (3, 3), activation="relu", input_shape=input_shape))
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dropout(0.5))

# model.add(Dense(512, activation="relu"))
# model.add(Dense(1, activation='softmax'))

# model.summary()

# model.add(MaxPool2D((2, 2)))
# model.add(Dropout(0.5))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPool2D((2, 2)))
# model.add(Dropout(0.5))
# model.add(Conv2D(64, (3, 3), activation='relu'))


# model.summary()

# model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=["categorical_accuracy", 'accuracy'])

model.summary()
# lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3) #monitors the validation loss for signs of a plateau and then alter the learning rate by the specified factor if a plateau is detected

# early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=6, mode='auto')  #This will monitor and stop the model training if it is not further converging

# checkpointer = tf.keras.callbacks.ModelCheckpoint('weights.hd5', monitor='val_loss', verbose=1, save_best_only=True) #This allows checkpoints to be saved each epoch just in case the model stops training

epochs = 100
batch_size = 64
learning_rate = 0.001

print(train_generator.classes, validation_generator.classes)

train_generator = np_utils.to_categorical(train_generator, 3)
validation_generator= np_utils.to_categorical(validation_generator, 3)


model.fit(
        train_generator,

        epochs = 50,
        steps_per_epoch=60,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size
        )

# model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# model.fit(
#     train_generator, 
#     steps_per_epoch = nb_train_samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size)


# model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# model.save_weights("first_try.h5")
# model.save("fisrt_try_model.h5")

# model.save_weights("first_try.h5")

img_pred = image.load_img("cat.12473.jpg", target_size=(150, 150))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)

result = model.predict(img_pred)
classes = np.argmax(result, axis=1)

print(result, classes)

if result[0][0] == 0:
    prediction = "pisicuta"
elif result[0][0] == 1:
    prediction = "doggo"
elif result[0][0] == 2:
    prediction = "spitalul coltea"
else:
    prediction = "cladire 5 to go"

print(prediction)

