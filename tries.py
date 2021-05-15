#ASTA MERGE PE DOUA CLASE
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
import gc # garbage colector

img_witdh, img_height = 150, 150

train_data_dir = "data/train"
validation_data_sir = "data/validation"
nb_train_samples = 1000
nb_validation_samples = 100
epochs = 64
batch_size = 20 #cate poze vrem sa proceseze inainte de a face un update la loss function


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
    class_mode="binary"
)

print(train_generator.classes)

validation_generator = test_datagen.flow_from_directory(
    validation_data_sir,
    target_size=(img_witdh, img_height),
    batch_size=batch_size,
    class_mode="binary"
)

print(train_generator.class_indices)
print(validation_generator.class_indices)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPool2D((2, 2)),
    
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# model.add(Conv2D(128, (3, 3), activation="relu", input_shape=input_shape))
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dropout(0.5))

# model.add(Dense(512, activation="relu"))
# model.add(Dense(1, activation='softmax'))

# model.summary()

# model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# model.fit(
#     train_generator, 
#     steps_per_epoch = nb_train_samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size)

model.fit(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


# model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# model.save_weights("first_try.h5")
# model.save("fisrt_try_model.h5")

model.save_weights("first_try.h5")

img_pred = image.load_img("spital.jpg", target_size=(150, 150))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)

result = model.predict(img_pred)
classes = np.argmax(result, axis=1)

print(result, classes)

if result[0][0] == 0:
    prediction = "doggo"
elif result[0][0] == 1:
    prediction = "spital "

print(prediction)

img_pred = image.load_img("dog.4.jpg", target_size=(150, 150))
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

img_pred = image.load_img("cat.4.jpg", target_size=(150, 150))
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

print(prediction)

