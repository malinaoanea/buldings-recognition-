from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image

img_witdh, img_height = 150, 150

train_data_dir = "data/train"
validation_data_sir = "data/validation"
nb_train_samples = 10
nb_validation_samples = 10
epochs = 10
batch_size = 5

# Image data is represented in a three-dimensional array where the first channel represents the color channels

# aici face toate pozele din fisiere de acelasi tip
if K.image_data_format() == 'chanel_first':
    input_shape = (3, img_witdh, img_height)
else:
    input_shape = (img_witdh, img_height, 3) # 3 arata ca este color imaginea

# fata augemantation
# casa poporului - train - 20

train_data_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. /255)

# se iau automat pozele din directory
train_generator = train_data_gen.flow_from_directory(
    train_data_dir, 
    target_size=(img_witdh, img_height),
    batch_size=batch_size,
    class_mode="binary"
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_sir,
    target_size=(img_witdh, img_height),
    batch_size=batch_size,
    class_mode="binary"
)


# put data in the neuronal network
# 
model = Sequential() # name of the neuronal network
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.summary()

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.summary()

model.compile(loss="binary_crossentropy", 
optimizer = "rmsprop",
metrics=["accuracy"])

# augmentation used for trainign configuration

model.fit_generator(
    train_generator, 
    steps_per_epoch = nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights("first_try.h5")

img_pred = image.load_img("cami.jpg", target_size=(150, 150))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)

result = model.predict(img_pred)
print(result)

if result[0][0] == 1:
    prediction = "cerc_militar"
else:
    prediction = "casa_poporului"

print(prediction)