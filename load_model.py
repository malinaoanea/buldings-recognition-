from keras.models import load_model
import numpy as np
from keras.preprocessing import image
# returns a compiled model
# identical to the previous one
model = load_model('first_try.h5')

img_pred = image.load_img("spital.jpg", target_size=(150, 150))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)

result = model.predict(img_pred)
classes = np.argmax(result, axis=1)

print(result, classes)

if result[0][0] == 0:
    prediction = "kitty kat"
elif result[0][0] == 1:
    prediction = "doggo "
else:
    prediction = "spital"

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

