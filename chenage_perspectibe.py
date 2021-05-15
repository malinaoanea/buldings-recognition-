import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("15.jpg")

cv2.circle(img, (215, 35), 5, (0, 0, 255), -1) # top left
cv2.circle(img, (550, 25), 5, (0, 0, 255), -1) # top right
cv2.circle(img, (130, 370), 5, (0, 0, 255), -1) # bottom left
cv2.circle(img, (630, 370), 5, (0, 0, 255), -1) # bottom right

cv2.imshow("Image", img)
cv2.waitKey(0)