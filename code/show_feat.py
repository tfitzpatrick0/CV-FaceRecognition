# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2019


import cv2
import numpy as np
from skimage import feature
from matplotlib import pyplot as plt

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

while (True):
    x=cv2.imread('features/Aaron_1_features.png')
    cv2.imshow("SAVED", x)
    action = cv2.waitKey(1)
    if action & 0xFF == 27:  # escap
        break
cv2.destroyAllWindows()

