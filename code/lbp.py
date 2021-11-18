# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2020
# ________________________________________________________________
# Adam Czajka, September 2020
#
# Simple example illustrating how to calculate Local Binary Pattern 
# representation of an image, and then how to build a histogram of the 
# resulting numbers to serve as image features. Note: in your applications, 
# you may want to split your image into N cells, calculate LPBs for each cell, 
# and concatenate resulting histograms into a single feature vector 
# for the entire image. This can be accomplished by using feature.multiblock_lbp(...).

import cv2
from skimage import feature
import numpy as np
from matplotlib import pyplot as plt

# 8 neighbors, 3x3 neighborhood (radius = 1) and 59 bins -- as discussed in class
numPoints = 8
radius = 1
bins = 59
eps = np.finfo(np.float32).eps

# Open the webcam
cam = cv2.VideoCapture(0)

while (True):

    # Get and show webcam frame
    _, img = cam.read()
    res_scale = .5             # rescale the input image if it's too large
    img = cv2.resize(img, (0,0), fx=res_scale, fy=res_scale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Webcam grayscale image", gray)

    # Calculate and show the Local Binary Pattern representation
    lbp = feature.local_binary_pattern(gray, numPoints, radius, method="nri_uniform")
    cv2.imshow("LBP representation",cv2.normalize(lbp, lbp, 0, 1, cv2.NORM_MINMAX))

    # Calculate and normalize the histogram
    (hist, _) = np.histogram(lbp.ravel(), bins = bins)
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    action = cv2.waitKey(1)
    if action & 0xFF == 27:  # escape
        break
    elif action == ord('h'): # show our image features: histogram of the LBP representation
        plt.bar(range(bins), hist, width = 0.9)
        plt.title('Histogram for the LBP representation')
        plt.show()

# Close the webcam and all windows
cam.release()
cv2.destroyAllWindows()

