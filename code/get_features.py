# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2019


import cv2
import numpy as np
from skimage import feature
from matplotlib import pyplot as plt

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
#cam = cv2.VideoCapture(0)

img = cv2.imread('1_training/Aaron_Peirsol/Aaron_Peirsol_0001.jpg')

while (True):
    #retval, img = cam.read()
    res_scale = .5             # rescale the input image if it's too large
    #img = cv2.resize(img, (0,0), fx=res_scale, fy=res_scale)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (16, 16),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        pass
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face detector", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cam.release()
cv2.destroyAllWindows()
pad=20
img_cropped= img[y-pad+1:y+h+pad, x-pad+1:x+w+pad]
im_reshape = cv2.resize(img_cropped , (img.shape[0], img.shape[1]), interpolation=cv2.INTER_LINEAR)
norm_img = np.zeros((img.shape[0], img.shape[1]))
norm_img = cv2.normalize(im_reshape, norm_img, 0, 255, cv2.NORM_MINMAX)
gray = cv2.cvtColor(norm_img,cv2.COLOR_BGR2GRAY)
#cv2.imwrite('preprocess/output.png', gray)
cv2.imshow("Face", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
# crop and normalize 128x128
####LBP
# 8 neighbors, 3x3 neighborhood (radius = 1) and 59 bins -- as discussed in class
numPoints = 8
radius = 1
bins = 59
eps = np.finfo(np.float32).eps

# Open the webcam

while (True):

    # Get and show webcam frame
    #_, img = cam.read()
    res_scale = .5             # rescale the input image if it's too large
    #img = cv2.resize(img, (0,0), fx=res_scale, fy=res_scale)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        plt.savefig('features/Aaron_1_features.png')
        plt.show()
        

# Close the webcam and all windows
#cam.release()
cv2.destroyAllWindows()

