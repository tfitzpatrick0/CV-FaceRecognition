# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2019
#hello from the other side
# ________________________________________________________________
# Adam Czajka, September 2017

import cv2
import numpy as np

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
cam = cv2.VideoCapture(0)

while (True):
    retval, img = cam.read()
    res_scale = .5             # rescale the input image if it's too large
    img = cv2.resize(img, (0,0), fx=res_scale, fy=res_scale)

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
        #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face detector", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
pad=20
img_cropped= img[y-pad+1:y+h+pad, x-pad+1:x+w+pad]
im_reshape = cv2.resize(img_cropped , (img.shape[0], img.shape[1]), interpolation=cv2.INTER_LINEAR)
norm_img = np.zeros((img.shape[0], img.shape[1]))
norm_img = cv2.normalize(im_reshape, norm_img, 0, 255, cv2.NORM_MINMAX)
gray = cv2.cvtColor(norm_img,cv2.COLOR_BGR2GRAY)
cv2.imwrite('preprocess/output.png', gray)
cv2.imshow("Face", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
# crop and normalize 128x128