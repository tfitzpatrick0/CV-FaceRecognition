import cv2, os, sys
import numpy as np
from PIL import Image

def check_path(path):
    dir = os.path.dirname(path)
    if os.path.exists(dir):
        return True
    else:
        return False

def make_path(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_training_params(detected):
    img_paths = [os.path.join(detected, face) for face in os.listdir(detected)]
    faces = []
    ids = []

    for img in img_paths:
        if (os.path.split(img)[-1][0] == '.'):
            continue
        
        grayscale_img = Image.open(img).convert('L')
        np_img = np.array(grayscale_img, 'uint8')

        face = face_detector.detectMultiScale(np_img)
        id = int(os.path.split(img)[-1].split("?")[1])

        for (x, y, w, h) in face:
            faces.append(np_img[y:y+h, x:x+w])
            ids.append(id)

    return faces, ids

def main():

    for arg in sys.argv[1:]:
        detected = ''
        trained = ''

        if (arg == '-t'):
            detected = '1_DETECTED/'
            trained = '1_TRAINED/'
        elif (arg == '-v'):
            detected = '2_DETECTED/'
            trained = '2_TRAINED/'
        elif (arg == '-u'):
            detected = '3_DETECTED/'
            trained = '3_TRAINED/'
        else:
            print('Not a valid argument')
            continue

        if (check_path(detected)):
            make_path(trained)
            faces, ids = get_training_params(detected)
            face_recognizer.train(faces, np.array(ids))

            for model in os.listdir(trained):
                os.remove(os.path.join(trained, model))
            model_path = trained + 'model.yml'
            face_recognizer.save(model_path)
        else:
            print('Assure the following directories are made: ' + detected)

if __name__ == '__main__':
    main()