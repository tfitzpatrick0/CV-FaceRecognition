import cv2, os, sys
import numpy as np

def check_path(path):
    dir = os.path.dirname(path)
    if os.path.exists(dir):
        return True
    else:
        return False

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def build_id_map(detected):
    img_paths = [os.path.join(detected, face) for face in os.listdir(detected)]
    id_map = {}

    for img in img_paths:
        if (os.path.split(img)[-1][0] == '.'):
            continue

        id = int(os.path.split(img)[-1].split("?")[1])
        name = os.path.split(img)[-1].split("?")[0]

        if id not in id_map.keys():
            id_map[id] = name

    return id_map

def recognize_face(img_path, id_map):
    img = cv2.imread(img_path)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_detector.detectMultiScale(grayscale, 1.3, 5)
    result = ''

    for (x, y, w, h) in face:
        cv2.rectangle(img, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 4)
        id, confidence = face_recognizer.predict(grayscale[y:y+h, x:x+w])

        result = id_map[id] + ", {0:.2f}%".format(round(100 - confidence, 2))

    return result

def accuracy_test(dir, detected, trained):
    id_map = build_id_map(detected)
    face_recognizer.read(trained)

    total = 0
    correct = 0

    for name in os.listdir(dir):
        name_path = os.path.join(dir, name)
        if (os.path.split(name_path)[-1][0] == '.'):
            continue

        for face in os.listdir(name_path):
            img_path = os.path.join(name_path, face)

            result = recognize_face(img_path, id_map)
            print("Name: " + name + " --> Result: " + result)
            recognized_name = result.split(',')[0]

            if (recognized_name == name):
                correct += 1
            total += 1

    accuracy = correct / total
    print("Accuracy: " + str(accuracy))

def main():

    for arg in sys.argv[1:]:
        dir = ''
        detected = ''
        trained = ''

        if (arg == '-t'):
            dir = '1_training/'
            detected = '1_detected/'
            trained = '1_trained/model.yml'
        elif (arg == '-v'):
            dir = '2_validation/'
            detected = '2_detected/'
            trained = '2_trained/model.yml'
        elif (arg == '-u'):
            dir = '3_unknown/'
            detected = '3_detected/'
            trained = '3_trained/model.yml'
        else:
            print('Not a valid argument')
            continue

        if (check_path(dir) and check_path(detected) and check_path(trained)):
            accuracy_test(dir, detected, trained)
            #result = recognize_face('2_validation/Kurt_Warner/Kurt_Warner_0001.jpg', id_map)
            #result = recognize_face('2_validation/LeBron_James/LeBron_James_0001.jpg', id_map)
            #print(result)
        else:
            print('Assure the following directories are made: ' + detected + ' ' + trained)

if __name__ == '__main__':
    main()