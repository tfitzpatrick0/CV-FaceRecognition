import cv2, os
import numpy as np

import detection
import trainer
import recognition

# Take images with webcam and add to unknown dataset
# Train unknown dataset
# Recognize faces in real time using trained model

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def populate_imgs(name, img_path):
    cam = cv2.VideoCapture(0)
    count = 0

    while(True):
        _, img = cam.read()
        cv2.imshow('image', img)
        count += 1
        
        cv2.imwrite(img_path + "/" + name + "_" + str(count) + ".jpg", img)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif count > 100:
            break

    cam.release()
    cv2.destroyAllWindows()

def realtime_recognize_face(model_path, id_map):
    face_recognizer.read(model_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cam = cv2.VideoCapture(0)

    while True:
        ret, img = cam.read()
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_detector.detectMultiScale(grayscale, 1.3, 5)

        for (x, y, w, h) in face:
            cv2.rectangle(img, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 4)
            id, confidence = face_recognizer.predict(grayscale[y:y+h, x:x+w])

            result = id_map[id] + ", {0:.2f}%".format(round(100 - confidence, 2))

            cv2.rectangle(img, (x-22, y-90), (x+w+22, y-22), (0, 255, 0), -1)
            cv2.putText(img, str(result), (x, y-40), font, 1, (255, 255, 255), 3)

        cv2.imshow('image', img)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def main():

    dir = '3_unknown/'
    detected = '3_DETECTED/'
    trained = '3_TRAINED/'

    detection.make_path(detected)
    detection.make_path(trained)

    name = input("Enter your name (ex: Tim_Fitzpatrick): ")

    if (detection.check_path(dir)):
        new_img_path = os.path.join(dir, name)
        detection.make_path(new_img_path)

        for f in os.listdir(new_img_path):
            os.remove(os.path.join(new_img_path, f))
        populate_imgs(name, new_img_path)

        print("Detecting faces...")
        detection.detect_face(dir, detected)
        
        print("Training model...")
        faces, ids = trainer.get_training_params(detected)
        face_recognizer.train(faces, np.array(ids))

        for model in os.listdir(trained):
            os.remove(os.path.join(trained, model))
        model_path = trained + 'model.yml'
        face_recognizer.save(model_path)

        print("Recognizing face...")
        id_map = recognition.build_id_map(detected)
        realtime_recognize_face(model_path, id_map)

if __name__ == '__main__':
    main()