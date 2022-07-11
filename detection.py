import cv2, os, sys

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

def detect_face(dir, detected):
    for f in os.listdir(detected):
        os.remove(os.path.join(detected, f))
    id = 0

    for name in os.listdir(dir):
        name_path = os.path.join(dir, name)
        if (os.path.split(name_path)[-1][0] == '.'):
            continue
        count = 0     
        id += 1

        for face in os.listdir(name_path):
            img_path = os.path.join(name_path, face)
            if (os.path.split(img_path)[-1][0] == '.'):
                continue

            img = cv2.imread(img_path)
            grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = face_detector.detectMultiScale(grayscale, 1.3, 5)

            for (x, y, w, h) in face:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                count += 1

                #print(detected + person + "_" + str(count) + ".jpg")
                cv2.imwrite(detected + name + "?" + str(id) + "?" + str(count) + ".jpg", grayscale[y:y+h,x:x+w])

def main():

    for arg in sys.argv[1:]:
        dir = ''
        detected = ''

        if (arg == '-t'):
            dir = '1_training/'
            detected = '1_DETECTED/'
        elif (arg == '-v'):
            dir = '2_validation/'
            detected = '2_DETECTED/'
        elif (arg == '-u'):
            dir = '3_unknown/'
            detected = '3_DETECTED/'
        else:
            print('Not a valid argument')
            continue

        if (check_path(dir)):
            make_path(detected)
            detect_face(dir, detected)
        else:
            print('Assure the following directories are made: ' + dir)

if __name__ == '__main__':
    main()
