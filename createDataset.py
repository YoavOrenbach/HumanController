from PIL import Image
from cv2 import cv2
import os
import time

#useful link:
# https://www.analyticsvidhya.com/blog/2021/05/create-your-own-image-dataset-using-opencv-in-machine-learning/
IMG_SIZE = (256, 256)


def create_data_folder(data):
    if not os.path.exists(data):
        os.mkdir(data)


def timer(camera):
    TIMER = int(5)
    prev = time.time()
    while TIMER >= 0:
        ret, img = camera.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(TIMER),
                    (200, 250), font,
                    7, (0, 255, 255),
                    4, cv2.LINE_AA)
        cv2.imshow("preview", img)
        cv2.waitKey(1)
        cur = time.time()
        if cur-prev >= 1:
            prev = cur
            TIMER = TIMER-1


def save_pose_data(camera, key):
    count = 0
    while count < 100:
        status, frame = camera.read()
        cv2.imshow("preview", frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        frame = cv2.resize(frame, IMG_SIZE)  # resize to model if needed
        #cv2.imwrite('dataset/'+key+'/img'+str(count)+'.png', frame)
        Image.fromarray(frame).save('dataset/'+key+'/img'+str(count)+'.png')
        cv2.waitKey(1)
        count = count + 1


def collectKeyData(key, camera_port):
    create_data_folder("dataset")
    cv2.namedWindow("preview")
    camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    create_data_folder("dataset/" + key)
    timer(camera)
    save_pose_data(camera, key)
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    key_input = input("please enter the key you want to train or type exit if you are finished: ")
    while key_input != "exit":
        collectKeyData(key_input, 1)
        key_input = input("please enter the key you want to train or type exit if you are finished: ")

