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


def pose_timer(camera, timer=5):
    prev = time.time()
    while timer >= 0:
        ret, img = camera.read()
        img = cv2.flip(img, 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(timer),
                    (200, 250), font,
                    7, (0, 255, 255),
                    4, cv2.LINE_AA)
        cv2.imshow("preview", img)
        cv2.waitKey(1)
        cur = time.time()
        if cur-prev >= 1:
            prev = cur
            timer = timer-1


def save_pose_data(camera, key, data_folder="dataset", count=0):
    for index in range(300):
        status, frame = camera.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow("preview", frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        frame = cv2.resize(frame, IMG_SIZE)  # resize to model if needed
        # cv2.imwrite('dataset/'+key+'/img'+str(count)+'.png', frame)
        Image.fromarray(frame).save(data_folder+'/'+key+'/img'+str(index+count)+'.png')
        cv2.waitKey(1)


def collect_key_data(key, camera_port=0, data_folder="dataset", timer=5, count=0):
    create_data_folder(data_folder)
    cv2.namedWindow("preview")
    camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    create_data_folder(data_folder + "/" + key)
    pose_timer(camera, timer)
    save_pose_data(camera, key, data_folder, count)
    camera.release()
    cv2.destroyAllWindows()


def choose_keys(data_folder="dataset"):
    key_input = input("please enter the key you want to train or type exit if you are finished: ")
    while key_input != "exit":
        collect_key_data(key_input, data_folder=data_folder)
        key_input = input("please enter the key you want to train or type exit if you are finished: ")


def test_data():
    create_data_folder("testDataset")
    test_input = input("please enter the test folder or type exit if you are finished: ")
    while test_input != "exit":
        choose_keys("testDataset" + "/" + test_input)
        test_input = input("please enter the test folder or type exit if you are finished: ")


if __name__ == '__main__':
    test_data()
