from PIL import Image
from cv2 import cv2
import os
import time

IMG_SIZE = (256, 256)
DATA_SIZE = 300


def create_data_folder(data):
    """Creates a data folder."""
    if not os.path.exists(data):
        os.mkdir(data)


def pose_timer(camera, timer=5):
    """
    Initiates a timer before capturing the pose frames.
    :param camera: the video capture device.
    :param timer: the time to wait.
    """
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
    """
    Captures and saves the images of the poses taken.
    :param camera: the video capture device.
    :param key: the key/pose that is being saved.
    :param data_folder: the directory for saving the images.
    :param count: an integer counter indicating from where to start saving the images.
    :return:
    """
    for index in range(DATA_SIZE):
        status, frame = camera.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow("preview", frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        frame = cv2.resize(frame, IMG_SIZE)  # resize to model if needed
        # cv2.imwrite('dataset/'+key+'/img'+str(count)+'.png', frame)
        Image.fromarray(frame).save(data_folder+'/'+key+'/img'+str(index+count)+'.png')
        cv2.waitKey(1)


def collect_key_data(key, camera_port=0, data_folder="dataset", timer=5, count=0):
    """
    This function collects pose key data.
    :param key: the key/pose to save.
    :param camera_port: the port of the camera to capture the images.
    :param data_folder: the directory for saving the images.
    :param timer: the amount of time to wait before capturing each frame.
    :param count: an integer counter indicating from where to start saving the images.
    """
    #create_data_folder(data_folder)
    cv2.namedWindow("preview")
    camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    create_data_folder(data_folder + "/" + key)
    pose_timer(camera, timer)
    save_pose_data(camera, key, data_folder, count)
    camera.release()
    cv2.destroyAllWindows()


def choose_keys(data_folder="dataset"):
    """
    This function collects data for given input keys.
    :param data_folder: the directory for saving the images.
    """
    key_input = input("please enter the key you want to train or type exit if you are finished: ")
    while key_input != "exit":
        collect_key_data(key_input, data_folder=data_folder)
        key_input = input("please enter the key you want to train or type exit if you are finished: ")


def test_data(test_data_folder="test_dataset"):
    """
    This function collects different test data for given input tests.
    :param test_data_folder: the directory for saving the images.
    """
    create_data_folder(test_data_folder)
    test_input = input("please enter the test folder or type exit if you are finished: ")
    while test_input != "exit":
        choose_keys(test_data_folder + "/" + test_input)
        test_input = input("please enter the test folder or type exit if you are finished: ")
