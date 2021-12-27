import numpy as np
import tensorflow as tf
from cv2 import cv2
import time
import os
from movenet_utils import load_movenet_model, movenet_inference_video, init_crop_region, determine_crop_region
from keys_for_game import PressKey, ReleaseKey, left_click_press, left_click_release, right_click_press, right_click_release

IMG_SIZE = (256, 256)
keycode_dictionary = {
    'Esc': 0x01,
    '1': 0x02,
    'W': 0x11,
    'A': 0x1E,
    'S': 0x1F,
    'D': 0x20,
    'B': 0x30,
    'C': 0x2E,
    'E': 0x12,
    'F': 0x21,
    'Q': 0x10,
    'Space': 0x39
}

def key_by_pose(prediction, prediction_flag):
    if prediction == "bow":
        prediction_flag = False
    elif prediction == "left_click":
        left_click_press()
        time.sleep(0.2)
        left_click_release()
    elif prediction == "right_click":
        right_click_press()
        time.sleep(0.2)
        right_click_release()
    elif prediction != "normal":
        PressKey(keycode_dictionary[prediction])
        time.sleep(0.2)
        ReleaseKey(keycode_dictionary[prediction])
    return prediction_flag


def predict_poses(camera, model, movenet, input_size):
    class_names = os.listdir("dataset")
    image_height, image_width = IMG_SIZE[0], IMG_SIZE[1]
    crop_region = init_crop_region(image_height, image_width)
    num_frames = 0
    prediction_flag = True
    while True:
        status, frame = camera.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMG_SIZE)
        predict_frame = movenet_inference_video(movenet, image, crop_region, crop_size=[input_size, input_size])
        crop_region = determine_crop_region(predict_frame, image_height, image_width)
        predict_frame[0][0][:, :2] *= image_height
        predict_frame = np.expand_dims(predict_frame, axis=0)
        prediction = model.predict(predict_frame)
        predicted_class = np.argmax(prediction)
        if np.max(prediction) > 0.99:
            prediction_flag = key_by_pose(class_names[predicted_class], prediction_flag)
        num_frames = num_frames + 1
        key = cv2.waitKey(1)
        if prediction_flag == False:
            break
    return num_frames


def print_poses(camera, model, movenet, input_size):
    class_names = os.listdir("dataset")
    image_height, image_width = IMG_SIZE[0], IMG_SIZE[1]
    crop_region = init_crop_region(image_height, image_width)
    num_frames = 0
    while True:
        status, frame = camera.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMG_SIZE)
        predict_frame = movenet_inference_video(movenet, image, crop_region, crop_size=[input_size, input_size])
        crop_region = determine_crop_region(predict_frame, image_height, image_width)
        predict_frame[0][0][:, :2] *= image_height
        predict_frame = np.expand_dims(predict_frame, axis=0)
        prediction = model.predict(predict_frame)
        predicted_class = np.argmax(prediction)
        if np.max(prediction) > 0.99:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, class_names[predicted_class], (50, 250), font, 3, (255, 0, 0), 4, cv2.LINE_AA)
        num_frames = num_frames + 1
        cv2.imshow("preview", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    return num_frames


def pose_and_play(camera_port=0):
    model = tf.keras.models.load_model("saved_model")
    model_name = "movenet_thunder"
    movenet, input_size = load_movenet_model(model_name)
    #cv2.namedWindow("preview")
    camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    start = time.time()
    num_frames = predict_poses(camera, model, movenet, input_size)
    end = time.time()
    camera.release()
    #cv2.destroyAllWindows()
    seconds = end - start
    fps = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))


if __name__ == '__main__':
    pose_and_play()
