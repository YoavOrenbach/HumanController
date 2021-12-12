import numpy as np
import tensorflow as tf
from cv2 import cv2
import time
import os
from movenet_utils import load_movenet_model, movenet_inference
from mediapipe.python.solutions import pose as mp_pose
from blazepose_utils import FullBodyPoseEmbedder
from keys_for_game import PressKey, ReleaseKey


def predict_poses(camera, model, movenet, input_size):
    class_names = os.listdir("dataset")
    num_frames = 0
    while True:
        status, frame = camera.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predict_frame = movenet_inference(image, movenet, input_size)
        predict_frame = np.expand_dims(predict_frame, axis=0)
        prediction = model.predict(predict_frame)
        predicted_class = np.argmax(prediction)
        if np.max(prediction) > 0.8:
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
    #model = tf.keras.models.load_model("movenet_saved_model")
    model_name = "movenet_thunder"
    movenet, input_size = load_movenet_model(model_name)
    cv2.namedWindow("preview")
    camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    start = time.time()
    num_frames = predict_poses(camera, model, movenet, input_size)
    end = time.time()
    camera.release()
    cv2.destroyAllWindows()
    seconds = end - start
    fps = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))


