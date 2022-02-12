import numpy as np
import tensorflow as tf
from cv2 import cv2
import time
import os
from mediapipe.python.solutions import pose as mp_pose
from blazepose_utils import FullBodyPoseEmbedder
from movenet_utils import load_movenet_model, movenet_inference_video, init_crop_region, determine_crop_region
from keys_for_game import keycodes, mouse_codes, wheel_codes, movement_codes,\
    PressKey, ReleaseKey, mouse_click, wheel_movement, move_mouse

IMG_SIZE = (256, 256)
SPECIAL_KEYS = ['Home', 'Up', 'PageUp', 'Left', 'Right', 'End', 'Down', 'PageDown', 'Insert', 'Delete']


def find_class_names(log_path):
    with open(log_path, 'r') as f:
        data = f.readlines()
    class_names = []
    for i in range(1, len(data)):
        class_names.append(data[i].split()[-1])
    return class_names


def simulate_press(key):
    if "mouse" in key:
        if "click" in key:
            mouse_click(mouse_codes[key][0])
        elif "wheel" in key:
            wheel_movement(wheel_codes[key])
        else:
            move_mouse(*movement_codes[key])
    else:
        if key in SPECIAL_KEYS:
            PressKey(0x2A)
        PressKey(keycodes[key])


def simulate_release(key):
    if "mouse" in key:
        if "click" in key:
            mouse_click(mouse_codes[key][1])
    else:
        ReleaseKey(keycodes[key])
        if key in SPECIAL_KEYS:
            ReleaseKey(0x2A)


def keystroke_press(keys):
    if '_' not in keys:
        simulate_press(keys)
    else:
        key1, key2 = keys.split('_')
        simulate_press(key1)
        simulate_press(key2)


def keystroke_release(keys):
    if '_' not in keys:
        simulate_release(keys)
    else:
        key1, key2 = keys.split('_')
        simulate_release(key1)
        simulate_release(key2)


def keystroke(keys, old_keys, keys_flag):
    if keys == old_keys:
        return keys, keys_flag

    if old_keys != "Normal" and old_keys != "Stop":
        keystroke_release(old_keys)

    if keys == "Stop":
        keys_flag = False
    elif keys != "Normal":
        keystroke_press(keys)
    return keys, keys_flag


def predict_single_pose(camera, model, movenet, input_size, crop_region, image_height, image_width):
    status, frame = camera.read()
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)
    predict_frame = movenet_inference_video(movenet, image, crop_region, crop_size=[input_size, input_size])
    crop_region = determine_crop_region(predict_frame, image_height, image_width)
    predict_frame[0][0][:, :2] *= image_height
    predict_frame = np.expand_dims(predict_frame, axis=0)
    prediction = model.predict(predict_frame)
    return prediction, crop_region


def predict_poses(class_names, model, movenet, input_size, image_height, image_width, crop_region, camera):
    keys_flag = True
    first_prediction, crop_region = predict_single_pose(camera, model, movenet, input_size, crop_region, image_height, image_width)
    predicted_class = np.argmax(first_prediction)
    old_keys = class_names[predicted_class]
    num_frames = 1
    while True:
        prediction, crop_region = predict_single_pose(camera, model, movenet, input_size, crop_region, image_height, image_width)
        predicted_class = np.argmax(prediction)
        if np.max(prediction) > 0.99:
            old_keys, keys_flag = keystroke(class_names[predicted_class], old_keys, keys_flag)
        num_frames = num_frames + 1
        cv2.waitKey(1)
        if keys_flag == False:
            break
    return num_frames


def pose_and_play(log_path, model_path, camera_port=0):
    class_names = find_class_names(log_path)
    model = tf.keras.models.load_model(model_path)
    movenet, input_size = load_movenet_model("movenet_thunder")
    image_height, image_width = IMG_SIZE[0], IMG_SIZE[1]
    crop_region = init_crop_region(image_height, image_width)
    camera = cv2.VideoCapture(camera_port)
    start = time.time()
    num_frames = predict_poses(class_names, model, movenet, input_size, image_height, image_width, crop_region, camera)
    end = time.time()
    camera.release()
    cv2.destroyAllWindows()
    seconds = end - start
    fps = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))


def print_poses(model_name, camera_port=0):
    model = tf.keras.models.load_model(model_name + "_saved_model")
    class_names = os.listdir("dataset")
    image_height, image_width = IMG_SIZE[0], IMG_SIZE[1]
    if model_name == "blazepose":
        pose_embedder = FullBodyPoseEmbedder()
        pose_tracker = mp_pose.Pose()
    else:
        movenet, input_size = load_movenet_model("movenet_thunder")
        crop_region = init_crop_region(image_height, image_width)
    camera = cv2.VideoCapture(camera_port)
    num_frames = 0
    start = time.time()
    while True:
        status, frame = camera.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMG_SIZE)
        if model_name == "blazepose":
            result = pose_tracker.process(image=image)
            pose_landmarks = result.pose_landmarks
            if pose_landmarks is None:
                continue
            pose_landmarks = np.array([[lmk.x * image_width, lmk.y * image_height, lmk.z * image_width]
                                       for lmk in pose_landmarks.landmark], dtype=np.float32)
            model_input = pose_embedder(pose_landmarks)
        else:
            model_input = movenet_inference_video(movenet, image, crop_region, crop_size=[input_size, input_size])
            crop_region = determine_crop_region(model_input, image_height, image_width)
            model_input[0][0][:, :2] *= image_height
        predict_frame = np.expand_dims(model_input, axis=0)
        prediction = model.predict(predict_frame)
        predicted_class = np.argmax(prediction)
        if np.max(prediction) > 0.99:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, class_names[predicted_class], (50, 250), font, 3, (255, 0, 0), 4, cv2.LINE_AA)
            """
            prediction_array = np.array2string(prediction, formatter={'float_kind':lambda x: "%.2f" % x})
            cv2.putText(frame, prediction_array[:27], (0, 150), font, 1, (255, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, prediction_array[27:52], (0, 190), font, 1, (255, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, prediction_array[52:77], (0, 230), font, 1, (255, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, prediction_array[77:102], (0, 270), font, 1, (255, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, prediction_array[102:127], (0, 310), font, 1, (255, 0, 0), 4, cv2.LINE_AA)
            """
        num_frames = num_frames + 1
        cv2.imshow("preview", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    if model_name == "blazepose":
        pose_tracker.close()
    end = time.time()
    camera.release()
    cv2.destroyAllWindows()
    seconds = end - start
    fps = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))


if __name__ == '__main__':
    #pose_and_play("logs/log1.txt", "saved_models/model1", 1)
    print_poses("movenet", 1)
