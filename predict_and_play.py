import numpy as np
import tensorflow as tf
from cv2 import cv2
import time
from pose_estimation_utils import load_movenet_model, movenet_inference_video, init_crop_region, \
    determine_crop_region, feature_engineering
from keys_for_game import keycodes, mouse_codes, wheel_codes, movement_codes,\
    PressKey, ReleaseKey, mouse_click, wheel_movement, move_mouse

IMG_SIZE = (256, 256)
THRESHOLD = 0.9
ENSEMBLE_SIZE = 5
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


def predict_single_pose(camera, model, movenet, input_size, crop_region):
    status, frame = camera.read()
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)
    predict_frame = movenet_inference_video(movenet, image, crop_region, crop_size=[input_size, input_size])
    crop_region = determine_crop_region(predict_frame, IMG_SIZE[0], IMG_SIZE[1])
    predict_frame[0][0][:, :2] *= IMG_SIZE[0]
    predict_frame = feature_engineering(predict_frame)
    predict_frame = np.expand_dims(predict_frame, axis=0)
    model_input = [predict_frame for _ in range(ENSEMBLE_SIZE)]
    prediction = model(model_input, training=False)
    return prediction, crop_region


def predict_poses(class_names, model, movenet, input_size, crop_region, camera, queue_size):
    keys_flag = True
    num_frames = 0
    predictions_lst = [-1] * queue_size

    # Warm-up and predicting first key
    while True:
        first_prediction, crop_region = predict_single_pose(camera, model, movenet, input_size, crop_region)
        predicted_class = np.argmax(first_prediction)
        old_keys = class_names[predicted_class]
        predictions_lst.pop(0)
        predictions_lst.append(predicted_class)
        num_frames += 1
        cv2.waitKey(1)
        if predictions_lst.count(predictions_lst[0]) == len(predictions_lst):
            break
    if old_keys == "Stop":
        keys_flag = False

    while keys_flag:
        prediction, crop_region = predict_single_pose(camera, model, movenet, input_size, crop_region)
        predicted_class = np.argmax(prediction)
        predictions_lst.pop(0)
        predictions_lst.append(predicted_class)
        if np.max(prediction) > THRESHOLD and predictions_lst.count(predictions_lst[0]) == len(predictions_lst):
            old_keys, keys_flag = keystroke(class_names[predicted_class], old_keys, keys_flag)
        num_frames = num_frames + 1
        cv2.waitKey(1)
    return num_frames


def pose_and_play(log_path, model_path, camera_port=0, queue_size=5, movenet_model="thunder"):
    class_names = find_class_names(log_path)
    model = tf.keras.models.load_model(model_path+"/ensemble")
    movenet, input_size = load_movenet_model("movenet_"+movenet_model)
    crop_region = init_crop_region(IMG_SIZE[0], IMG_SIZE[1])
    camera = cv2.VideoCapture(camera_port)
    start = time.time()
    num_frames = predict_poses(class_names, model, movenet, input_size, crop_region, camera, queue_size)
    end = time.time()
    camera.release()
    cv2.destroyAllWindows()
    seconds = end - start
    fps = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))


def find_pose_names(log_path):
    with open(log_path, 'r') as f:
        data = f.readlines()
    class_names = []
    for i in range(1, len(data)):
        class_names.append(' '.join(data[i].split()[1:-1]))
    return class_names


def pose_and_print(log_path, model_path, camera_port=0, queue_size=5, movenet_model="thunder"):
    model = tf.keras.models.load_model(model_path+"/ensemble")
    class_names = find_pose_names(log_path)
    image_height, image_width = IMG_SIZE[0], IMG_SIZE[1]
    movenet, input_size = load_movenet_model("movenet_"+movenet_model)
    crop_region = init_crop_region(image_height, image_width)
    predictions_lst = [-1] * queue_size
    camera = cv2.VideoCapture(camera_port)
    num_frames = 0
    start = time.time()

    while True:
        status, frame = camera.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMG_SIZE)
        model_input = movenet_inference_video(movenet, image, crop_region, crop_size=[input_size, input_size])
        crop_region = determine_crop_region(model_input, image_height, image_width)
        model_input[0][0][:, :2] *= image_height
        model_input = feature_engineering(model_input)
        predict_frame = np.expand_dims(model_input, axis=0)
        predict_frame = [predict_frame for _ in range(5)]
        prediction = model(predict_frame, training=False)
        predicted_class = np.argmax(prediction)
        predictions_lst.pop(0)
        predictions_lst.append(predicted_class)
        if np.max(prediction) > 0.9 and predictions_lst.count(predictions_lst[0]) == len(predictions_lst):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, class_names[predicted_class], (50, 250), font, 3, (255, 0, 0), 4, cv2.LINE_AA)
        num_frames = num_frames + 1
        cv2.imshow("preview", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    end = time.time()
    camera.release()
    cv2.destroyAllWindows()
    seconds = end - start
    fps = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))


if __name__ == '__main__':
    pose_and_print("logs/log2.txt", "saved_models/model2", camera_port=1, queue_size=5, movenet_model="thunder")
    #pose_and_play("logs/log2.txt", "saved_models/model2", camera_port=1, queue_size=5, movenet_model="thunder")
