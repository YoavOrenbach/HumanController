import numpy as np
from cv2 import cv2
import time
from keys_for_game import keycodes, mouse_codes, wheel_codes, movement_codes, \
    PressKey, ReleaseKey, mouse_click, wheel_movement, move_mouse, get_numlock_state
from pose_estimation_models.pose_estimation_logic import PoseEstimation
from feature_engineering.feature_engineering_logic import FeatureEngineering
from classifiers.classifier_logic import Classifier

IMG_SIZE = (256, 256)
THRESHOLD = 0.9
SPECIAL_KEYS = ['Home', 'Up', 'PageUp', 'Left', 'Right', 'End', 'Down', 'PageDown', 'Insert', 'Delete']


def find_class_names(log_path):
    """
    This function finds the class names (the keys) in the given log path.
    :param log_path: the log path containing triplets of [pose_id, pose_name, keys]
    :return: the class names (keys).
    """
    with open(log_path, 'r') as f:
        data = f.readlines()
    class_names = []
    for i in range(1, len(data)):
        class_names.append(data[i].split()[-1])
    return class_names


def simulate_press(key):
    """
    This function simulates a key press according to the given key.
    :param key: a string representing keys. It it starts with mouse then it is a mouse event,
    otherwise it is a keyboard event.
    """
    if "mouse" in key:
        if "click" in key:
            mouse_click(mouse_codes[key][0])
        elif "wheel" in key:
            wheel_movement(wheel_codes[key])
        else:
            move_mouse(*movement_codes[key])
    else:
        #if key in SPECIAL_KEYS:
        #    PressKey(0x2A)
        PressKey(keycodes[key])


def simulate_release(key):
    """
    This function simulates a key press according to the given key.
    :param key: a string representing a key. It it starts with mouse then it is a mouse event,
    otherwise it is a keyboard event.
    """
    if "mouse" in key:
        if "click" in key:
            mouse_click(mouse_codes[key][1])
    else:
        ReleaseKey(keycodes[key])
        #if key in SPECIAL_KEYS:
        #    ReleaseKey(0x2A)


def keystroke(keys, old_keys, keys_flag):
    """
    This function creates keystrokes according to the old keys that are being pressed and the
     current keys that should be pressed.
    :param keys: the current keys to simulate a press for.
    :param old_keys: the previous keys that should be released.
    :param keys_flag: a flag for stopping execution if the pose is a Stop pose.
    :return: the new keys pressed and the keys_flag.
    """
    if keys == old_keys:
        return keys, keys_flag

    if old_keys != "Normal" and old_keys != "Stop":
        for key in old_keys.split('_'):
            simulate_release(key)

    if keys == "Stop":
        keys_flag = False
    elif keys != "Normal":
        for key in keys.split('_'):
            simulate_press(key)
    return keys, keys_flag


def predict_single_pose(camera, pose_estimation_model: PoseEstimation,
                        feature_engineering: FeatureEngineering, classifier: Classifier):
    """
    This function predicts a pose from a single frame.
    :param camera: The video capture device to read the frames.
    :param pose_estimation_model: the pose estimation model to extract keypoints.
    :param feature_engineering: the feature engineering methods used to process keypoints.
    :param classifier: The classifier that classifies the processed input.
    :return: The predicted class by the classifier.
    """
    status, frame = camera.read()
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)
    landmarks = pose_estimation_model.process_frame(image)
    model_input = feature_engineering(landmarks)
    prediction = classifier.predict(model_input)
    return prediction


def predict_poses(class_names, pose_estimation_model: PoseEstimation,
                  feature_engineering: FeatureEngineering, classifier: Classifier, camera, queue_size):
    """
    This function is the main driver of the simulation - it predicts the poses and simulates the corresponding
    key presses.
    :param class_names: the keys mapped to every pose.
    :param pose_estimation_model: the pose estimation model to extract keypoints.
    :param feature_engineering: the feature engineering methods used to process keypoints.
    :param classifier: The classifier that classifies the processed input.
    :param camera: The video capture device to read the frames.
    :param queue_size: the queue size for simulating a keypress only when the queue is full.
    :return: the number of frames that were processed.
    """
    keys_flag = True
    num_frames = 0
    predictions_lst = [-1] * queue_size

    # Warm-up and predicting first key
    while True:
        first_prediction = predict_single_pose(camera, pose_estimation_model, feature_engineering, classifier)
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
        prediction = predict_single_pose(camera, pose_estimation_model, feature_engineering, classifier)
        predicted_class = np.argmax(prediction)
        predictions_lst.pop(0)
        predictions_lst.append(predicted_class)
        if np.max(prediction) > THRESHOLD and predictions_lst.count(predictions_lst[0]) == len(predictions_lst):
            old_keys, keys_flag = keystroke(class_names[predicted_class], old_keys, keys_flag)
        num_frames = num_frames + 1
        cv2.waitKey(1)
    return num_frames


def pose_and_play(log_path, model_path, pose_estimation_model: PoseEstimation,
                  feature_engineering: FeatureEngineering, classifier: Classifier, camera_port=0, queue_size=5):
    """
    This function sets and loads the models, and starts predicting poses and simulating keys.
    In the end this function prints the FPS of playing with poses.
    :param log_path: given log path to find the class names.
    :param model_path: the classifier model path.
    :param pose_estimation_model: the pose estimation model to extract keypoints.
    :param feature_engineering: the feature engineering methods used to process keypoints.
    :param classifier: The classifier that classifies the processed input.
    :param camera_port: the port of the webcam used.
    :param queue_size: the queue size for simulating a keypress only when the queue is full.
    """
    classifier.load(model_path)
    class_names = find_class_names(log_path)
    pose_estimation_model.load_model()
    pose_estimation_model.start()
    camera = cv2.VideoCapture(camera_port)
    if get_numlock_state():  # set numlock as not pressed
        PressKey(0x45)
        ReleaseKey(0x45)
    start = time.time()
    num_frames = predict_poses(class_names, pose_estimation_model, feature_engineering, classifier, camera, queue_size)
    end = time.time()
    pose_estimation_model.end()
    camera.release()
    cv2.destroyAllWindows()
    seconds = end - start
    fps = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))


def find_pose_names(log_path):
    """
    This function finds the pose names in the given log path.
    :param log_path: the log path containing triplets of [pose_id, pose_name, keys]
    :return: the pose names.
    """
    with open(log_path, 'r') as f:
        data = f.readlines()
    class_names = []
    for i in range(1, len(data)):
        class_names.append(' '.join(data[i].split()[1:-1]))
    return class_names


def pose_and_print(log_path, model_path, pose_estimation_model: PoseEstimation,
                   feature_engineering: FeatureEngineering, classifier: Classifier, camera_port=0, queue_size=5):
    """
    This function is the same as pose_and_play, but instead of simulating keypresses it prints the poses names to
    the opencv window. Used by users for testing the system.
    :param log_path: given log path to find the class names.
    :param model_path: the classifier model path.
    :param pose_estimation_model: the pose estimation model to extract keypoints.
    :param feature_engineering: the feature engineering methods used to process keypoints.
    :param classifier: The classifier that classifies the processed input.
    :param camera_port: the port of the webcam used.
    :param queue_size: the queue size for simulating a keypress only when the queue is full.
    """
    classifier.load(model_path)
    class_names = find_pose_names(log_path)
    pose_estimation_model.load_model()
    pose_estimation_model.start()
    predictions_lst = [-1] * queue_size
    camera = cv2.VideoCapture(camera_port)
    num_frames = 0
    start = time.time()

    while True:
        status, frame = camera.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMG_SIZE)
        landmarks = pose_estimation_model.process_frame(image)
        model_input = feature_engineering(landmarks)
        prediction = classifier.predict(model_input)
        predicted_class = np.argmax(prediction)
        predictions_lst.pop(0)
        predictions_lst.append(predicted_class)
        if np.max(prediction) > THRESHOLD and predictions_lst.count(predictions_lst[0]) == len(predictions_lst):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, class_names[predicted_class], (50, 250), font, 3, (255, 0, 0), 4, cv2.LINE_AA)
        num_frames = num_frames + 1
        cv2.imshow("preview", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    end = time.time()
    pose_estimation_model.end()
    camera.release()
    cv2.destroyAllWindows()
    seconds = end - start
    fps = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))

