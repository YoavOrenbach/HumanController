import numpy as np
from cv2 import cv2
from sklearn.model_selection import train_test_split
import os
import string
from tqdm import tqdm
from pose_estimation_models.pose_estimation_logic import PoseEstimation
from feature_engineering.feature_engineering_logic import FeatureEngineering
from classifiers.classifier_logic import Classifier


def preprocess_data(data_directory: string, pose_estimation_model: PoseEstimation,
                    feature_engineering: FeatureEngineering):
    """
    This function preprocesses the pose data in the given data directory. It passes each image to the pose estimation
    model to extract keypoints, then process them with the feature engineering object.
    :param data_directory: the directory from which to process pose data.
    :param pose_estimation_model: the pose estimation model to extract keypoints.
    :param feature_engineering: the feature engineering methods used to process keypoints.
    :return: the processed keypoints landmarks array (X), a matching label array (y), and the number of classes.
    """
    pose_estimation_model.load_model()
    poses_directories = ['pose'+str(i+1) for i in range(len(os.listdir(data_directory)))]
    landmarks_list = []
    label_list = []
    class_num = 0
    for pose_directory in tqdm(poses_directories):
        pose_images_path = os.path.join(data_directory, pose_directory)
        pose_images = os.listdir(pose_images_path)
        pose_estimation_model.start()
        for pose_image in pose_images:
            image = cv2.imread(os.path.join(pose_images_path, pose_image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            landmarks = pose_estimation_model.process_frame(image)
            if landmarks is None:
                continue
            embedding = feature_engineering(landmarks)
            landmarks_list.append(embedding)
            label_list.append(class_num)
        class_num = class_num + 1
        pose_estimation_model.end()
    landmarks_array = np.asarray(landmarks_list)
    label_array = np.asarray(label_list)
    return landmarks_array, label_array, class_num


def controller_model(data_directory, model_path, pose_estimation_model: PoseEstimation,
                     feature_engineering: FeatureEngineering, classifier: Classifier):
    """
    This function trains the given classifier object and saves it for playing with the poses.
    :param data_directory: the directory from which to process pose data.
    :param model_path: the path at which to save the classifier.
    :param pose_estimation_model: the pose estimation model to extract keypoints.
    :param feature_engineering: the feature engineering methods used to process keypoints.
    :param classifier: the classifier to train.
    :return:
    """
    X, y, num_classes = preprocess_data(data_directory, pose_estimation_model, feature_engineering)
    X, y = classifier.prepare_input(X, y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    classifier.define_model(feature_engineering.get_input_size(), num_classes)
    classifier.train_model(X_train, X_val, y_train, y_val)
    classifier.save(model_path)
