import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from movenet import movenet_preprocess_data
from blazepose import blazepose_preprocess_data
from efficientpose import efficientpose_preprocess_data

from cv2 import cv2
from mediapipe.python.solutions import pose as mp_pose
from blazepose_utils import FullBodyPoseEmbedder
from movenet_utils import load_movenet_model, movenet_inference_video, init_crop_region, determine_crop_region, feature_engineering
from tqdm import tqdm
from test_analysis import test1_pose_switches, test1_movements, test2_pose_switches, test2_movements
import joblib
import time

IMG_SIZE = (160, 160)
ENSEMBLE_MODELS = 5


def evaluate_model(model_name, pose_estimation_model, preprocess_data, test_dir="test_dataset", ml_model=False,
                   pose_estimation=True, preprocessing=True):
    """
    This function evaluates a given model accuracy on test sets in the test directory.
    :param model_name: Name of the model to evaluate - It is the path to the model without saved_models/
    :param pose_estimation_model: Name of the pose estimation model used for evaluation.
    :param preprocess_data: Function for preprocessing data by the pose estimation model.
    :param test_dir: directory containing the test sets.
    :param ml_model: boolean flag representing if we are using a machine learning model or deep learning model.
    :param pose_estimation: boolean flag representing if we are using a pose estimation model or image classification model.
    :param preprocessing: boolean flag representing if we preprocess the images or load preprocessed images data.
    :return: A dictionary containing the accuracy results for each of the tests in test_dir.
    """
    if not ml_model:
        model = tf.keras.models.load_model("saved_models/"+model_name)
    else:
        model = joblib.load(f'saved_models/{model_name}.joblib')
    results_dic = {}
    test_data = os.listdir(test_dir)
    if preprocessing:
        for i, test_folder in enumerate(test_data):
            test_folder_path = os.path.join(test_dir, test_folder)
            if pose_estimation:
                X, y, _ = preprocess_data(data_directory=test_folder_path, ml_model=ml_model)
                np.save(f"preprocessing/{pose_estimation_model}_X_test{i+1}.npy", X)
                np.save(f"preprocessing/{pose_estimation_model}_y_test{i+1}.npy", y)
                if not ml_model:
                    loss, accuracy = model.evaluate(X, y)
                else:
                    accuracy = model.score(X, y)
                    print(accuracy)
            else:
                test_dataset = tf.keras.utils.image_dataset_from_directory(test_folder_path, image_size=IMG_SIZE)
                AUTOTUNE = tf.data.AUTOTUNE
                test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
                loss, accuracy = model.evaluate(test_dataset)
            results_dic[test_folder[:5]] = accuracy * 100
    else:
        for i, test_folder in enumerate(test_data):
            X = np.load(f"preprocessing/{pose_estimation_model}_X_test{i+1}.npy")
            y = np.load(f"preprocessing/{pose_estimation_model}_y_test{i+1}.npy")
            if not ml_model:
                X = [X for _ in range(ENSEMBLE_MODELS)]
                loss, accuracy = model.evaluate(X, y)
            else:
                X = X.reshape()
                accuracy = model.score(X, y)
                print(accuracy)
            results_dic[test_folder[:5]] = accuracy * 100
    return results_dic


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], "{:.2f}".format(y[i]), ha='center')


def results_bar_plot(model_name, results_dic):
    keys = list(results_dic.keys())
    values = list(results_dic.values())
    plt.figure(figsize=(10, 5))
    plt.bar(keys, values)
    addlabels(keys, values)
    plt.title(model_name + " accuracy on test sets")
    plt.xlabel("test sets")
    plt.ylabel("test accuracy")
    plt.show()


def models_bar_plot(vals1, vals2, vals3, vals4):
    # set width of bar
    bar_width = 0.2
    fig = plt.subplots(figsize=(13, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(vals1))
    br2 = [x + bar_width for x in br1]
    br3 = [x + bar_width for x in br2]
    br4 = [x + bar_width for x in br3]

    # Make the plot
    plt.bar(br1, vals1, color='darkorange', width=bar_width, edgecolor='grey', label='MobileNet')
    for i in range(len(vals1)):
        plt.text(i, vals1[i], "{:.1f}".format(vals1[i]), ha='center')
    plt.bar(br2, vals2, color='royalblue', width=bar_width, edgecolor='grey', label='MoveNet')
    for i in range(len(vals2)):
        plt.text(i+0.2, vals2[i], "{:.1f}".format(vals2[i]), ha='center')
    plt.bar(br3, vals3, color='firebrick', width=bar_width, edgecolor='grey', label='BlazePose')
    for i in range(len(vals3)):
        plt.text(i+0.4, vals3[i], "{:.1f}".format(vals3[i]), ha='center')
    plt.bar(br4, vals4, color='springgreen', width=bar_width, edgecolor='grey', label='EfficientPose')
    for i in range(len(vals4)):
        plt.text(i+0.6, vals4[i], "{:.1f}".format(vals4[i]), ha='center')

    # Adding Xticks
    plt.xlabel('Tests', fontweight='bold', fontsize = 15)
    plt.ylabel('Accuracy', fontweight='bold', fontsize=15)
    plt.xticks([r + bar_width for r in range(len(vals1))],
               ['test 1\nsame background\ndifferent clothes', 'test 2\ndifferent background\nsame lighting',
                'test 3\ndifferent background\ndifferent lighting', 'test 4\noutside',
                'test 5\nbad lighting', 'test 6\n different perspective'])

    plt.legend(bbox_to_anchor=(1.1,1.05))
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """Plots the confusion matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    sns.set(style="white")
    plt.subplots(figsize=(10, 10))
    heatmap = sns.heatmap(cm, cmap=cmap, center=0, square=True, linewidths=.5, annot=True, fmt='d')
    heatmap.set_title(title)
    heatmap.xaxis.set_ticklabels(classes, rotation='vertical')
    heatmap.yaxis.set_ticklabels(classes, rotation='horizontal')
    heatmap.set_xlabel('Predicted label')
    heatmap.set_ylabel('True label')
    plt.show()


def model_confusion_matrix(model_name, preprocess_data, test_folder):
    model = tf.keras.models.load_model("saved_models/"+model_name)
    class_names = os.listdir(test_folder)
    # Classify pose in the TEST dataset using the trained model
    X_test, y_test, _ = preprocess_data(data_directory=test_folder)
    y_pred = model.predict(X_test)

    # Convert the prediction result to class name
    y_pred_label = [class_names[i] for i in np.argmax(y_pred, axis=1)]
    y_true_label = [class_names[i] for i in np.argmax(y_test, axis=1)]

    # Plot the confusion matrix
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    plot_confusion_matrix(cm, class_names, title=model_name + ' confusion matrix')

    # Print the classification report
    print('\nClassification Report:\n', classification_report(y_true_label,
                                                              y_pred_label))


def pose_estimation_preprocessing(model, pose_estimation_model, test_dir, threshold, test_movements, test_pose_switches,
                                  predictions_lst, ml_model, preprocess_name):
    if pose_estimation_model == "blazepose":
        pose_embedder = FullBodyPoseEmbedder()
        pose_tracker = mp_pose.Pose()
    else:
        movenet, input_size = load_movenet_model("movenet_thunder")
        crop_region = init_crop_region(256, 256)

    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
    pose_images = ['img'+str(i)+'.png' for i in range(len(os.listdir(test_dir)))]
    X = []
    for img_num, pose_image in enumerate(tqdm(pose_images)):
        image = cv2.imread(os.path.join(test_dir, pose_image))
        image_height, image_width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if pose_estimation_model == "blazepose":
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
            model_input = feature_engineering(model_input)

        X.append(model_input)
        true_positive, true_negative, false_positive, false_negative = \
            calculate_metrics(model, model_input, predictions_lst, threshold, test_movements, test_pose_switches,
                              img_num, true_positive, true_negative, false_positive, false_negative, ml_model)
    if pose_estimation_model == "blazepose":
        pose_tracker.close()
    X = np.array(X)
    np.save(f"preprocessing/{pose_estimation_model}_{preprocess_name}.npy", X)
    return true_positive, true_negative, false_positive, false_negative


def calculate_metrics(model, model_input, predictions_lst, threshold, test_movements, test_pose_switches,
                      img_num, true_positive, true_negative, false_positive, false_negative, ml_model=False):
    class_num = test_movements(img_num)
    if not ml_model:
        predict_frame = np.expand_dims(model_input, axis=0)
        predict_frame = [predict_frame for _ in range(ENSEMBLE_MODELS)]
        prediction = model(predict_frame, training=False)
    else:
        prediction = model.predict_proba(model_input.reshape(1, -1))
    predicted_class = np.argmax(prediction)

    predictions_lst.pop(0)
    predictions_lst.append(predicted_class)

    if np.max(prediction) > threshold and predictions_lst.count(predictions_lst[0]) == len(predictions_lst):
        if predicted_class == class_num:
            true_positive += 1
        else:
            false_positive += 1
    else:
        if img_num in test_pose_switches:
            true_negative += 1
        else:
            false_negative += 1
    return true_positive, true_negative, false_positive, false_negative


def evaluate_model_threshold(model_name, pose_estimation_model, threshold, test_dir, test_movements, test_pose_switches,
                             ml_model=False, preprocessing=True, preprocess_name="video2"):
    if not ml_model:
        model = tf.keras.models.load_model("saved_models/"+model_name)
    else:
        model = joblib.load(f'saved_models/{model_name}.joblib')
    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
    predictions_lst = [-1] * 4

    if preprocessing:
        true_positive, true_negative, false_positive, false_negative = \
            pose_estimation_preprocessing(model, pose_estimation_model, test_dir, threshold, test_movements,
                                          test_pose_switches, predictions_lst, ml_model, preprocess_name)
    else:
        X = np.load(f"preprocessing/{pose_estimation_model}_{preprocess_name}.npy")
        for img_num, model_input in enumerate(tqdm(X)):
            true_positive, true_negative, false_positive, false_negative = \
                calculate_metrics(model, model_input, predictions_lst, threshold, test_movements, test_pose_switches,
                                  img_num, true_positive, true_negative, false_positive, false_negative, ml_model)

    correct = true_positive + true_negative
    total = true_positive + false_positive + true_negative + false_negative
    accuracy = 100*correct/total
    precision = 100*true_positive/(true_positive + false_positive)
    recall = 100*true_positive/(true_positive + false_negative)
    F1_score = 100*2*true_positive/(2*true_positive + false_positive + false_negative)

    print(model_name + " accuracy : ", accuracy)
    print(model_name + " precision : ", precision)
    print(model_name + " recall : ", recall)
    print(model_name + " F1-score : ", F1_score)

    return accuracy, precision, recall, F1_score


def plot_model_metrics(model_name, pose_estimation_model, test_dir, test_movements, test_pose_switches,
                       ml_model=False, preprocessing=True, preprocess_name="video2"):
    accuracy_list, precision_list, recall_list, F1_list = [], [], [], []
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]  # 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
    for threshold in thresholds:
        accuracy, precision, recall, F1_score = \
            evaluate_model_threshold(model_name, pose_estimation_model, threshold, test_dir, test_movements,
                                     test_pose_switches, ml_model, preprocessing, preprocess_name)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        F1_list.append(F1_score)

    plt.plot(thresholds, accuracy_list, label="accuracy")
    plt.plot(thresholds, precision_list, label="precision")
    plt.plot(thresholds, recall_list, label="recall")
    plt.plot(thresholds, F1_list, label="F1 score")
    plt.title(model_name + " metrics")
    plt.ylabel("metric percentage")
    plt.xlabel("threshold")
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()


def print_poses(model_name, pose_estimation_model, ml_model=False, camera_port=0):
    if not ml_model:
        model = tf.keras.models.load_model("saved_models/"+model_name)
    else:
        model = joblib.load(f'saved_models/{model_name}.joblib')
    class_names = os.listdir("dataset")
    #model = tf.keras.models.load_model("saved_models/model2")
    #class_names = find_pose_names("logs/log2.txt")
    image_height, image_width = IMG_SIZE[0], IMG_SIZE[1]
    if pose_estimation_model == "blazepose":
        pose_embedder = FullBodyPoseEmbedder()
        pose_tracker = mp_pose.Pose()
    else:
        movenet, input_size = load_movenet_model("movenet_thunder")
        crop_region = init_crop_region(image_height, image_width)
    predictions_lst = [-1] * 5
    camera = cv2.VideoCapture(camera_port)
    num_frames = 0
    start = time.time()
    while True:
        status, frame = camera.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMG_SIZE)
        if pose_estimation_model == "blazepose":
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
            model_input = feature_engineering(model_input)
        if not ml_model:
            predict_frame = np.expand_dims(model_input, axis=0)
            predict_frame = [predict_frame for _ in range(5)]
            prediction = model(predict_frame, training=False)
        else:
            prediction = model.predict_proba(model_input.reshape(1, -1))
        predicted_class = np.argmax(prediction)
        predictions_lst.pop(0)
        predictions_lst.append(predicted_class)
        if np.max(prediction) > 0.9 and predictions_lst.count(predictions_lst[0]) == len(predictions_lst):
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
    if pose_estimation_model == "blazepose":
        pose_tracker.close()
    end = time.time()
    camera.release()
    cv2.destroyAllWindows()
    seconds = end - start
    fps = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))


if __name__ == '__main__':
    # individual bar plots:
    accuracy_dic = evaluate_model("movenet_ensemble/ensemble", "movenet", movenet_preprocess_data,
                                  ml_model=False, preprocessing=False)
    results_bar_plot("movenet ensemble Nadam ", accuracy_dic)

    # all models bar plot:
    """
    mobilenet_dic = evaluate_model("mobilenet", None)
    movenet_dic = evaluate_model("movenet", movenet_preprocess_data)
    blazepose_dic = evaluate_model("blazepose", blazepose_preprocess_data)
    efficientpose_dic = evaluate_model("efficientpose", efficientpose_preprocess_data)
    models_bar_plot(list(mobilenet_dic.values()), list(movenet_dic.values()),
                    list(blazepose_dic.values()), list(efficientpose_dic.values()))
    """

    # confusion matrix:
    """
    model_confusion_matrix("movenet", movenet_preprocess_data, "test_dataset/test3-diff_back_diff_lighting")
    model_confusion_matrix("blazepose", blazepose_preprocess_data, "test_dataset/test7-ultimate")
    """

    # ultimate test:
    evaluate_model_threshold("movenet_ensemble/ensemble", "movenet", 0.9, "test_video/test2-long", test2_movements,
                             test2_pose_switches, ml_model=False, preprocessing=False, preprocess_name="video2")
    #evaluate_model_threshold("blazepose", "blazepose", 0.95, "test_video/test2-long", test2_movements,
    #                         test2_pose_switches, ml_model=False, preprocessing=False, preprocess_name="video2")
    #plot_model_metrics("movenet_ensemble/ensemble", "movenet", "test_video/test2-long", test2_movements,
    #                   test2_pose_switches, ml_model=False, preprocessing=False, preprocess_name="video2")
    #plot_model_metrics("blazepose", "test_video/test1-ultimate", test1_movements, test1_pose_switches)