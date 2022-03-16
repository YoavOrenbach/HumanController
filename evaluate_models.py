import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from movenet import movenet_preprocess_data
from blazepose import blazepose_preprocess_data
from efficientpose import efficientpose_preprocess_data
from train_model import preprocess_data

from cv2 import cv2
from mediapipe.python.solutions import pose as mp_pose
from blazepose_utils import FullBodyPoseEmbedder
from movenet_utils import load_movenet_model, movenet_inference_video, init_crop_region, determine_crop_region, feature_engineering
import re
from tqdm import tqdm
from test_analysis import test1_pose_switches, test1_movements, test2_pose_switches, test2_movements
import joblib

IMG_SIZE = (160, 160)


def evaluate_model(model_name, preprocess_data):
    results_dic = {}
    model = tf.keras.models.load_model("saved_models/"+model_name)
    test_data = os.listdir("test_dataset")
    for test_folder in test_data:
        test_folder_path = os.path.join("test_dataset", test_folder)
        if model_name != "mobilenet":
            X, y, _ = preprocess_data(data_directory=test_folder_path)
            loss, accuracy = model.evaluate(X, y)
        else:
            test_dataset = tf.keras.utils.image_dataset_from_directory(test_folder_path, image_size=IMG_SIZE)
            AUTOTUNE = tf.data.AUTOTUNE
            test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
            loss, accuracy = model.evaluate(test_dataset)
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


def evaluate_model_threshold(model_name, threshold, test_dir, test_movements, test_pose_switches):
    #model = tf.keras.models.load_model("saved_models/"+model_name)
    model = joblib.load(f'saved_models/{model_name}_knn.joblib')
    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
    if model_name == "blazepose":
        pose_embedder = FullBodyPoseEmbedder()
        pose_tracker = mp_pose.Pose()
    else:
        movenet, input_size = load_movenet_model("movenet_thunder")
        crop_region = init_crop_region(256, 256)
    predictions_lst = [-1, -1]
    #pose_images = os.listdir(test_dir)
    pose_images = ['img'+str(i)+'.png' for i in range(len(os.listdir(test_dir)))]
    for img_num, pose_image in enumerate(tqdm(pose_images)):
        #img_num = int(re.search(r'\d+', pose_image)[0])
        class_num = test_movements(img_num)
        image = cv2.imread(os.path.join(test_dir, pose_image))
        image_height, image_width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
            model_input = feature_engineering(model_input)
        predict_frame = np.expand_dims(model_input, axis=0)
        #prediction = model(predict_frame, training=False)
        prediction = model.predict_proba(model_input.reshape(1,-1))
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
    if model_name == "blazepose":
        pose_tracker.close()
    correct = true_positive + true_negative
    total = true_positive + false_positive + true_negative + false_negative
    accuracy = 100*correct/total
    precision = 100*true_positive/(true_positive + false_positive)
    recall = 100*true_positive/(true_positive + false_negative)
    F1_score = 100*2*true_positive/(2*true_positive + false_positive + false_negative)

    print(model_name + " accuracy : ", 100*correct/total)
    print(model_name + " precision : ", 100*true_positive/(true_positive + false_positive))
    print(model_name + " recall : ", 100*true_positive/(true_positive + false_negative))
    print(model_name + " F1-score : ", 100*2*true_positive/(2*true_positive + false_positive + false_negative))

    return accuracy, precision, recall, F1_score


def plot_model_metrics(model_name, test_dir, test_movements, test_pose_switches):
    accuracy_list, precision_list, recall_list, F1_list = [], [], [], []
    thresholds = [0.6, 0.7, 0.8, 0.90, 0.99]  # 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
    for threshold in thresholds:
        accuracy, precision, recall, F1_score = evaluate_model_threshold(model_name, threshold, test_dir,
                                                                         test_movements, test_pose_switches)
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


def evaluate_machine_learning(model_name, preprocess_data):
    results_dic = {}
    model = joblib.load(f'saved_models/{model_name}_knn.joblib')
    test_data = os.listdir("test_dataset")
    for test_folder in test_data:
        test_folder_path = os.path.join("test_dataset", test_folder)
        X, y, _ = preprocess_data(data_directory=test_folder_path)
        accuracy = model.score(X, y)
        results_dic[test_folder[:5]] = accuracy * 100
        print(accuracy)
    return results_dic


if __name__ == '__main__':
    # individual bar plots:

    #movenet_dic = evaluate_model("movenet", movenet_preprocess_data)
    #results_bar_plot("movenet", movenet_dic)
    #blazepose_dic = evaluate_model("blazepose", blazepose_preprocess_data)
    #results_bar_plot("blazepose+mlp", blazepose_dic)
    #movenet_dic = evaluate_machine_learning("movenet", movenet_preprocess_data)
    #results_bar_plot("movenet+KNN", movenet_dic)
    #blazepose_dic = evaluate_machine_learning('blazepose', blazepose_preprocess_data)
    #results_bar_plot("blazepose+KNN", blazepose_dic)

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
    #evaluate_model_threshold("movenet", 0.95, "test_video/test2-long", test2_movements, test2_pose_switches)
    #evaluate_model_threshold("blazepose", 0.95, "test_video/test2-long", test2_movements, test2_pose_switches)
    #plot_model_metrics("movenet", "test_video/test1-ultimate", test1_movements, test1_pose_switches)
    #plot_model_metrics("blazepose", "test_video/test1-ultimate", test1_movements, test1_pose_switches)
