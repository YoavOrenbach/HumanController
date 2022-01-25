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
from movenet_utils import load_movenet_model, movenet_inference_video, init_crop_region, determine_crop_region
import re
from tqdm import tqdm

IMG_SIZE = (160, 160)


def evaluate_model(model_name, preprocess_data):
    results_dic = {}
    model = tf.keras.models.load_model(model_name + "_saved_model")
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
    heatmap = sns.heatmap(cm, cmap=cmap, center=0, square=True, linewidths=.5, annot=True)
    heatmap.set_title(title)
    heatmap.xaxis.set_ticklabels(classes, rotation='vertical')
    heatmap.yaxis.set_ticklabels(classes, rotation='horizontal')
    heatmap.set_xlabel('Predicted label')
    heatmap.set_ylabel('True label')
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    """
    plt.show()


def model_confusion_matrix(model_name, preprocess_data, test_folder):
    model = tf.keras.models.load_model(model_name + "_saved_model")
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


def analyze_movements(img_num):
    class_num = 0
    if 0 <= img_num <= 8:
        class_num = 12
    elif 9 <= img_num <= 20:
        class_num = 22
    elif 21 <= img_num <= 38:
        class_num = 23
    elif 39 <= img_num <= 55:
        class_num = 2
    elif 56 <= img_num <= 68:
        class_num = 1
    elif 69 <= img_num <= 76:
        class_num = 12
    elif 77 <= img_num <= 86:
        class_num = 11
    elif 87 <= img_num <= 99:
        class_num = 9
    elif 100 <= img_num <= 106:
        class_num = 12
    elif 107 <= img_num <= 121:
        class_num = 21
    elif 122 <= img_num <= 130:
        class_num = 19
    elif 131 <= img_num <= 138:
        class_num = 12
    elif 139 <= img_num <= 159:
        class_num = 7
    elif 160 <= img_num <= 182:
        class_num = 8
    elif 183 <= img_num <= 186:
        class_num = 12
    elif 187 <= img_num <= 211:
        class_num = 6
    elif 212 <= img_num <= 214:
        class_num = 12
    elif 215 <= img_num <= 233:
        class_num = 5
    elif 234 <= img_num <= 242:
        class_num = 12
    elif 243 <= img_num <= 262:
        class_num = 18
    elif 263 <= img_num <= 279:
        class_num = 15
    elif 280 <= img_num <= 294:
        class_num = 16
    elif 295 <= img_num <= 297:
        class_num = 18
    elif 298 <= img_num <= 304:
        class_num = 12
    elif 305 <= img_num <= 318:
        class_num = 17
    elif 319 <= img_num <= 326:
        class_num = 12
    elif 327 <= img_num <= 344:
        class_num = 14
    elif 345 <= img_num <= 346:
        class_num = 24
    elif 347 <= img_num <= 364:
        class_num = 3
    elif 365 <= img_num <= 375:
        class_num = 12
    elif 375 <= img_num <= 407:
        class_num = 13
    return class_num


def evaluate_ultimate_model(model_name, threshold):
    model = tf.keras.models.load_model(model_name + "_saved_model")
    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
    negative_images = [7,8,9,20,21,22,38,39,55,56,57,68,69,75,76,77,87,99,100,105,106,107,122,131,132,138,139,159,160,
                       183,187,188,211,212,213,214,215,233,234,235,236,241,242,243,260,261,262,263,264,265,266,277,278,
                       279,280,281,294,295,296,297,298,304,305,306,316,317,318,319,320,321,325,326,327,344,345,346,347,
                       348,364,365,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389]
    if model_name == "blazepose":
        pose_embedder = FullBodyPoseEmbedder()
        pose_tracker = mp_pose.Pose()
    else:
        movenet, input_size = load_movenet_model("movenet_thunder")
        crop_region = init_crop_region(256, 256)
    test_dir = "test_ultimate/all_keys"
    pose_images = os.listdir(test_dir)
    for pose_image in tqdm(pose_images):
        img_num = int(re.search(r'\d+', pose_image)[0])
        class_num = analyze_movements(img_num)
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
        predict_frame = np.expand_dims(model_input, axis=0)
        prediction = model.predict(predict_frame)
        predicted_class = np.argmax(prediction)
        if np.max(prediction) > threshold:
            if predicted_class == class_num:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if img_num in negative_images:
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
    """
    print(model_name + " accuracy : ", 100*correct/total)
    print(model_name + " precision : ", 100*true_positive/(true_positive + false_positive))
    print(model_name + " recall : ", 100*true_positive/(true_positive + false_negative))
    print(model_name + " F1-score : ", 100*2*true_positive/(2*true_positive + false_positive + false_negative))
    """
    return accuracy, precision, recall, F1_score


def plot_ultimate_model(model_name):
    accuracy_list, precision_list, recall_list, F1_list = [], [], [], []
    thresholds = [0.6, 0.7, 0.8, 0.90, 0.99] # 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
    for threshold in thresholds:
        accuracy, precision, recall, F1_score = evaluate_ultimate_model(model_name, threshold)
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


if __name__ == '__main__':
    # individual bar plots:
    """
    movenet_dic = evaluate_model("movenet", movenet_preprocess_data)
    results_bar_plot("movenet", movenet_dic)
    #blazepose_dic = evaluate_model("blazepose", blazepose_preprocess_data)
    #results_bar_plot("blazepose", blazepose_dic)
    """

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
    model_confusion_matrix("blazepose", blazepose_preprocess_data, "test_dataset/test5-bad_lighting")
    """

    # ultimate test:
    #evaluate_ultimate_model("blazepose")
    #evaluate_ultimate_model("movenet")
    plot_ultimate_model("blazepose")
