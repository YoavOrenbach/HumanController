import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from movenet import movenet_preprocess_data
from blazepose import blazepose_preprocess_data
from efficientpose import efficientpose_preprocess_data

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


if __name__ == '__main__':
    dic = evaluate_model("movenet", movenet_preprocess_data)
    results_bar_plot("MoveNet thunder - distances", dic)
    #model_confusion_matrix("blazepose", blazepose_preprocess_data, "test_dataset/test1-different_clothes")
