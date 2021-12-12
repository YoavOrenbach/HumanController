import tensorflow as tf
import os
import matplotlib.pyplot as plt
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
        results_dic[test_folder[:5]] = accuracy
    return results_dic


def results_bar_plot(model_name, results_dic):
    keys = list(results_dic.keys())
    values = list(results_dic.values())
    plt.bar(keys, values)
    plt.title(model_name + " accuracy on test sets")
    plt.xlabel("test sets")
    plt.ylabel("test accuracy")
    plt.show()


if __name__ == '__main__':
    dic = evaluate_model("movenet", movenet_preprocess_data)
    results_bar_plot("MoveNet", dic)
