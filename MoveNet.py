import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt
from movenetUtils import load_movenet_model, movenet_inference, landmarks_to_embedding
from movenetUtils import movenet_inference_video, init_crop_region, determine_crop_region

# useful links:
# https://tfhub.dev/s?q=movenet - tf hub for moveNet
# https://www.tensorflow.org/hub/tutorials/movenet - moveNet tutorial
# https://www.tensorflow.org/lite/tutorials/pose_classification - pose classification with moveNet

LEARNING_RATE = 0.01
IMG_SIZE = (256, 256)


def movenet_preprocess_data(movenet, input_size, data_directory="dataset", static=True):
    poses_directories = os.listdir(data_directory)
    landmarks_list = []
    label_list = []
    class_num = 0
    for pose_directory in poses_directories:
        image_height, image_width = IMG_SIZE[0], IMG_SIZE[1]
        crop_region = init_crop_region(image_height, image_width)
        pose_images_path = os.path.join(data_directory, pose_directory)
        pose_images = os.listdir(pose_images_path)
        for pose_image in pose_images:
            image = tf.io.read_file(os.path.join(pose_images_path, pose_image))
            image = tf.image.decode_jpeg(image)
            if static:
                landmarks = movenet_inference(image, movenet, input_size)
            else:
                landmarks = movenet_inference_video(movenet, image, crop_region, crop_size=[input_size, input_size])
                crop_region = determine_crop_region(landmarks, image_height, image_width)
            landmarks_list.append(landmarks)
            label_list.append(class_num)
        class_num = class_num + 1

    landmarks_array = np.asarray(landmarks_list)
    label_array = np.asarray(label_list)
    categorical_labels = tf.keras.utils.to_categorical(label_array)
    return landmarks_array, categorical_labels, class_num


def define_model(num_classes):
    inputs = tf.keras.Input(shape=(1, 1, 17, 3))
    flatten = tf.keras.layers.Flatten()(inputs)
    embedding = landmarks_to_embedding(flatten)
    layer = tf.keras.layers.Dense(128, activation='relu')(embedding)
    #layer = tf.keras.layers.Dropout(0.5)(layer)
    layer = tf.keras.layers.Dense(64, activation='relu')(layer)
    #layer = tf.keras.layers.Dropout(0.5)(layer)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(layer)
    model = tf.keras.Model(inputs, outputs)
    model.summary()
    return model


def train_model(model, X_train, X_val, y_train, y_val):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train,
                        epochs=20,
                        batch_size=16,
                        validation_data=(X_val, y_val))
    return history


def plot_train_test(history, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.suptitle(model_name)
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def movenet():
    model_name = "movenet_thunder"
    movenet, input_size = load_movenet_model(model_name)
    X, y, num_classes = movenet_preprocess_data(movenet, input_size, static=False)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    model = define_model(num_classes)
    history = train_model(model, X_train, X_val, y_train, y_val)
    plot_train_test(history, "MoveNet")
    model.save("movenet_saved_model")


def evaluate_model():
    model = tf.keras.models.load_model("movenet_saved_model")
    model_name = "movenet_thunder"
    movenet, input_size = load_movenet_model(model_name)

    results_dic = {}
    test_data = os.listdir("testDataset")
    for test_folder in test_data:
        test_folder_path = os.path.join("testDataset", test_folder)
        X, y, _ = movenet_preprocess_data(movenet, input_size, data_directory=test_folder_path, static=False)
        loss, accuracy = model.evaluate(X, y)
        results_dic[test_folder[:5]] = accuracy

    keys = list(results_dic.keys())
    values = list(results_dic.values())
    plt.bar(keys, values)
    plt.title("BlazePose accuracy on test sets")
    plt.xlabel("test sets")
    plt.ylabel("test accuracy")
    plt.show()


if __name__ == '__main__':
    movenet()