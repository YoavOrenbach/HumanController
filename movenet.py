import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from movenet_utils import load_movenet_model, movenet_inference, feature_engineering, landmarks_to_embedding
from movenet_utils import movenet_inference_video, init_crop_region, determine_crop_region
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
import joblib

# useful links:
# https://tfhub.dev/s?q=movenet - tf hub for moveNet
# https://www.tensorflow.org/hub/tutorials/movenet - moveNet tutorial
# https://www.tensorflow.org/lite/tutorials/pose_classification - pose classification with moveNet

LEARNING_RATE = 0.001
IMG_SIZE = (256, 256)


def movenet_preprocess_data(data_directory="dataset", static=False):
    model_name = "movenet_thunder"
    movenet, input_size = load_movenet_model(model_name)
    poses_directories = os.listdir(data_directory)
    landmarks_list = []
    label_list = []
    class_num = 0
    for pose_directory in tqdm(poses_directories):
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
            landmarks[0][0][:, :2] *= image_height
            #landmarks_list.append(landmarks)
            #embedding = landmarks_to_embedding(landmarks).numpy()
            embedding = feature_engineering(landmarks)
            landmarks_list.append(embedding)
            label_list.append(class_num)
        class_num = class_num + 1

    landmarks_array = np.asarray(landmarks_list)
    label_array = np.asarray(label_list)
    #categorical_labels = tf.keras.utils.to_categorical(label_array)
    #return landmarks_array, categorical_labels, class_num

    landmarks_array = landmarks_array.reshape(landmarks_array.shape[0], landmarks_array.shape[1]*landmarks_array.shape[2])
    return landmarks_array, label_array, class_num


def define_model(num_classes):
    #inputs = tf.keras.Input(shape=(1, 1, 17, 3))
    #embedding = landmarks_to_embedding(inputs)
    inputs = tf.keras.Input(shape=(78, 2))
    embedding = tf.keras.layers.Flatten()(inputs)
    layer = tf.keras.layers.Dense(256, activation='relu')(embedding)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    layer = tf.keras.layers.Dense(128, activation='relu')(layer)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    layer = tf.keras.layers.Dense(64, activation='relu')(layer)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(layer)
    model = tf.keras.Model(inputs, outputs)
    model.summary()
    return model


def train_model(model, X_train, X_val, y_train, y_val):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)
    history = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        callbacks=[earlystopping])
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
    X, y, num_classes = movenet_preprocess_data(data_directory="dataset_enhanced/300", static=False)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    #X_val, y_val, _ = movenet_preprocess_data(data_directory="test_dataset/test7-ultimate")
    model = define_model(num_classes)
    history = train_model(model, X_train, X_val, y_train, y_val)
    plot_train_test(history, "MoveNet")
    model.save("saved_models/movenet")


def knn():
    X_train, y_train, num_classes = movenet_preprocess_data(data_directory="dataset_enhanced/300", static=False)
    model = KNeighborsClassifier(n_neighbors=1, weights='uniform', leaf_size=20, metric='manhattan')
    model.fit(X_train, y_train)
    joblib.dump(model, 'saved_models/movenet_knn.joblib')


if __name__ == '__main__':
    knn()
