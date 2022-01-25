import tensorflow as tf
import numpy as np
import os
from movenet_utils import load_movenet_model, movenet_inference, landmarks_to_embedding
from movenet_utils import movenet_inference_video, init_crop_region, determine_crop_region
from tqdm import tqdm

LEARNING_RATE = 0.001
IMG_SIZE = (256, 256)


def preprocess_data(data_directory="dataset"):
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
            landmarks = movenet_inference_video(movenet, image, crop_region, crop_size=[input_size, input_size])
            crop_region = determine_crop_region(landmarks, image_height, image_width)
            landmarks[0][0][:, :2] *= image_height
            landmarks_list.append(landmarks)
            label_list.append(class_num)
        class_num = class_num + 1

    landmarks_array = np.asarray(landmarks_list)
    label_array = np.asarray(label_list)
    categorical_labels = tf.keras.utils.to_categorical(label_array)
    return landmarks_array, categorical_labels, class_num


def define_model(num_classes):
    inputs = tf.keras.Input(shape=(1, 1, 17, 3))
    embedding = landmarks_to_embedding(inputs)
    layer = tf.keras.layers.Dense(256, activation='relu')(embedding)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    layer = tf.keras.layers.Dense(128, activation='relu')(layer)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    layer = tf.keras.layers.Dense(64, activation='relu')(layer)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(layer)
    model = tf.keras.Model(inputs, outputs)
    return model


def train(model, X_train, y_train):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train,
              epochs=50,
              batch_size=32)
    model.save("saved_model")


def train_movenet():
    X, y, num_classes = preprocess_data()
    model = define_model(num_classes)
    train(model, X, y)


if __name__ == '__main__':
    train_movenet()
