import tensorflow as tf
import numpy as np
import os
from movenet_utils import load_movenet_model, movenet_inference, landmarks_to_embedding

LEARNING_RATE = 0.01


def preprocess_data(movenet, input_size):
    files = os.listdir("dataset")
    img_list = []
    label_list = []
    class_num = 0
    for file in files:
        images_path = os.path.join("dataset", file)
        images = os.listdir(images_path)
        for img in images:
            image = tf.io.read_file(os.path.join(images_path, img))
            image = tf.image.decode_jpeg(image)
            landmarks = movenet_inference(image, movenet, input_size)
            img_list.append(landmarks)
            label_list.append(class_num)
        class_num = class_num + 1

    img_array = np.asarray(img_list)
    label_array = np.asarray(label_list)
    categorical_labels = tf.keras.utils.to_categorical(label_array)
    return img_array, categorical_labels, class_num


def define_model(num_classes):
    inputs = tf.keras.Input(shape=(1, 1, 17, 3))
    flatten = tf.keras.layers.Flatten()(inputs)
    embedding = landmarks_to_embedding(flatten)
    layer = tf.keras.layers.Dense(128, activation='relu')(embedding)
    layer = tf.keras.layers.Dense(64, activation='relu')(layer)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(layer)
    model = tf.keras.Model(inputs, outputs)
    return model


def train(model, X_train, y_train):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train,
              epochs=20,
              batch_size=16)
    model.save("saved_model")


def train_movenet():
    model_name = "movenet_thunder"
    movenet, input_size = load_movenet_model(model_name)
    X, y, num_classes = preprocess_data(movenet, input_size)
    model = define_model(num_classes)
    train(model, X, y)


if __name__ == '__main__':
    train_movenet()
