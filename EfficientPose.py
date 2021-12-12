import tensorflow as tf
from tensorflow.keras.backend import set_learning_phase
from tensorflow.keras.models import load_model
import numpy as np
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import efficientpose_utils

LEARNING_RATE = 0.01


def get_model(model_name):
    model_variant = model_name.lower()
    model_variant = model_variant[13:] if len(model_variant) > 7 else model_variant
    lite = True if model_variant.endswith('_lite') else False
    set_learning_phase(0)
    model = load_model(os.path.join('efficientpose_models', 'keras', 'EfficientPose{0}.h5'.format(model_variant.upper())),
                       custom_objects={'BilinearWeights': efficientpose_utils.keras_BilinearWeights,
                                       'Swish': efficientpose_utils.Swish(efficientpose_utils.eswish), 'eswish': efficientpose_utils.eswish,
                                       'swish1': efficientpose_utils.swish1})
    return model, {'rt': 224, 'i': 256, 'ii': 368, 'iii': 480, 'iv': 600, 'rt_lite': 224, 'i_lite': 256, 'ii_lite': 368}[model_variant], lite


def infer(batch, model, lite):
    if lite:
        batch_outputs = model.predict(batch)
    else:
        batch_outputs = model.predict(batch)[-1]
    return batch_outputs


def analyze(file_path, model, resolution, lite):
    from PIL import Image
    image = np.array(Image.open(file_path))
    image_height, image_width = image.shape[:2]
    batch = np.expand_dims(image, axis=0)

    # Preprocess batch
    batch = efficientpose_utils.preprocess(batch, resolution, lite)

    # Perform inference
    batch_outputs = infer(batch, model, lite)

    # Extract coordinates
    coordinates = [efficientpose_utils.extract_coordinates(batch_outputs[0, ...], image_height, image_width)]
    return coordinates


def coordinates_to_landmarks(coordinates):
    landmarks = np.zeros((16, 2))
    for i in range(len(coordinates[0])):
        landmarks[i] = coordinates[0][i][1:]
    return landmarks


def efficientpose_preprocess_data(data_directory="dataset"):
    model_name = 'II_Lite'
    framework_name = 'keras'
    model, resolution, lite = get_model(model_name)
    poses_directories = os.listdir(data_directory)
    landmarks_list = []
    label_list = []
    class_num = 0
    for pose_directory in poses_directories:
        images_path = os.path.join(data_directory, pose_directory)
        images = os.listdir(images_path)
        for img in images:
            coordinates = analyze(os.path.join(images_path, img), model, resolution, lite)
            landmarks = coordinates_to_landmarks(coordinates)
            landmarks_list.append(landmarks)
            label_list.append(class_num)
        class_num = class_num + 1

    landmarks_array = np.asarray(landmarks_list)
    label_array = np.asarray(label_list)
    categorical_labels = tf.keras.utils.to_categorical(label_array)
    return landmarks_array, categorical_labels, class_num


def define_model(num_classes):
    inputs = tf.keras.Input(shape=(16, 2))
    flatten = tf.keras.layers.Flatten()(inputs)
    layer = tf.keras.layers.Dense(128, activation='relu')(flatten)
    layer = tf.keras.layers.Dense(64, activation='relu')(layer)
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


def efficientpose():
    X, y, num_classes = efficientpose_preprocess_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    model = define_model(num_classes)
    history = train_model(model, X_train, X_val, y_train, y_val)
    plot_train_test(history, "EfficientPose")
    model.save("efficientpose_saved_model")


if __name__ == '__main__':
    efficientpose()
