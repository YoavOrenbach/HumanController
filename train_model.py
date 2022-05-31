import tensorflow as tf
from keras.layers.merge import concatenate
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pose_estimation_utils import load_movenet_model, feature_engineering
from pose_estimation_utils import movenet_inference_video, init_crop_region, determine_crop_region
from tqdm import tqdm

LEARNING_RATE = 0.001
IMG_SIZE = (256, 256)
ENSEMBLE_SIZE = 3


def preprocess_data(data_directory):
    model_name = "movenet_thunder"
    movenet, input_size = load_movenet_model(model_name)
    poses_directories = ['pose'+str(i+1) for i in range(len(os.listdir(data_directory)))]
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
            embedding = feature_engineering(landmarks)
            landmarks_list.append(embedding)
            label_list.append(class_num)
        class_num = class_num + 1

    landmarks_array = np.asarray(landmarks_list)
    label_array = np.asarray(label_list)
    categorical_labels = tf.keras.utils.to_categorical(label_array)
    return landmarks_array, categorical_labels, class_num


def define_model(num_classes, initializer="glorot_uniform"):
    inputs = tf.keras.Input(shape=(78, 2))
    flatten = tf.keras.layers.Flatten()(inputs)
    layer = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer)(flatten)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    layer = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer)(layer)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(layer)
    model = tf.keras.Model(inputs, outputs)
    return model


def train_model(model, X_train, X_val, y_train, y_val, optimizer="Adam", patience=40):
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)
    model.fit(X_train, y_train,
              epochs=50,
              batch_size=32,
              validation_data=(X_val, y_val),
              callbacks=[earlystopping])


def define_stacked_model(members, num_classes):
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            layer.trainable = False
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
    ensemble_visible = [model.input for model in members]
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)
    hidden = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(merge)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(hidden)
    model = tf.keras.Model(inputs=ensemble_visible, outputs=output)
    return model


def controller_model(data_directory, model_directory):
    X, y, num_classes = preprocess_data(data_directory)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    sub_models = []

    for i in range(ENSEMBLE_SIZE):
        model = define_model(num_classes, initializer=tf.keras.initializers.HeNormal())
        train_model(model, X_train, X_val, y_train, y_val, optimizer="Nadam")
        model.save(model_directory+"/model_"+str(i+1))
        sub_models.append(model)

    meta_learner = define_stacked_model(sub_models, num_classes)
    ensemble_X_train = [X_train for _ in range(ENSEMBLE_SIZE)]
    ensemble_X_val = [X_val for _ in range(ENSEMBLE_SIZE)]
    train_model(meta_learner, ensemble_X_train, ensemble_X_val, y_train, y_val, patience=10)
    meta_learner.save(model_directory+"/ensemble")
