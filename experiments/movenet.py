import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from movenet_utils import load_movenet_model, movenet_inference, feature_engineering, landmarks_to_embedding
from movenet_utils import movenet_inference_video, init_crop_region, determine_crop_region
from tqdm import tqdm
from keras.layers.merge import concatenate
#from tensorflow.keras.utils import plot_model
#os.environ["PATH"] += os.pathsep + 'E:/school/Graphviz/bin/'

from sklearn.neighbors import KNeighborsClassifier
import joblib

# useful links:
# https://tfhub.dev/s?q=movenet - tf hub for moveNet
# https://www.tensorflow.org/hub/tutorials/movenet - moveNet tutorial
# https://www.tensorflow.org/lite/tutorials/pose_classification - pose classification with moveNet

LEARNING_RATE = 0.001
IMG_SIZE = (256, 256)


def movenet_preprocess_data(data_directory="dataset", static=False, ml_model=False):
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
    if not ml_model:
        categorical_labels = tf.keras.utils.to_categorical(label_array)
        return landmarks_array, categorical_labels, class_num

    else:
        landmarks_array = landmarks_array.reshape(landmarks_array.shape[0],
                                                  landmarks_array.shape[1]*landmarks_array.shape[2])
        return landmarks_array, label_array, class_num


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


def define_model2(num_classes, initializer="glorot_uniform"):
    inputs = tf.keras.Input(shape=(78, 2))
    flatten = tf.keras.layers.Flatten()(inputs)
    layer = tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer)(flatten)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(layer)
    model = tf.keras.Model(inputs, outputs)
    return model


def define_model3(num_classes, initializer="glorot_uniform"):
    inputs = tf.keras.Input(shape=(78, 2))
    flatten = tf.keras.layers.Flatten()(inputs)
    layer = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializer)(flatten)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    layer = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer)(layer)
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


def movenet(ml_model=False, preprocessing=True):
    if preprocessing:
        X, y, num_classes = movenet_preprocess_data(data_directory="dataset_enhanced/300", static=False, ml_model=ml_model)
        np.save("preprocessing/movenet_X_train.npy", X)
        np.save("preprocessing/movenet_y_train.npy", y)
    else:
        X = np.load("preprocessing/movenet_X_train.npy")
        y = np.load("preprocessing/movenet_y_train.npy")
        num_classes = 25
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    #X_val, y_val, _ = movenet_preprocess_data(data_directory="test_dataset/test7-ultimate")
    model = define_model(num_classes, initializer=tf.keras.initializers.HeNormal())
    history = train_model(model, X_train, X_val, y_train, y_val, optimizer="Nadam")
    plot_train_test(history, "MoveNet")
    model.save("saved_models/movenet")


def knn():
    X_train, y_train, num_classes = movenet_preprocess_data(data_directory="dataset_enhanced/300", static=False, ml_model=True)
    model = KNeighborsClassifier(n_neighbors=1, weights='uniform', leaf_size=20, metric='manhattan')
    model.fit(X_train, y_train)
    joblib.dump(model, 'saved_models/movenet_knn.joblib')


def define_stacked_model(members, num_classes):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)
    hidden = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(merge)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(hidden)
    model = tf.keras.Model(inputs=ensemble_visible, outputs=output)
    #plot_model(model, show_shapes=True, to_file='model_graph.png')
    return model


def ensemble():
    #X, y, num_classes = movenet_preprocess_data(data_directory="dataset_enhanced/300", static=False)
    X = np.load("preprocessing/movenet_X_train.npy")
    y = np.load("preprocessing/movenet_y_train.npy")
    num_classes = 25
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    num_sub_models = 5
    sub_models = []

    for i in range(num_sub_models):
        model = define_model(num_classes, initializer=tf.keras.initializers.HeNormal())
        train_model(model, X_train, X_val, y_train, y_val, optimizer="Nadam")
        model.save("saved_models/movenet_ensemble/model_"+str(i+1))
        sub_models.append(model)

    meta_learner = define_stacked_model(sub_models, num_classes)
    ensemble_X_train = [X_train for _ in range(num_sub_models)]
    ensemble_X_val = [X_val for _ in range(num_sub_models)]
    history = train_model(meta_learner, ensemble_X_train, ensemble_X_val, y_train, y_val, patience=10)
    plot_train_test(history, "MoveNet Ensemble")
    meta_learner.save("saved_models/movenet_ensemble/ensemble")


def define_ensemble_model():
    sub_models = []
    for i in range(4):
        model = tf.keras.models.load_model("saved_models/movenet_ensemble/model_"+str(i+1))
        model._name = 'model' + str(i)
        sub_models.append(model)
    model_input = tf.keras.Input(shape=(78, 2))
    model_outputs = [model(model_input, training=False) for model in sub_models]
    ensemble_output = tf.keras.layers.Average()(model_outputs)
    ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)
    return ensemble_model


def ensemble_part2():
    X = np.load("preprocessing/movenet_X_train.npy")
    y = np.load("preprocessing/movenet_y_train.npy")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    model = define_ensemble_model()
    history = train_model(model, X_train, X_val, y_train, y_val, patience=10)
    plot_train_test(history, "MoveNet Ensemble avg")
    model.save("saved_models/movenet_ensemble/ensemble_avg")


def model_initializers():
    X, y, num_classes = movenet_preprocess_data(data_directory="dataset_enhanced/300", static=False)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    model_he_uniform = define_model(num_classes, tf.keras.initializers.HeUniform())
    model_he_normal = define_model(num_classes, tf.keras.initializers.HeNormal())
    model_constant = define_model(num_classes, tf.keras.initializers.Constant())
    model_glorot_normal = define_model(num_classes, tf.keras.initializers.GlorotNormal())
    model_glorot_uniform = define_model(num_classes, tf.keras.initializers.GlorotUniform())
    model_random_normal = define_model(num_classes, tf.keras.initializers.RandomNormal())
    model_random_uniform = define_model(num_classes, tf.keras.initializers.RandomUniform())
    model_zeros = define_model(num_classes, tf.keras.initializers.Zeros())

    train_model(model_he_uniform, X_train, X_val, y_train, y_val)
    train_model(model_he_normal, X_train, X_val, y_train, y_val)
    train_model(model_constant, X_train, X_val, y_train, y_val)
    train_model(model_glorot_normal, X_train, X_val, y_train, y_val)
    train_model(model_glorot_uniform, X_train, X_val, y_train, y_val)
    train_model(model_random_normal, X_train, X_val, y_train, y_val)
    train_model(model_random_uniform, X_train, X_val, y_train, y_val)
    train_model(model_zeros, X_train, X_val, y_train, y_val)

    model_he_uniform.save("saved_models/movenet_initializers/he_uniform")
    model_he_normal.save("saved_models/movenet_initializers/he_normal")
    model_constant.save("saved_models/movenet_initializers/constant")
    model_glorot_normal.save("saved_models/movenet_initializers/glorot_normal")
    model_glorot_uniform.save("saved_models/movenet_initializers/glorot_uniform")
    model_random_normal.save("saved_models/movenet_initializers/random_normal")
    model_random_uniform.save("saved_models/movenet_initializers/random_uniform")
    model_zeros.save("saved_models/movenet_initializers/zeros")


def ensemble_part3():
    X, y, num_classes = movenet_preprocess_data(data_directory="dataset_enhanced/300", static=False)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    num_sub_models = 6

    model_he_uniform = define_model(num_classes, tf.keras.initializers.HeUniform())
    model_he_normal = define_model(num_classes, tf.keras.initializers.HeNormal())
    model_glorot_normal = define_model(num_classes, tf.keras.initializers.GlorotNormal())
    model_glorot_uniform = define_model(num_classes, tf.keras.initializers.GlorotUniform())
    model_random_normal = define_model(num_classes, tf.keras.initializers.RandomNormal())
    model_random_uniform = define_model(num_classes, tf.keras.initializers.RandomUniform())

    train_model(model_he_uniform, X_train, X_val, y_train, y_val)
    train_model(model_he_normal, X_train, X_val, y_train, y_val)
    train_model(model_glorot_normal, X_train, X_val, y_train, y_val)
    train_model(model_glorot_uniform, X_train, X_val, y_train, y_val)
    train_model(model_random_normal, X_train, X_val, y_train, y_val)
    train_model(model_random_uniform, X_train, X_val, y_train, y_val)

    model_he_uniform.save("saved_models/movenet_ensemble2/he_uniform")
    model_he_normal.save("saved_models/movenet_ensemble2/he_normal")
    model_glorot_normal.save("saved_models/movenet_ensemble2/glorot_normal")
    model_glorot_uniform.save("saved_models/movenet_ensemble2/glorot_uniform")
    model_random_normal.save("saved_models/movenet_ensemble2/random_normal")
    model_random_uniform.save("saved_models/movenet_ensemble2/random_uniform")

    sub_models = [model_he_uniform, model_he_normal, model_glorot_normal, model_glorot_uniform,
                  model_random_normal, model_random_uniform]

    meta_learner = define_stacked_model(sub_models, num_classes)
    ensemble_X_train = [X_train for _ in range(num_sub_models)]
    ensemble_X_val = [X_val for _ in range(num_sub_models)]
    history = train_model(meta_learner, ensemble_X_train, ensemble_X_val, y_train, y_val)
    plot_train_test(history, "MoveNet")
    meta_learner.save("saved_models/movenet_ensemble2/ensemble")


if __name__ == '__main__':
    ensemble()
