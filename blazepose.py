import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from cv2 import cv2
from mediapipe.python.solutions import pose as mp_pose
from blazepose_utils import FullBodyPoseEmbedder
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

LEARNING_RATE = 0.001


def blazepose_preprocess_data(data_directory="dataset"):
    pose_embedder = FullBodyPoseEmbedder()
    poses_directories = os.listdir(data_directory)
    embedding_list = []
    label_list = []
    class_num = 0
    for pose_directory in tqdm(poses_directories):
        pose_images_path = os.path.join(data_directory, pose_directory)
        pose_images = os.listdir(pose_images_path)
        with mp_pose.Pose() as pose_tracker:
            for pose_image in pose_images:
                image = cv2.imread(os.path.join(pose_images_path, pose_image))
                image_height, image_width, _ = image.shape
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = pose_tracker.process(image=image)
                pose_landmarks = result.pose_landmarks
                if pose_landmarks is None:
                    print(os.path.join(pose_images_path, pose_image))
                    continue
                pose_landmarks = np.array([[lmk.x * image_width, lmk.y * image_height, lmk.z * image_width]
                                           for lmk in pose_landmarks.landmark], dtype=np.float32)
                embedding = pose_embedder(pose_landmarks)
                embedding_list.append(embedding)
                label_list.append(class_num)
        class_num = class_num + 1

    embedding_array = np.asarray(embedding_list)
    label_array = np.asarray(label_list)
    #categorical_labels = tf.keras.utils.to_categorical(label_array)
    #return embedding_array, categorical_labels, class_num
    embedding_array = embedding_array.reshape(embedding_array.shape[0], (embedding_array.shape[1]*embedding_array.shape[2]))
    return embedding_array, label_array, class_num


def define_model(num_classes):
    inputs = tf.keras.Input(shape=(78, 2))
    flatten = tf.keras.layers.Flatten()(inputs)
    #layer = tf.keras.layers.Dense(128, activation=tf.nn.relu6, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(flatten)
    #layer = tf.keras.layers.Dense(64, activation=tf.nn.relu6, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(layer)
    layer = tf.keras.layers.Dense(128, activation='relu')(flatten)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    layer = tf.keras.layers.Dense(64, activation='relu')(layer)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(layer)
    model = tf.keras.Model(inputs, outputs)
    model.summary()
    return model


def define_model_attn(num_classes):
    query_input = tf.keras.Input(shape=(78, 2))
    value_input = tf.keras.Input(shape=(78, 2))

    query_flatten = tf.keras.layers.Flatten()(query_input)
    value_flatten = tf.keras.layers.Flatten()(value_input)

    query_encoding = tf.keras.layers.Dense(128, activation='relu')(query_flatten)
    value_encoding = tf.keras.layers.Dense(128, activation='relu')(value_flatten)

    attention = tf.keras.layers.Attention()([query_encoding, value_encoding])
    input_layer = tf.keras.layers.Concatenate()([query_encoding, attention])


    layer = tf.keras.layers.Dense(256, activation='relu')(input_layer)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    layer = tf.keras.layers.Dense(128, activation='relu')(query_encoding)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    layer = tf.keras.layers.Dense(64, activation='relu')(layer)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(layer)
    model = tf.keras.Model([query_input, value_input], outputs)
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


def blazepose():
    X, y, num_classes = blazepose_preprocess_data(data_directory="dataset_enhanced/300")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    #X_val, y_val, _ = blazepose_preprocess_data(data_directory="test_dataset/test7-ultimate")
    model = define_model(num_classes)
    history = train_model(model, X_train, X_val, y_train, y_val)
    plot_train_test(history, "BlazePose")
    model.save("blazepose_saved_model")


def sk(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    print(model_name + " accuracy: ", model.score(X_test, y_test), end='\n\n')


def blaze_sklearn():
    X_train, y_train, num_classes = blazepose_preprocess_data(data_directory="dataset")
    X_test, y_test, _ = blazepose_preprocess_data(data_directory="test_dataset/test1-different_clothes")
    sk(KNeighborsClassifier(n_neighbors=10, weights='distance'), X_train, y_train, X_test, y_test, "knn")
    sk(DecisionTreeClassifier(), X_train, y_train, X_test, y_test, "decision tree")
    sk(RandomForestClassifier(), X_train, y_train, X_test, y_test, "random forest")
    sk(MLPClassifier(solver='lbfgs'), X_train, y_train, X_test, y_test, "mlp")


if __name__ == '__main__':
    #blazepose()
    blaze_sklearn()
