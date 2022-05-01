import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from cv2 import cv2
from mediapipe.python.solutions import pose as mp_pose
from blazepose_utils import FullBodyPoseEmbedder
#from keras.callbacks import CSVLogger
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from keras.layers.merge import concatenate


LEARNING_RATE = 0.001


def blazepose_preprocess_data(data_directory="dataset", ml_model=False):
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
    if not ml_model:
        categorical_labels = tf.keras.utils.to_categorical(label_array)
        return embedding_array, categorical_labels, class_num

    else:
        embedding_array = embedding_array.reshape(embedding_array.shape[0],
                                                  (embedding_array.shape[1]*embedding_array.shape[2]))
        return embedding_array, label_array, class_num


def define_model(num_classes, initializer="glorot_uniform"):
    inputs = tf.keras.Input(shape=(78, 2))
    flatten = tf.keras.layers.Flatten()(inputs)
    #layer = tf.keras.layers.Dense(128, activation=tf.nn.relu6, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(flatten)
    #layer = tf.keras.layers.Dense(64, activation=tf.nn.relu6, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(layer)
    layer = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer)(flatten)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    layer = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer)(layer)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(layer)
    model = tf.keras.Model(inputs, outputs)
    model.summary()
    return model


def train_model(model, X_train, X_val, y_train, y_val, patience=40):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)
    #csv_logger = CSVLogger('saved_models/training5.log', separator=',', append=False)
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


def blazepose(ml_model=False, preprocessing=True):
    if preprocessing:
        X, y, num_classes = blazepose_preprocess_data(data_directory="dataset_enhanced/300", ml_model=ml_model)
        np.save("preprocessing/blazepose_X_train.npy", X)
        np.save("preprocessing/blazepose_y_train.npy", y)
    else:
        X = np.load("preprocessing/blazepose_X_train.npy")
        y = np.load("preprocessing/blazepose_y_train.npy")
        num_classes = 25
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    #X_val, y_val, _ = blazepose_preprocess_data(data_directory="test_dataset/test1-different_clothes")
    model = define_model(num_classes)
    history = train_model(model, X_train, X_val, y_train, y_val)
    plot_train_test(history, "BlazePose")
    model.save("saved_models/blazepose")


def feature_engineering():
    log_data1 = pd.read_csv('saved_models/training1.log', sep=',', engine='python')
    log_data2 = pd.read_csv('saved_models/training2.log', sep=',', engine='python')
    log_data3 = pd.read_csv('saved_models/training3.log', sep=',', engine='python')
    log_data4 = pd.read_csv('saved_models/training4.log', sep=',', engine='python')
    log_data5 = pd.read_csv('saved_models/training5.log', sep=',', engine='python')

    plt.plot(log_data1['val_accuracy'], label="No feature engineering")
    plt.plot(log_data2['val_accuracy'], label="Normalizing key points")
    plt.plot(log_data3['val_accuracy'], label="Normalizing + Euclidean distances")
    plt.plot(log_data4['val_accuracy'], label="Normalizing + Angles")
    plt.plot(log_data5['val_accuracy'], label="Normalizing + Pairwise distances")
    plt.title("Accuracy of different feature engineering methods")
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


def logistic_grid():
    X_train, y_train, num_classes = blazepose_preprocess_data(data_directory="dataset_enhanced/300")
    param = {'penalty': ['l1', 'l2'],
             'C': np.logspace(-4, 4, 20),
             'solver':  ['liblinear']}
    gs_logistic = GridSearchCV(LogisticRegression(), param, verbose=1, scoring='accuracy', n_jobs=-1, cv=5)
    gs_logistic.fit(X_train, y_train)
    print(gs_logistic.best_params_)
    print(gs_logistic.best_score_)
    # best param: {'C': 11.288378916846883, 'penalty': 'l2', 'solver': 'liblinear'}


def knn_grid():
    X_train, y_train, num_classes = blazepose_preprocess_data(data_directory="dataset_enhanced/300")
    param = {'n_neighbors': [1,5,10,15,20,25,30],
             'leaf_size': [20,30,40],
             'weights': ['uniform', 'distance'],
             'metric': ['euclidean', 'manhattan', 'chebyshev']}
    gs_knn = GridSearchCV(KNeighborsClassifier(), param, verbose=1, scoring='accuracy', n_jobs=-1, cv=5)
    gs_knn.fit(X_train, y_train)
    print(gs_knn.best_params_)
    print(gs_knn.best_score_)
    # best params : {'leaf_size': 20, 'metric': 'manhattan', 'n_neighbors': 1, 'weights': 'uniform'}


def knn_tuning():
    X_train, y_train, num_classes = blazepose_preprocess_data(data_directory="dataset_enhanced/300")
    X_test, y_test, _ = blazepose_preprocess_data(data_directory="test_dataset/test3-diff_back_diff_lighting")
    test_accuracies = []

    knn_range = range(1, 21)
    for k in knn_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', leaf_size=20, metric='manhattan')
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test)
        test_accuracies.append(accuracy)
        print(k, " : ", accuracy)

    plt.plot(knn_range, test_accuracies)
    plt.xlabel('Number of neighbors')
    plt.ylabel('Test accuracy')
    plt.title('KNN accuracy using different number of neighbors')
    plt.show()

    leaf_range = [1, 5, 10, 15, 20, 25, 30, 35, 40] # assuming n_neighbors=2 is best
    for leaf in leaf_range:
        knn = KNeighborsClassifier(n_neighbors=2, weights='uniform', leaf_size=leaf, metric='manhattan')
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test)
        print(leaf, " : ", accuracy)

    # Number of neighbors changes between tests - sometimes one, two or three but minimal changes
    # Leaf size does not matter


def tree_grid():
    X_train, y_train, num_classes = blazepose_preprocess_data(data_directory="dataset_enhanced/300")
    param = {'criterion': ['gini', 'entropy'],
             'max_depth': [None, 1, 10, 20, 30, 40, 50, 60, 70],
             'min_samples_split': range(1, 10),
             'min_samples_leaf': range(1, 5)}

    gs_decision_tree = GridSearchCV(DecisionTreeClassifier(), param, verbose=1, scoring='accuracy', n_jobs=-1, cv=5)
    gs_decision_tree.fit(X_train, y_train)
    print(gs_decision_tree.best_params_)
    print(gs_decision_tree.best_score_)
    # best params : {'criterion': 'gini', 'max_depth': 70, 'min_samples_leaf': 1, 'min_samples_split': 4}


def xgboost_grid():
    X_train, y_train, num_classes = blazepose_preprocess_data(data_directory="dataset_enhanced/300")
    param = {'min_child_weight': [1, 5, 10],
             'gamma': [0, 0.5, 1, 1.5, 2],
             'subsample': [0.6, 0.8, 1.0],
             'colsample_bytree': [0.6, 0.8, 1.0],
             'max_depth': [3, 4, 5, 6],
             'learning_rate': [0.1, 0.01, 0.05, 0.3]}

    gs_xgboost = RandomizedSearchCV(XGBClassifier(objective='multi:softmax', num_class=num_classes), param, verbose=3, scoring='accuracy', n_jobs=-1, cv=5)
    gs_xgboost.fit(X_train, y_train)
    print(gs_xgboost.best_params_)
    print(gs_xgboost.best_score_)
    # best param: {'subsample': 0.8, 'min_child_weight': 5, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 1.5, 'colsample_bytree': 0.6}


def fit_machine_learning_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    print(model_name + " accuracy: ", model.score(X_test, y_test), end='\n\n')
    joblib.dump(model, f'saved_models/blazepose_{model_name}.joblib')


def machine_learning_models():
    X_train, y_train, num_classes = blazepose_preprocess_data(data_directory="dataset_enhanced/300")
    X_test, y_test, _ = blazepose_preprocess_data(data_directory="test_dataset/test3-diff_back_diff_lighting")

    # Logistic regression:
    fit_machine_learning_model(LogisticRegression(), X_train, y_train, X_test, y_test, "logistic")

    # Ridge classifier:
    fit_machine_learning_model(RidgeClassifier(), X_train, y_train, X_test, y_test, "ridge")

    # K-nearest neighbors:
    knn = KNeighborsClassifier(n_neighbors=1, weights='uniform', leaf_size=20, metric='manhattan')
    fit_machine_learning_model(knn, X_train, y_train, X_test, y_test, "knn")

    # Decision Tree:
    decision_tree = DecisionTreeClassifier(criterion='gini', max_depth=70, min_samples_split=4, min_samples_leaf=1)
    fit_machine_learning_model(decision_tree, X_train, y_train, X_test, y_test, "decision_tree")

    # Random Forest:
    random_forest = RandomForestClassifier(criterion='gini', max_depth=70, min_samples_split=4, min_samples_leaf=1)
    fit_machine_learning_model(random_forest, X_train, y_train, X_test, y_test, "random_forest")

    # Extreme boosting:
    fit_machine_learning_model(XGBClassifier(objective='multi:softmax', num_class=num_classes), X_train, y_train,
                               X_test, y_test, "xgboost")

    # MLP Classifier:
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", batch_size=32,
                        learning_rate="constant", learning_rate_init=LEARNING_RATE, early_stopping=True,
                        max_iter=50, verbose=True)
    fit_machine_learning_model(mlp, X_train, y_train, X_test, y_test, "mlp")


def cnn_model(num_classes):
    inputs = tf.keras.Input(shape=(78, 2))
    x = tf.keras.layers.Conv1D(filters=8, kernel_size=8, activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(filters=4, kernel_size=8, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=2, kernel_size=8, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=1, kernel_size=8, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def cnn_lstm_model(num_classes):
    inputs = tf.keras.Input(shape=(78, 2))
    cnn_layer = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
    x = cnn_layer(inputs)
    #x = tf.keras.layers.GlobalAveragePooling1D(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def cnn_attention(num_classes):
    query_input = tf.keras.Input(shape=(78, 2))
    #value_input = tf.keras.Input(shape=(78, 2))

    #query_flatten = tf.keras.layers.Flatten()(query_input)
    #value_flatten = tf.keras.layers.Flatten()(value_input)

    cnn_layer = tf.keras.layers.Conv1D(filters=4, kernel_size=4, padding='same')
    query_encoding = cnn_layer(query_input)
    #value_encoding = cnn_layer(value_input)
    value_encoding = cnn_layer(query_input)

    attention = tf.keras.layers.Attention()([query_encoding, value_encoding])

    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_encoding)
    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(attention)
    input_layer = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])

    layer = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    layer = tf.keras.layers.Dense(64, activation='relu')(layer)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(layer)
    #model = tf.keras.Model([query_input, value_input], outputs)
    model = tf.keras.Model(query_input, outputs)
    model.summary()
    return model


def attention_model(num_classes):
    inputs = tf.keras.Input(shape=(78, 2))

    #attention = tf.keras.layers.Attention()([inputs, inputs])
    #attention = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2)(inputs, inputs)

    query_encoding = tf.keras.layers.Flatten()(inputs)
    query_encoding = tf.keras.layers.Dense(128, activation='relu')(query_encoding)
    attention = tf.keras.layers.Attention()([query_encoding, query_encoding])
    query_value_attention = tf.keras.layers.Flatten()(attention)
    input_layer = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])

    layer = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    layer = tf.keras.layers.Dense(64, activation='relu')(layer)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(layer)
    model = tf.keras.Model(inputs, outputs)
    model.summary()
    return model


def vision_transformer(num_classes):
    inputs = tf.keras.Input(shape=(78, 2))
    z = inputs
    for _ in range(5):
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(z)
        x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2)(x, x)
        x = tf.keras.layers.Add()([x, z])

        y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        y = tf.keras.layers.Dense(64, activation=tf.nn.gelu)(y)
        y = tf.keras.layers.Dropout(0.1)(y)
        y = tf.keras.layers.Dense(2, activation=tf.nn.gelu)(y)
        y = tf.keras.layers.Dropout(0.1)(y)

        z = tf.keras.layers.Add()([y, x])

    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(z)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(0.5)(representation)
    layer = tf.keras.layers.Dense(128, activation=tf.nn.gelu)(representation)
    layer = tf.keras.layers.Dropout(0.1)(layer)
    layer = tf.keras.layers.Dense(64, activation=tf.nn.gelu)(layer)
    layer = tf.keras.layers.Dropout(0.1)(layer)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(layer)
    model = tf.keras.Model(inputs, outputs)
    model.summary()
    return model


def knn():
    X_train, y_train, num_classes = blazepose_preprocess_data(data_directory="dataset", ml_model=True)
    model = KNeighborsClassifier(n_neighbors=1, weights='uniform', leaf_size=20, metric='manhattan')
    model.fit(X_train, y_train)
    joblib.dump(model, 'saved_models/blazepose_knn_small.joblib')


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
    return model


def ensemble():
    X = np.load("preprocessing/blazepose_X_train.npy")
    y = np.load("preprocessing/blazepose_y_train.npy")
    num_classes = 25
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    num_sub_models = 3
    sub_models = []

    for i in range(num_sub_models):
        model = define_model(num_classes, initializer=tf.keras.initializers.HeNormal())
        train_model(model, X_train, X_val, y_train, y_val)
        model.save("saved_models/blazepose_ensemble/model_"+str(i+1))
        sub_models.append(model)

    meta_learner = define_stacked_model(sub_models, num_classes)
    ensemble_X_train = [X_train for _ in range(num_sub_models)]
    ensemble_X_val = [X_val for _ in range(num_sub_models)]
    history = train_model(meta_learner, ensemble_X_train, ensemble_X_val, y_train, y_val, patience=10)
    plot_train_test(history, "Blazepose Ensemble")
    meta_learner.save("saved_models/blazepose_ensemble/ensemble")


def plot_networks():
    X_train = np.load("preprocessing/blazepose_X_train.npy")
    y_train = np.load("preprocessing/blazepose_y_train.npy")
    num_classes = 25
    X_val = np.load("preprocessing/blazepose_X_test4.npy")
    y_val = np.load("preprocessing/blazepose_y_test4.npy")

    mlp = define_model(num_classes)
    cnn = cnn_model(num_classes)
    cnn_lstm = cnn_lstm_model(num_classes)
    cnn_attn = cnn_attention(num_classes)
    attention = attention_model(num_classes)
    transformer = vision_transformer(num_classes)

    mlp_history = train_model(mlp, X_train, X_val, y_train, y_val)
    cnn_history = train_model(cnn, X_train, X_val, y_train, y_val)
    cnn_lstm_history = train_model(cnn_lstm, X_train, X_val, y_train, y_val)
    cnn_attn_history = train_model(cnn_attn, X_train, X_val, y_train, y_val)
    attention_history = train_model(attention, X_train, X_val, y_train, y_val)
    transformer_history = train_model(transformer, X_train, X_val, y_train, y_val)

    plt.plot(mlp_history.history['val_accuracy'], label="MLP")
    plt.plot(cnn_history.history['val_accuracy'], label="CNN - 1D convolution")
    plt.plot(cnn_lstm_history.history['val_accuracy'], label="CNN + LSTM cell")
    plt.plot(cnn_attn_history.history['val_accuracy'], label="CNN + Attention layer")
    plt.plot(attention_history.history['val_accuracy'], label="Attention + Linear layers")
    plt.plot(transformer_history.history['val_accuracy'], label="Vision Transformer")
    plt.title('Different network architectures validation accuracies')
    plt.ylabel('Validation accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    plot_networks()
