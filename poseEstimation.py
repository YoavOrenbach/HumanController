import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt
from movenetUtils import load_movenet_model, movenet_inference, landmarks_to_embedding

# useful links:
# https://tfhub.dev/s?q=movenet - tf hub for moveNet
# https://www.tensorflow.org/hub/tutorials/movenet - moveNet tutorial
# https://www.tensorflow.org/lite/tutorials/pose_classification - pose classification with moveNet


model_name = "movenet_thunder"
movenet, input_size = load_movenet_model(model_name)


def preprocess_data():
    files = os.listdir("dataset")
    img_list = []
    label_list = []
    label_val = 0
    for file in files:
        images_path = os.path.join("dataset", file)
        images = os.listdir(images_path)
        for img in images:
            image = tf.io.read_file(os.path.join(images_path, img))
            image = tf.image.decode_jpeg(image)
            landmarks = movenet_inference(image, movenet, input_size)
            img_list.append(landmarks)
            label_list.append(label_val)
        label_val = label_val + 1

    img_array = np.asarray(img_list)
    label_array = np.asarray(label_list)
    categorical_labels = tf.keras.utils.to_categorical(label_array)
    return img_array, categorical_labels


X, y = preprocess_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)


# Define the model
inputs = tf.keras.Input(shape=(1, 1, 17, 3))
flatten = tf.keras.layers.Flatten()(inputs)
embedding = landmarks_to_embedding(flatten)

layer = tf.keras.layers.Dense(128, activation='relu')(embedding)
#layer = tf.keras.layers.Dropout(0.5)(layer)
layer = tf.keras.layers.Dense(64, activation='relu')(layer)
#layer = tf.keras.layers.Dropout(0.5)(layer)
outputs = tf.keras.layers.Dense(4, activation="softmax")(layer)

model = tf.keras.Model(inputs, outputs)
model.summary()

base_learning_rate = 0.01
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Start training
history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=16,
                    validation_data=(X_val, y_val))

# Visualize the training history to see whether you're overfitting.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.suptitle('MoveNet')
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

model.save("movenet_saved_model")