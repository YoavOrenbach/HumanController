import tensorflow as tf
from tensorflow.keras.backend import set_learning_phase
from tensorflow.keras.models import load_model
import numpy as np
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import efficientPoseUtils


def get_model(model_name):
    model_variant = model_name.lower()
    model_variant = model_variant[13:] if len(model_variant) > 7 else model_variant
    lite = True if model_variant.endswith('_lite') else False
    set_learning_phase(0)
    model = load_model(os.path.join('efficientPoseModels', 'keras', 'EfficientPose{0}.h5'.format(model_variant.upper())),
                       custom_objects={'BilinearWeights': efficientPoseUtils.keras_BilinearWeights,
                                       'Swish': efficientPoseUtils.Swish(efficientPoseUtils.eswish), 'eswish': efficientPoseUtils.eswish,
                                       'swish1': efficientPoseUtils.swish1})
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
    batch = efficientPoseUtils.preprocess(batch, resolution, lite)

    # Perform inference
    batch_outputs = infer(batch, model, lite)

    # Extract coordinates
    coordinates = [efficientPoseUtils.extract_coordinates(batch_outputs[0,...], image_height, image_width)]
    return coordinates


model_name = 'II_Lite'
framework_name = 'keras'
model, resolution, lite = get_model(model_name)


def coordinates_to_landmarks(coordinates):
    landmarks = np.zeros((16, 2))
    for i in range(len(coordinates[0])):
        landmarks[i] = coordinates[0][i][1:]
    return landmarks


def preprocess_data():
    files = os.listdir("dataset")
    img_list = []
    label_list = []
    label_val = 0
    for file in files:
        images_path = os.path.join("dataset", file)
        images = os.listdir(images_path)
        for img in images:
            coordinates = analyze(os.path.join(images_path, img), model, resolution, lite)
            landmarks = coordinates_to_landmarks(coordinates)
            img_list.append(landmarks)
            label_list.append(label_val)
        label_val = label_val + 1

    img_array = np.asarray(img_list)
    label_array = np.asarray(label_list)
    categorical_labels = tf.keras.utils.to_categorical(label_array)
    return img_array, categorical_labels, label_val


X, y, class_num = preprocess_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)


# Define the model
inputs = tf.keras.Input(shape=(16, 2))
flatten = tf.keras.layers.Flatten()(inputs)
layer = tf.keras.layers.Dense(128, activation='relu')(flatten)
#layer = tf.keras.layers.Dropout(0.5)(layer)
layer = tf.keras.layers.Dense(64, activation='relu')(layer)
#layer = tf.keras.layers.Dropout(0.5)(layer)
outputs = tf.keras.layers.Dense(class_num, activation="softmax")(layer)

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
plt.suptitle('EfficientPose')
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

model.save("efficientPose_saved_model")
