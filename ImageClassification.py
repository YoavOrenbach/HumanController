import matplotlib.pyplot as plt
import tensorflow as tf
import os

# useful links:
# https://www.tensorflow.org/tutorials/load_data/images - loading data
# https://www.tensorflow.org/tutorials/images/transfer_learning - image classification transfer learning
# https://www.tensorflow.org/lite/tutorials/model_maker_image_classification - maybe

data_dir = "dataset"
BATCH_SIZE = 32
IMG_SIZE = (160, 160)


def preprocess_data():
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.1,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.1,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE)
    class_names = train_dataset.class_names
    return train_dataset, validation_dataset, class_names


def visualize_data(train_dataset, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in train_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()


def load_image_classification_model(model_name):
    IMG_SHAPE = IMG_SIZE + (3,)
    if model_name == "mobilenet":
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    elif model_name == "vgg":
        preprocess_input = tf.keras.applications.vgg16.preprocess_input
        base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    elif model_name == "resnet":
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
        base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    else:
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    return preprocess_input, base_model


def define_model(train_dataset, validation_dataset, class_names, model_name):
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    preprocess_input, base_model = load_image_classification_model(model_name)
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model, train_dataset, validation_dataset


def train_model(model, train_dataset, validation_dataset):
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_dataset,
                        epochs=10,
                        validation_data=validation_dataset)
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


def image_classification_model(model_name):
    train_dataset, validation_dataset, class_names = preprocess_data()
    model, train_dataset, validation_dataset = define_model(train_dataset, validation_dataset, class_names, model_name)
    history = train_model(model, train_dataset, validation_dataset)
    plot_train_test(history, model_name)
    model.save(model_name + "_saved_model")


def evaluate_model(model_name):
    model = tf.keras.models.load_model(model_name + "_saved_model")
    results_dic = {}
    test_data = os.listdir("testDataset")
    for test_folder in test_data:
        test_folder_path = os.path.join("testDataset", test_folder)
        test_dataset = tf.keras.utils.image_dataset_from_directory(test_folder_path, image_size=IMG_SIZE)
        AUTOTUNE = tf.data.AUTOTUNE
        test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
        loss, accuracy = model.evaluate(test_dataset)
        results_dic[test_folder[:5]] = accuracy

    keys = list(results_dic.keys())
    values = list(results_dic.values())
    plt.bar(keys, values)
    plt.title("BlazePose accuracy on test sets")
    plt.xlabel("test sets")
    plt.ylabel("test accuracy")
    plt.show()


if __name__ == '__main__':
    image_classification_model("vgg")
    #evaluate_model("vgg")
