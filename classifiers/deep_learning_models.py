import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from abc import abstractmethod
from classifiers.classifier_logic import Classifier


class DLClassifier(Classifier):
    def __init__(self, model_name):
        super(DLClassifier, self).__init__(model_name)

    @abstractmethod
    def define_model(self, input_size, output_size):
        pass

    def prepare_input(self, X, y):
        y = tf.keras.utils.to_categorical(y)
        return X, y

    def load(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def save(self, model_path):
        self.model.save(model_path)

    def predict(self, model_input):
        predict_frame = np.expand_dims(model_input, axis=0)
        return self.model(predict_frame, training=False)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def train_model(self, X_train, X_val, y_train, y_val, optimizer="Adam", patience=30):
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)
        history = self.model.fit(X_train, y_train,
                                 epochs=50,
                                 batch_size=32,
                                 validation_data=(X_val, y_val),
                                 callbacks=[earlystopping])
        self.plot_train_test(history)

    def plot_train_test(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.suptitle(self.model_name)
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


class MLP(DLClassifier):
    def __init__(self):
        super(MLP, self).__init__("mlp")

    def define_model(self, input_size, output_size, initializer="glorot_uniform"):
        inputs = tf.keras.Input(shape=input_size)
        flatten = tf.keras.layers.Flatten()(inputs)
        layer = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer)(flatten)
        layer = tf.keras.layers.Dropout(0.5)(layer)
        layer = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer)(layer)
        layer = tf.keras.layers.Dropout(0.5)(layer)
        outputs = tf.keras.layers.Dense(output_size, activation="softmax")(layer)
        self.model = tf.keras.Model(inputs, outputs)

    def define_model_shallow(self, input_size, output_size, initializer="glorot_uniform"):
        inputs = tf.keras.Input(shape=input_size)
        flatten = tf.keras.layers.Flatten()(inputs)
        layer = tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer)(flatten)
        layer = tf.keras.layers.Dropout(0.5)(layer)
        outputs = tf.keras.layers.Dense(output_size, activation="softmax")(layer)
        self.model = tf.keras.Model(inputs, outputs)

    def define_model_deep(self, input_size, output_size, initializer="glorot_uniform"):
        inputs = tf.keras.Input(shape=input_size)
        flatten = tf.keras.layers.Flatten()(inputs)
        layer = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializer)(flatten)
        layer = tf.keras.layers.Dropout(0.5)(layer)
        layer = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer)(layer)
        layer = tf.keras.layers.Dropout(0.5)(layer)
        layer = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer)(layer)
        layer = tf.keras.layers.Dropout(0.5)(layer)
        outputs = tf.keras.layers.Dense(output_size, activation="softmax")(layer)
        self.model = tf.keras.Model(inputs, outputs)


class CNN(DLClassifier):
    def __init__(self):
        super(CNN, self).__init__("1DConv")

    def define_model(self, input_size, output_size):
        inputs = tf.keras.Input(shape=input_size)
        x = tf.keras.layers.Conv1D(filters=8, kernel_size=8, activation='relu')(inputs)
        x = tf.keras.layers.Conv1D(filters=4, kernel_size=8, activation='relu')(x)
        x = tf.keras.layers.Conv1D(filters=2, kernel_size=8, activation='relu')(x)
        x = tf.keras.layers.Conv1D(filters=1, kernel_size=8, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(output_size, activation="softmax")(x)
        self.model = tf.keras.Model(inputs, outputs)


class ConvolutionLSTM(DLClassifier):
    def __init__(self):
        super(ConvolutionLSTM, self).__init__("1DConvLstm")

    def define_model(self, input_size, output_size):
        query_input = tf.keras.Input(shape=input_size)
        cnn_layer = tf.keras.layers.Conv1D(filters=4, kernel_size=4, padding='same')
        query_encoding = cnn_layer(query_input)
        value_encoding = cnn_layer(query_input)

        attention = tf.keras.layers.Attention()([query_encoding, value_encoding])

        query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_encoding)
        query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(attention)
        input_layer = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])

        layer = tf.keras.layers.Dense(128, activation='relu')(input_layer)
        layer = tf.keras.layers.Dropout(0.5)(layer)
        layer = tf.keras.layers.Dense(64, activation='relu')(layer)
        layer = tf.keras.layers.Dropout(0.5)(layer)
        outputs = tf.keras.layers.Dense(output_size, activation="softmax")(layer)
        self.model = tf.keras.Model(query_input, outputs)


class Attention(DLClassifier):
    def __init__(self):
        super(Attention, self).__init__("attention")

    def define_model(self, input_size, output_size):
        inputs = tf.keras.Input(shape=input_size)
        query_encoding = tf.keras.layers.Flatten()(inputs)
        query_encoding = tf.keras.layers.Dense(128, activation='relu')(query_encoding)
        attention = tf.keras.layers.Attention()([query_encoding, query_encoding])
        query_value_attention = tf.keras.layers.Flatten()(attention)
        input_layer = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])

        layer = tf.keras.layers.Dense(128, activation='relu')(input_layer)
        layer = tf.keras.layers.Dropout(0.5)(layer)
        layer = tf.keras.layers.Dense(64, activation='relu')(layer)
        layer = tf.keras.layers.Dropout(0.5)(layer)
        outputs = tf.keras.layers.Dense(output_size, activation="softmax")(layer)
        self.model = tf.keras.Model(inputs, outputs)


class VisionTransformer(DLClassifier):
    def __init__(self):
        super(VisionTransformer, self).__init__("visionTransformer")

    def define_model(self, input_size, output_size):
        inputs = tf.keras.Input(shape=input_size)
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
        outputs = tf.keras.layers.Dense(output_size, activation="softmax")(layer)
        self.model = tf.keras.Model(inputs, outputs)



