import tensorflow as tf
import numpy as np
from classifiers.deep_learning_models import DLClassifier, MLP

ENSEMBLE_SIZE = 3


class StackedEnsemble(DLClassifier):
    def __init__(self):
        super(StackedEnsemble, self).__init__("ensemble")
        self.input_size = None
        self.output_size = None

    def predict(self, model_input):
        predict_frame = np.expand_dims(model_input, axis=0)
        predict_frame = [predict_frame for _ in range(ENSEMBLE_SIZE)]
        return self.model(predict_frame, training=False)

    def define_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def define_stacked_model(self, members):
        for i in range(len(members)):
            model = members[i]
            for layer in model.layers:
                layer.trainable = False
                layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
        ensemble_visible = [model.input for model in members]
        ensemble_outputs = [model.output for model in members]
        merge = tf.keras.layers.concatenate(ensemble_outputs)
        hidden = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(merge)
        output = tf.keras.layers.Dense(self.output_size, activation='softmax')(hidden)
        self.model = tf.keras.Model(inputs=ensemble_visible, outputs=output)

    def train_model(self, X_train, X_val, y_train, y_val, optimizer="Adam", patience=30):
        sub_models = []
        for _ in range(ENSEMBLE_SIZE):
            mlp = MLP()
            mlp.define_model(self.input_size, self.output_size, initializer=tf.keras.initializers.HeNormal())
            mlp.train_model(X_train, X_val, y_train, y_val, optimizer="Nadam")
            sub_models.append(mlp.model)
        self.define_stacked_model(sub_models)
        ensemble_X_train = [X_train for _ in range(ENSEMBLE_SIZE)]
        ensemble_X_val = [X_val for _ in range(ENSEMBLE_SIZE)]
        super().train_model(ensemble_X_train, ensemble_X_val, y_train, y_val, patience=10)
        

class AvgEnsemble(DLClassifier):
    def __int__(self):
        super(AvgEnsemble, self).__init__("EnsembleAvg")
        self.input_size = None
        self.output_size = None

    def define_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def define_ensemble(self, sub_models):
        model_input = tf.keras.Input(shape=self.input_size)
        model_outputs = [model(model_input, training=False) for model in sub_models]
        ensemble_output = tf.keras.layers.Average()(model_outputs)
        self.model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)

    def train_model(self, X, y, optimizer="Adam", patience=30):
        sub_models = []
        for i in range(ENSEMBLE_SIZE):
            mlp = MLP()
            mlp.define_model(self.input_size, self.output_size, initializer=tf.keras.initializers.HeNormal())
            mlp.train_model(X, y, optimizer="Nadam")
            mlp.model._name = 'model' + str(i)
            sub_models.append(mlp.model)
        self.define_ensemble(sub_models)
        super().train_model(X, y, patience=10)
