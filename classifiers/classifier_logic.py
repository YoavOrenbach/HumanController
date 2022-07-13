from abc import ABC, abstractmethod


class Classifier(ABC):
    def __init__(self, model_name=""):
        self.model = None
        self.model_name = model_name

    def get_name(self):
        return self.model_name

    @abstractmethod
    def prepare_input(self, X, y):
        pass

    @abstractmethod
    def define_model(self, input_size, output_size):
        pass

    @abstractmethod
    def load(self, model_path):
        pass

    @abstractmethod
    def save(self, model_path):
        pass

    @abstractmethod
    def predict(self, model_input):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        pass

    @abstractmethod
    def train_model(self, X_train, X_val, y_train, y_val):
        pass
