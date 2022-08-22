from abc import ABC, abstractmethod


class Classifier(ABC):
    """An abstract class representing a classifier"""
    def __init__(self, model_name=""):
        """Initializing a classifier with a name"""
        self.model = None
        self.model_name = model_name

    def get_name(self):
        """Returns the name of the model"""
        return self.model_name

    @abstractmethod
    def prepare_input(self, X, y):
        """Prepares the input before training (for differences between sklearn and tensorflow)."""
        pass

    @abstractmethod
    def define_model(self, input_size, output_size):
        """Defines the model used - ML classifier or network architecture."""
        pass

    @abstractmethod
    def load(self, model_path):
        """Loads the model from the given path."""
        pass

    @abstractmethod
    def save(self, model_path):
        """Saves the model in the given path."""
        pass

    @abstractmethod
    def predict(self, model_input):
        """Returns a softmax vector of prediction probabilities given the input."""
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """Evaluates and returns the accuracy of the predicted labels from X against the ground truth y."""
        pass

    @abstractmethod
    def train_model(self, X_train, X_val, y_train, y_val):
        """Trains the model by the given train and validation sets."""
        pass
