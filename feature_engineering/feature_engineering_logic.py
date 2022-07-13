from abc import ABC, abstractmethod


class FeatureEngineering(ABC):
    def __init__(self, landmark_names, name=""):
        self.landmark_names = landmark_names
        self.name = name
        self.input_size = None

    def get_input_size(self):
        return self.input_size

    def get_name(self):
        return self.name

    @abstractmethod
    def __call__(self, landmarks):
        pass
