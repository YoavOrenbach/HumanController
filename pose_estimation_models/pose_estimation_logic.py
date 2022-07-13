from abc import ABC, abstractmethod


class PoseEstimationLogic(ABC):
    def __init__(self, name=""):
        self.image_height = 256
        self.image_width = 256
        self.name = name
        self.sub_model = None

    def get_name(self):
        return self.name

    def set_sub_model(self, sub_model):
        self.sub_model = sub_model

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def process_frame(self, frame):
        pass

    @abstractmethod
    def end(self):
        pass

    @abstractmethod
    def get_landmark_names(self):
        pass
