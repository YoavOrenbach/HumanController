from abc import ABC, abstractmethod


class PoseEstimation(ABC):
    """An abstract class representing a pose estimation model logic."""
    def __init__(self, name=""):
        """Initializing a pose estimation model with image height and width, a name, and a sub-model."""
        self.image_height = 256
        self.image_width = 256
        self.name = name
        self.sub_model = None

    def get_name(self):
        """Returns the name of the pose estimation model."""
        return self.name

    def set_sub_model(self, sub_model):
        """Sets the sub model of the pose estimation logic
        (switches from MoveNet Thunder to Lightning and vice-versa)."""
        self.sub_model = sub_model

    @abstractmethod
    def load_model(self):
        """Loads the pose estimation model."""
        pass

    @abstractmethod
    def start(self):
        """Starts the pose estimation model's run, e.g., resetting the model after every process run."""
        pass

    @abstractmethod
    def process_frame(self, frame):
        """Process a frame to produce keypoints."""
        pass

    @abstractmethod
    def end(self):
        """Ends the pose estimation model's run"""
        pass

    @abstractmethod
    def get_landmark_names(self):
        """Returns the landmark names produces by the pose estimation model."""
        pass
