from abc import ABC, abstractmethod


class FeatureEngineering(ABC):
    """An abstract class representing a feature engineering method."""
    def __init__(self, landmark_names, name=""):
        """Initializing a feature engineering method with a  name and landmark names (from the pose estimation)."""
        self.landmark_names = landmark_names
        self.name = name
        self.input_size = None

    def get_input_size(self):
        """Returns the input size for the classifier."""
        return self.input_size

    def get_name(self):
        """Returns the name of the feature engineering method."""
        return self.name

    @abstractmethod
    def __call__(self, landmarks):
        """The call method to process the landmark keypoints."""
        pass
