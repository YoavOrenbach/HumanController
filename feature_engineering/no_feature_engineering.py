import numpy as np
from feature_engineering.feature_engineering_logic import FeatureEngineering


class NoFeatureEngineering(FeatureEngineering):
    """A no feature engineering class extending the FeatureEngineering class with no processing method."""
    def __init__(self, landmark_names):
        super(NoFeatureEngineering, self).__init__(landmark_names, "noFeatureEngineering")

    def __call__(self, landmarks):
        """The call method to process the landmark keypoints - it doesn't do anything but change the size
        of the landmarks."""
        landmarks = np.copy(landmarks[:, :2])
        self.input_size = landmarks.shape
        return landmarks
