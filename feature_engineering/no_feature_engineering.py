import numpy as np
from feature_engineering.feature_engineering_logic import FeatureEngineering


class NoFeatureEngineering(FeatureEngineering):
    def __init__(self, landmark_names):
        super(NoFeatureEngineering, self).__init__(landmark_names, "noFeatureEngineering")

    def __call__(self, landmarks):
        landmarks = np.copy(landmarks[:, :2])
        self.input_size = landmarks.shape
        return landmarks
