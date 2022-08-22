import numpy as np
from itertools import combinations
import math
from feature_engineering.normalization import Normalization


class Angle(Normalization):
    """An angle feature engineering class extending the Normalization class."""
    def __init__(self, landmark_names):
        super(Angle, self).__init__(landmark_names, "angles")

    def angle(self, landmarks, landmark1, landmark2, landmark3):
        """Calculates the angle between 3 landmarks."""
        p1 = landmarks[self.landmark_names.index(landmark1)]
        p2 = landmarks[self.landmark_names.index(landmark2)]
        p3 = landmarks[self.landmark_names.index(landmark3)]
        angle = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
        if angle < 0:
            angle = angle + 360
        return angle

    def angle_embedding(self, landmarks):
        """Calculates the angles between every triplet of points from relevant landmarks only."""
        relevant_landmarks = ['nose',
                              'left_shoulder', 'right_shoulder',
                              'left_elbow', 'right_elbow',
                              'left_wrist', 'right_wrist',
                              'left_hip', 'right_hip',
                              'left_knee', 'right_knee',
                              'left_ankle', 'right_ankle']
        embedding_list = []
        combinations_list = list(combinations(relevant_landmarks, 3))
        for triplet in combinations_list:
            embedding_list.append(self.angle(landmarks, *triplet))
        embedding = np.asarray(embedding_list)
        return embedding

    def __call__(self, landmarks):
        """The call method to process the landmark keypoints."""
        landmarks = self.normalize_landmarks(landmarks)
        embedding = self.angle_embedding(landmarks)
        self.input_size = embedding.shape
        return embedding
