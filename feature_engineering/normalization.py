import numpy as np
from feature_engineering.feature_engineering_logic import FeatureEngineering


class Normalization(FeatureEngineering):
    """A normalization class extending the FeatureEngineering class."""
    def __init__(self, landmark_names, name="normalization"):
        super(Normalization, self).__init__(landmark_names, name)

    def center_point(self, landmarks, left_bodypart, right_bodypart):
        """Calculates pose center between two body parts."""
        left = landmarks[self.landmark_names.index(left_bodypart)]
        right = landmarks[self.landmark_names.index(right_bodypart)]
        center = (left + right) * 0.5
        return center

    def get_pose_size(self, landmarks, torso_size_multiplier=2.5):
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        # Hips center
        hips_center = self.center_point(landmarks, 'left_hip', 'right_hip')

        # Shoulders center
        shoulders_center = self.center_point(landmarks, 'left_shoulder', 'right_shoulder')

        # Torso size as the minimum body size
        torso_size = np.linalg.norm(shoulders_center - hips_center)

        # Pose center
        pose_center_new = self.center_point(landmarks, 'left_hip', 'right_hip')
        max_dist = np.max(np.linalg.norm(landmarks - pose_center_new, axis=0))

        # Normalize scale
        pose_size = np.maximum(torso_size * torso_size_multiplier, max_dist)

        return pose_size

    def normalize_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale."""
        landmarks = np.copy(landmarks[:, :2])

        # Move landmarks so that the pose center becomes (0,0)
        pose_center = self.center_point(landmarks, 'left_hip', 'right_hip')
        landmarks -= pose_center

        # Scale the landmarks to a constant pose size
        pose_size = self.get_pose_size(landmarks)
        landmarks /= pose_size

        return landmarks

    def __call__(self, landmarks):
        """The call method to process the landmark keypoints."""
        landmarks = self.normalize_landmarks(landmarks)
        self.input_size = landmarks.shape
        return landmarks
