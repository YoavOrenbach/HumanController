import numpy as np
from itertools import combinations
from feature_engineering.normalization import Normalization


class EuclideanDistance(Normalization):
    def __init__(self, landmark_names):
        super(EuclideanDistance, self).__init__(landmark_names, "euclideanDistance")

    def euclidean_distance(self, landmarks, landmark1, landmark2):
        lmk_from = landmarks[self.landmark_names.index(landmark1)]
        lmk_to = landmarks[self.landmark_names.index(landmark2)]
        dis = np.linalg.norm(lmk_from[:2] - lmk_to[:2])
        return dis

    def euclidean_embedding(self, landmarks):
        relevant_landmarks = ['nose',
                              'left_shoulder', 'right_shoulder',
                              'left_elbow', 'right_elbow',
                              'left_wrist', 'right_wrist',
                              'left_hip', 'right_hip',
                              'left_knee', 'right_knee',
                              'left_ankle', 'right_ankle']
        embedding_list = []
        combinations_list = list(combinations(relevant_landmarks, 2))
        for pair in combinations_list:
            embedding_list.append(self.euclidean_distance(landmarks, pair[0], pair[1]))
        embedding = np.asarray(embedding_list)
        return embedding

    def __call__(self, landmarks):
        landmarks = self.normalize_landmarks(landmarks)
        embedding = self.euclidean_embedding(landmarks)
        self.input_size = embedding.shape
        return embedding
