import numpy as np
from itertools import combinations
from feature_engineering.normalization import Normalization


class PairwiseDistance(Normalization):
    def __init__(self, landmark_names):
        super(PairwiseDistance, self).__init__(landmark_names, "pairwiseDistance")

    def get_distances(self, landmarks, keypoint1, keypoint2):
        landmark1 = landmarks[self.landmark_names.index(keypoint1)]
        landmark2 = landmarks[self.landmark_names.index(keypoint2)]
        return landmark2 - landmark1

    def distance_embedding(self, landmarks):
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
            embedding_list.append(self.get_distances(landmarks, pair[0], pair[1]))
        embedding = np.asarray(embedding_list)
        return embedding

    def __call__(self, landmarks):
        landmarks = self.normalize_landmarks(landmarks)
        embedding = self.distance_embedding(landmarks)
        self.input_size = embedding.shape
        return embedding
