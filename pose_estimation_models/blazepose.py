from pose_estimation_models.pose_estimation_logic import PoseEstimationLogic
from mediapipe.python.solutions import pose as mp_pose
import numpy as np


class BlazePose(PoseEstimationLogic):
    def __init__(self):
        super(BlazePose, self).__init__("blazepose")
        self.pose_tracker = None

    def load_model(self):
        pass

    def start(self):
        self.pose_tracker = mp_pose.Pose()

    def process_frame(self, frame):
        result = self.pose_tracker.process(image=frame)
        pose_landmarks = result.pose_landmarks
        if pose_landmarks is not None:
            pose_landmarks = np.array([[lmk.x * self.image_width, lmk.y * self.image_height, lmk.z * self.image_width] 
                                       for lmk in pose_landmarks.landmark], dtype=np.float32)
        return pose_landmarks
    
    def end(self):
        self.pose_tracker.close()

    def get_landmark_names(self):
        landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]
        return landmark_names
