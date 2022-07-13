from pose_estimation_models import MoveNet, BlazePose, EfficientPose
from feature_engineering import Normalization, EuclideanDistance, Angle, PairwiseDistance, NoFeatureEngineering
from classifiers import KNN, LogisticReg, DecisionTree, RandomForest, Xgboost, \
    MLP, CNN, ConvolutionLSTM, Attention, VisionTransformer, StackedEnsemble, AvgEnsemble

pose_estimation_dic = {
    "movenet": MoveNet,
    "blazepose": BlazePose,
    "efficientpose": EfficientPose
}

feature_engineering_dic = {
    "noFeatureEngineering": NoFeatureEngineering,
    "normalization": Normalization,
    "euclideanDistance": EuclideanDistance,
    "angles": Angle,
    "pairwiseDistance": PairwiseDistance
}

classifier_dic = {
    "knn": KNN,
    "logistic": LogisticReg,
    "decisionTree": DecisionTree,
    "randomForest": RandomForest,
    "xgboost": Xgboost,
    "mlp": MLP,
    "1DConv": CNN,
    "1DConvLstm": ConvolutionLSTM,
    "attention": Attention,
    "visionTransformer": VisionTransformer,
    "ensemble": StackedEnsemble,
    "ensembleAvg": AvgEnsemble
}

impossible_combinations = [
    ("efficientpose", "euclideanDistance"),
    ("efficientpose", "angles"),
    ("efficientpose", "pairwiseDistance")
]


def pose_estimation_object(pose_estimation_name):
    return pose_estimation_dic[pose_estimation_name]()


def feature_engineering_object(feature_engineering_name, landmark_names):
    return feature_engineering_dic[feature_engineering_name](landmark_names)


def classifier_object(classifier_name):
    return classifier_dic[classifier_name]()
