from argparse import ArgumentParser
from factory import pose_estimation_object, feature_engineering_object, classifier_object
from factory import pose_estimation_dic, feature_engineering_dic, classifier_dic, impossible_combinations
from gui import app


def parse_args():
    """
    Parses the command line arguments to decide on various game features.
    :return: the Parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('-p', '--pose-estimation', type=str,
                        choices=[pose_estimation_type for pose_estimation_type in pose_estimation_dic],
                        default='movenet', help='which pose estimation model use')
    parser.add_argument('-f', '--feature-engineering', type=str,
                        choices=[feature_engineering_type for feature_engineering_type in feature_engineering_dic],
                        default='pairwiseDistance', help='which feature engineering method to use')
    parser.add_argument('-c', '--classifier', type=str,
                        choices=[classifier_type for classifier_type in classifier_dic],
                        default='ensemble', help='which classifier to use')
    return parser.parse_args()


def main():
    """
    The main function running the application with the chosen pose estimation model, feature engineering method,
    and classifier.
    """
    # Parse arguments
    args = parse_args()

    if (args.pose_estimation, args.feature_engineering) in impossible_combinations:
        print("must use possible combinations between pose estimation models and feature engineering methods")
        exit()

    pose_estimation_model = pose_estimation_object(args.pose_estimation)
    feature_engineering = feature_engineering_object(args.feature_engineering, pose_estimation_model.get_landmark_names())
    classifier = classifier_object(args.classifier)
    app(pose_estimation_model, feature_engineering, classifier)


if __name__ == '__main__':
    main()
