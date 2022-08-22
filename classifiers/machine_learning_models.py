from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
from abc import abstractmethod

from classifiers.classifier_logic import Classifier
from create_dataset import create_data_folder


class MLCLassifier(Classifier):
    """An abstract class representing a machine learning classifier"""
    def __init__(self, model_name):
        """Initializing a machine learning classifier."""
        super(MLCLassifier, self).__init__(model_name)

    @abstractmethod
    def define_model(self, input_size, output_size):
        """Defines the ML model used"""
        pass

    def prepare_input(self, X, y):
        """Prepares the input before training for sklearn."""
        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
        return X, y

    def load(self, model_path):
        """Loads the model from the given path."""
        self.model = joblib.load(f'{model_path}/{self.model_name}.joblib')

    def save(self, model_path):
        """Saves the model in the given path."""
        create_data_folder(model_path)
        joblib.dump(self.model, f'{model_path}/{self.model_name}.joblib')

    def predict(self, model_input):
        """Returns a softmax vector of prediction probabilities given the input."""
        return self.model.predict_proba(model_input.reshape(1, -1))

    def evaluate(self, X, y):
        """Evaluates and returns the accuracy of the predicted labels from X against the ground truth y."""
        return self.model.score(X, y)

    def train_model(self, X_train, X_val, y_train, y_val):
        """Trains the model by the given train and validation sets."""
        self.model.fit(X_train, y_train)
        print(self.model_name + " accuracy: ", self.model.score(X_val, y_val))


class KNN(MLCLassifier):
    """A K-Nearest Neighbors class extending the MLClassifier class"""
    def __init__(self):
        super(KNN, self).__init__("knn")

    def define_model(self, input_size, output_size):
        """Defines the KNN - parameters received from a grid search against custom test sets."""
        self.model = KNeighborsClassifier(n_neighbors=1, weights='uniform', leaf_size=20, metric='manhattan')


class LogisticReg(MLCLassifier):
    """A Logistic Regression class extending the MLClassifier class"""
    def __init__(self):
        super(LogisticReg, self).__init__("logistic")

    def define_model(self, input_size, output_size):
        """Defines the Logistic Regression - parameters received from a grid search against custom test sets."""
        self.model = LogisticRegression(C=11.288378916846883, penalty='l2', solver='liblinear')


class DecisionTree(MLCLassifier):
    """A Decision Tree class extending the MLClassifier class"""
    def __init__(self):
        super(DecisionTree, self).__init__("decisionTree")

    def define_model(self, input_size, output_size):
        """Defines the Decision Tree - parameters received from a grid search against custom test sets."""
        self.model = DecisionTreeClassifier(criterion='gini', max_depth=70, min_samples_split=4, min_samples_leaf=1)


class RandomForest(MLCLassifier):
    """A Random Forest class extending the MLClassifier class"""
    def __init__(self):
        super(RandomForest, self).__init__("randomForest")

    def define_model(self, input_size, output_size):
        """Defines the Random Forest - parameters received from a grid search against custom test sets."""
        self.model = RandomForestClassifier(criterion='gini', max_depth=70, min_samples_split=4, min_samples_leaf=1)


class Xgboost(MLCLassifier):
    """An Extreme gradient boosting class extending the MLClassifier class"""
    def __init__(self):
        super(Xgboost, self).__init__("xgboost")

    def define_model(self, input_size, output_size):
        """Defines the XGBoost - parameters received from a grid search against custom test sets."""
        self.model = XGBClassifier(objective='multi:softmax', num_class=output_size, subsample=0.8, min_child_weight=5,
                                   max_depth=3, learning_rate=0.1, gamma=1.5,colsample_bytree=0.6)
