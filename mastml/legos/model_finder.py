"""
This module provides a name_to_constructor dict for all models/estimators in scikit-learn, plus a couple test models and
error handling functions
"""

import warnings

import sklearn.base
import sklearn.utils.testing
import numpy as np

#from . import keras_models
from mastml import utils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    name_to_constructor = dict(sklearn.utils.testing.all_estimators())

class AlwaysFive(sklearn.base.RegressorMixin):
    """
    Class used as a test model that always predicts a value of 5.

    Args:

        constant: (int), the value to predict. Always 5 by default

    Methods:

        fit: Just passes through to maintain scikit-learn structure

        predict: Provides predicted model values based on X features

            Args:

                X: (numpy array), array of X features

            Returns:

                (numpy array), prediction array where all values are equal to constant

    """
    def __init__(self, constant = 5):
        self.five = constant

    def fit(self, X, y, groups=None):
        return self

    def predict(self, X):
        return np.array([self.five for _ in range(len(X))])

class RandomGuesser(sklearn.base.RegressorMixin):
    """
    Class used as a test model that always predicts random values for y data.

    Args:

        None

    Methods:

        fit: Constructs possible predicted values based on y data

            Args:

                y: (numpy array), array of y data

        predict: Provides predicted model values based on X features

            Args:

                X: (numpy array), array of X features

            Returns:

                (numpy array), prediction array where all values are random selections of y data

    """
    def __init__(self):
        pass

    def fit(self, X, y, groups=None):
        self.possible_answers = y
        return self

    def predict(self, X):
        return np.random.choice(self.possible_answers, size=X.shape[0])

custom_models = {
    'AlwaysFive': AlwaysFive,
    'RandomGuesser': RandomGuesser,
    #'DNNClassifier': keras_models.DNNClassifier
}

name_to_constructor.update(custom_models)

def find_model(model_name):
    """
    Method used to check model names conform to scikit-learn model/estimator names

    Args:

        model_name: (str), the name of a model/estimator

    Returns:

        (str), the scikit-learn model name or raises InvalidModel error

    """
    try:
        return name_to_constructor[model_name]
    except KeyError:
        raise utils.InvalidModel(f"Model '{model_name}' does not exist in scikit-learn.")

def check_models_mixed(model_names):
    """
    Method used to check whether the user has mixed regression and classification tasks

    Args:

        model_names: (list), list containing names of models/estimators

    Returns:

        (bool), whether or not a classifier was found, or raises exception if both regression and classification models present.

    """

    found_classifier = found_regressor = False
    for name in model_names:
        if name in custom_models: continue
        class1 = find_model(name)
        if issubclass(class1, sklearn.base.ClassifierMixin):
            found_classifier = True
        elif issubclass(class1, sklearn.base.RegressorMixin):
            found_regressor = True
        else:
            raise Exception(f"Model '{name}' is neither a classifier nor a regressor")

    if found_classifier and found_regressor:
        raise Exception("Both classifiers and regressor models have been included")

    return found_classifier

