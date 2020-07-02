"""
This module provides a name_to_constructor dict for all models/estimators in scikit-learn, plus a couple test models and
error handling functions
"""

import warnings
import inspect

import sklearn.base
import sklearn.utils.testing
import joblib
import numpy as np
import os

# Sometimes xgboost is hard to install so make it optional
try:
    import xgboost as xgb
except:
    pass

import keras
from keras.models import model_from_json
from keras.models import load_model
from keras.models import Sequential

import random

import pandas as pd

#from . import keras_models
from mastml import utils

from scipy import stats

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

class KerasRegressor():
    def __init__(self, conf_dict):
        self.conf_dict = conf_dict
        self.model = self.build_model()

    def build_model(self):
        model_vals = self.conf_dict
        model = Sequential()

        for layer_dict, layer_val in model_vals.items():
            if (layer_dict != 'FitParams'):
                layer_type = layer_val.get('layer_type')
                layer_name_asstr = layer_type
                if layer_name_asstr == 'Dense':
                    neuron_num = int(layer_val.get('neuron_num'))
                    if (layer_dict == 'Layer1'):
                        input_dim = int(layer_val.get('input_dim'))
                    kernel_initializer = layer_val.get('kernel_initializer')
                    activation = layer_val.get('activation')
                elif layer_name_asstr == 'Dropout':
                    rate = float(layer_val.get('rate'))
                for layer_name, cls in inspect.getmembers(keras.layers, inspect.isclass):
                    layer_type = getattr(keras.layers, layer_name_asstr)  # (neuron_num)

            else:
                if layer_val.get('rate'):
                    self.rate = float(layer_val.get('rate'))
                if layer_val.get('epochs'):
                    self.epochs = int(layer_val.get('epochs'))
                else:
                    self.epochs = 1
                if layer_val.get('batch_size'):
                    self.batch_size = int(layer_val.get('batch_size'))
                else:
                    self.batch_size = None
                if layer_val.get('loss'):
                    self.loss = str(layer_val.get('loss'))
                else:
                    self.loss = 'mean_squared_error'
                if layer_val.get('optimizer'):
                    self.optimizer = str(layer_val.get('optimizer'))
                else:
                    self.optimizer = 'adam'
                if layer_val.get('metrics'):
                    self.metrics = layer_val.get('metrics').split(',')
                else:
                    self.metrics = ['mae']
                if layer_val.get('verbose'):
                    self.verbose = str(layer_val.get('verbose'))
                else:
                    self.verbose = 0
                if layer_val.get('shuffle'):
                    self.shuffle = bool(layer_val.get('shuffle'))
                else:
                    self.shuffle = True
                if layer_val.get('validation_split'):
                    self.validation_split = float(layer_val.get('validation_split'))
                else:
                    self.validation_split = 0.0
                continue

            if (layer_dict == 'Layer1'):
                model.add(layer_type(neuron_num, input_dim=input_dim, kernel_initializer=kernel_initializer,
                                     activation=activation))

            else:
                if layer_name_asstr == 'Dense':
                    model.add(layer_type(neuron_num, kernel_initializer=kernel_initializer, activation=activation))
                if layer_name_asstr == 'Dropout':
                    model.add(layer_type(rate=rate))

        return model

    def fit(self, X, Y):
        # Need to rebuild and re-compile model at every fit instance so don't have information of weights from other fits
        self.model = self.build_model()
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        return self.model.fit(X, Y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
                              validation_split=self.validation_split, shuffle=self.shuffle)

    def predict(self, X):
        return self.model.predict(X)

    def summary(self):
        return self.model.summary()

# ref: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html#sklearn.ensemble.BaggingRegressor
# NOTE: in order to use this, other models for the custom ensemble must be defined 
#       in the conf file with "_ensemble" somewhere in the name
class EnsembleRegressor():
    def __init__(self, n_estimators, num_samples, model_list, num_models):
        self.model_list = model_list # should be list of strings
        self.num_models = num_models # how many of each of the specified models should be included in the ensemble
        self.n_estimators = sum(self.num_models)
        self.num_samples = num_samples
        self.max_samples = num_samples
        self.bootstrapped_datasets = []
        self.bootstrapped_idxs = []
        self.all_preds = []
        self.path = ""
        self.model = self.build_models() # actually a list of models for use as the members in the ensemble

        self.fold = -1

        self.bootstrap = True

    def build_models(self):
        model = []

        for i, num_m in enumerate(self.num_models):
            for j in range(num_m):
                model.append(self.model_list[i])

        return model

    def setup(self, path):
        self.fold += 1
        self.bootstrapped_idxs = []
        self.bootstrapped_datasets = []
        self.path = path

    def fit(self, X, Y):
        X = X.values
        Y = Y.values

        idxs = np.arange(len(X))
        # fit each model in the ensemble
        for i in range(self.n_estimators):
            model = self.model[i]

            # do bootstrapping given the validation data
            bootstrap_idxs = random.choices(idxs, k=self.num_samples)
            bootstrap_X = X[bootstrap_idxs]
            bootstrap_Y = Y[bootstrap_idxs]
            if 1 == len(bootstrap_X.shape):
                bootstrap_X = np.expand_dims(np.asarray(bootstrap_X), -1)
            if 1 == len(bootstrap_Y.shape):
                bootstrap_Y = np.expand_dims(np.asarray(bootstrap_Y), -1)

            self.bootstrapped_idxs.append(bootstrap_idxs)
            self.bootstrapped_datasets.append(bootstrap_X)
            model.fit(bootstrap_X, bootstrap_Y)

    def predict(self, X, return_std=False):

        if isinstance(X, pd.DataFrame):
            X = X.values

        all_preds = []
        means = []

        for x_i in range(len(X)):
            preds = []
            for i in range(self.n_estimators):
                sample_X = X[x_i]
                if 1 == len(sample_X.shape):
                    sample_X = np.expand_dims(np.asarray(sample_X), 0)
                preds.append(self.model[i].predict(sample_X))
            all_preds.append(preds)
            means.append(np.mean(preds))

            # NOTE for ref (if manual jackknife implementation is necessary)
            # https://www.jpytr.com/post/random_forests_and_jackknife_variance/
            # https://github.com/scikit-learn-contrib/forest-confidence-interval/tree/master/forestci
            # http://contrib.scikit-learn.org/forest-confidence-interval/reference/forestci.html

        self.all_preds = all_preds

        return np.asarray(means)

    # check for failed fits, warn users, and re-calculate
    def stats_check_models(self, X, Y):
        if self.n_estimators > 10:
            maes = []
            for i in range(self.n_estimators):
                abs_errors = np.absolute(np.absolute(np.squeeze(np.asarray(self.all_preds)[:,i])) - Y)
                maes.append(sum(abs_errors) / len(abs_errors))

            alpha = 0.01
            bad_idxs = []
            for i in range(self.n_estimators):
                other_maes = np.delete(maes, [i])
                # ref: https://towardsdatascience.com/statistical-significance-hypothesis-testing-the-normal-curve-and-p-values-93274fa32687
                z_score = (maes[i] - np.mean(other_maes)) / np.std(other_maes)
                # ref: https://stackoverflow.com/questions/3496656/convert-z-score-z-value-standard-score-to-p-value-for-normal-distribution-in/3508321
                p_val = stats.norm.sf(abs(z_score))*2

                if p_val <= alpha:
                    # TODO ok to print these/how to print/log properly?
                    print("Estimator {} failed under statistical significance threshold {} (p_val {}), relevant dataset output to file with name format \'<fold>_<estimator idx>_bootstrapped_dataset.csv\'".format(i, alpha, p_val))
                    print("bad estimator mae: {}".format(maes[i]))
                    print("mean mae (for ref):")
                    print(np.mean(maes))
                    np.savetxt(self.path + "\\{}_{}_bootstrapped_dataset.csv".format(self.fold, i), self.bootstrapped_datasets[i], delimiter=",")
                    bad_idxs.append(i)

            if len(bad_idxs) == self.n_estimators:
                print("ALL models failed, wtf is your data")
                return
            #self.all_preds = np.delete(self.all_preds, bad_idxs, 1)

        y_preds = []
        for idx, x_i in enumerate(self.all_preds):
            y_preds.append(np.mean(x_i))

        return np.asarray(y_preds)

class ModelImport():
    """
    Class used to import pickled models from previous machine learning fits

    Args:

        model_path (str): string designating the path to load the saved .pkl model file

    Methods:

        fit: Does nothing, present for compatibility purposes

            Args:

                X: Nonetype

                y: Nonetype

                groups: Nonetype

        predict: Provides predicted model values based on X features

            Args:

                X: (numpy array), array of X features

            Returns:

                (numpy array), prediction array using imported model

    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = joblib.load(self.model_path)

    def fit(self, X=None, y=None, groups=None):
        """ Only here for compatibility """
        return

    def predict(self, X):
        return self.model.predict(X)

# Optional to have xgboost working
try:
    custom_models = {
        'AlwaysFive': AlwaysFive,
        'RandomGuesser': RandomGuesser,
        'ModelImport': ModelImport,
        'XGBRegressor': xgb.XGBRegressor,
        'XGBClassifier': xgb.XGBClassifier,
        'KerasRegressor': KerasRegressor,
        'EnsembleRegressor': EnsembleRegressor
        #'DNNClassifier': keras_models.DNNClassifier
    }
except NameError:
    custom_models = {
        'AlwaysFive': AlwaysFive,
        'RandomGuesser': RandomGuesser,
        'ModelImport': ModelImport,
        'KerasRegressor': KerasRegressor,
        'EnsembleRegressor': EnsembleRegressor
        # 'DNNClassifier': keras_models.DNNClassifier
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

