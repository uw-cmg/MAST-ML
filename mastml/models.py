"""
This module provides a name_to_constructor dict for all models/estimators in scikit-learn, plus a couple test models and
error handling functions
"""

import random
import pandas as pd
import joblib
import numpy as np
from scipy import stats
import sklearn.base
import sklearn.utils
import inspect
from pprint import pprint

try:
    import xgboost as xgb
except:
    print('If you want to use XGBoost models, please manually install xgboost package')

class SklearnModel():
    """
    Class to wrap any sklearn estimator, and provide some new dataframe functionality

    Args:

        model: (str), string denoting the name of an sklearn estimator object, e.g. KernelRidge
        kwargs: keyword pairs of values to include for model, e.g. for KernelRidge can specify kernel, alpha, gamma values

    Methods:

        fit: method that fits the model parameters to the provided training data

            Args:
                X: (pd.DataFrame), dataframe of X features
                y: (pd.Series), series of y target data
            Returns:
                fitted model

        predict: method that evaluates model on new data to give predictions

            Args:
                X: (pd.DataFrame), dataframe of X features
                as_frame: (bool), whether to return data as pandas dataframe (else numpy array)
            Returns:
                series or array of predicted values

        help: method to output key information on class use, e.g. methods and parameters

            Args:
                None

            Returns:
                None, but outputs help to screen
    """
    def __init__(self, model, **kwargs):
        self.model = dict(sklearn.utils.all_estimators())[model](**kwargs)

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X, as_frame=True):
        if as_frame == True:
            return pd.DataFrame(self.model.predict(X), columns=['y_pred']).squeeze()
        else:
            return self.model.predict(X).ravel()

    def help(self):
        print('Documentation for', self.model)
        pprint(dict(inspect.getmembers(self.model))['__doc__'])
        print('\n')
        print('Class methods for,', self.model)
        pprint(dict(inspect.getmembers(self.model, predicate=inspect.ismethod)))
        print('\n')
        print('Class attributes for,', self.model)
        pprint(self.model.__dict__)
        return

#TODO: add the below models into new formulation
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
#custom_models = {
#        'ModelImport': ModelImport,
#        'XGBRegressor': xgb.XGBRegressor,
#        'XGBClassifier': xgb.XGBClassifier,
#        'EnsembleRegressor': EnsembleRegressor
#    }