"""
This module provides a name_to_constructor dict for all models/estimators in scikit-learn, plus a couple test models and
error handling functions
"""

import pandas as pd
import sklearn.base
import sklearn.utils
from sklearn.ensemble import BaggingRegressor
import inspect
from pprint import pprint

from sklearn.base import BaseEstimator, TransformerMixin

try:
    import xgboost as xgb
except:
    print('If you want to use XGBoost models, please manually install xgboost package')

# TODO: add XGBoost functionality back in

class SklearnModel(BaseEstimator, TransformerMixin):
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

    def get_params(self, deep=True):
        return self.model.get_params(deep)

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

class EnsembleModel(BaseEstimator, TransformerMixin):
    '''

    '''
    def __init__(self, model, n_estimators, **kwargs):
        super(EnsembleModel, self).__init__()
        model = dict(sklearn.utils.all_estimators())[model](**kwargs)
        self.n_estimators = n_estimators
        self.model = BaggingRegressor(base_estimator=model, n_estimators=self.n_estimators)
        self.base_estimator_ = model.__class__.__name__

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X, as_frame=True):
        if as_frame == True:
            return pd.DataFrame(self.model.predict(X), columns=['y_pred']).squeeze()
        else:
            return self.model.predict(X).ravel()

    def get_params(self, deep=True):
        return self.model.get_params(deep)
