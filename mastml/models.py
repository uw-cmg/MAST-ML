"""
Module for constructing models for use in MAST-ML.

SklearnModel:
    Class that wraps scikit-learn models to have MAST-ML type functionality. Providing the model name as a string
    and the keyword arguments for the model parameters will construct the model. Note that this class also supports
    construction of XGBoost models and Keras neural network models via Keras' keras.wrappers.scikit_learn.KerasRegressor
    model.

EnsembleModel:
    Class that constructs a model which is an ensemble of many base models (sometimes called weak learners). This
    class supports construction of ensembles of most scikit-learn regression models as well as ensembles of neural
    networks that are made via Keras' keras.wrappers.scikit_learn.KerasRegressor class.

"""

import pandas as pd
import sklearn.base
import sklearn.utils
from sklearn.ensemble import BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import inspect
from pprint import pprint
import numpy as np
import re

from sklearn.base import BaseEstimator, TransformerMixin

try:
    import xgboost
except:
    print('XGBoost is an optional dependency. If you want to use XGBoost models, please manually install xgboost package with '
          'pip install xgboost. If have error with finding libxgboost.dylib library, do'
          'brew install libomp. If do not have brew on your system, first do'
          ' ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" from the Terminal')
try:
    from sklego.linear_model import LowessRegression
except:
    print('scikit-lego is an optional dependency, enabling use of the LowessRegression model. If you want to use this model, '
          'do "pip install scikit-lego"')

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
        if model == 'XGBoostRegressor':
            self.model = xgboost.XGBRegressor(**kwargs)
        elif model == 'GaussianProcessRegressor':
            kernel = kwargs['kernel']
            kernel = _make_gpr_kernel(kernel_string=kernel)
            del kwargs['kernel']
            self.model = GaussianProcessRegressor(kernel=kernel, **kwargs)
        elif model == 'LowessRegression':
            self.model = LowessRegression(**kwargs)
        else:
            self.model = dict(sklearn.utils.all_estimators())[model](**kwargs)

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X, as_frame=True):
        if as_frame == True:
            return pd.DataFrame(self.model.predict(X), columns=['y_pred']).squeeze()
        else:
            return self.model.predict(X).ravel()

    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)

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
    """
    Class used to construct ensemble models with a particular number and type of weak learner (base model). The
    ensemble model is compatible with most scikit-learn regressor models and KerasRegressor models

    Args:
        model: (str), string name denoting the name of the model type to use as the base model

        n_estimators: (int), the number of base models to include in the ensemble

        kwargs: keyword arguments for the base model parameter names and values

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

        get_params: method to output key model parameters
            Args:
                deep: (bool), determines the extent of information returned, default True

            Returns:
                information on model parameters
    """
    def __init__(self, model, n_estimators, **kwargs):
        super(EnsembleModel, self).__init__()
        try:
            if model == 'XGBoostRegressor':
                model = xgboost.XGBRegressor(**kwargs)
            elif model == 'GaussianProcessRegressor':
                kernel = kwargs['kernel']
                kernel = _make_gpr_kernel(kernel_string=kernel)
                del kwargs['kernel']
                model = GaussianProcessRegressor(kernel=kernel, **kwargs)
            else:
                model = dict(sklearn.utils.all_estimators())[model](**kwargs)
        except:
            print('Could not find designated model type in scikit-learn model library. Note the other supported model'
                  'type is the keras.wrappers.scikit_learn.KerasRegressor model')
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


def _make_gpr_kernel(kernel_string):
    """
    Method to transform a supplied string to a kernel object for use in GPR models

    Args:
        kernel_string: (str), a string containing the desired name of the kernel

    Return:
        kernel: sklearn.gaussian_process.kernels object

    """
    kernel_list = ['WhiteKernel', 'RBF', 'ConstantKernel', 'Matern', 'RationalQuadratic', 'ExpSineSquared', 'DotProduct']
    kernel_operators = ['+', '*', '-']
    # Parse kernel_string to identify kernel types and any kernel operations to combine kernels
    kernel_types_asstr = list()
    kernel_types_ascls = list()
    kernel_operators_used = list()

    for s in kernel_string[:]:
        if s in kernel_operators:
            kernel_operators_used.append(s)

    # Do case for single kernel, no operators
    if len(kernel_operators_used) == 0:
        kernel_types_asstr.append(kernel_string)
    else:
        # New method, using re
        unique_operators = np.unique(kernel_operators_used).tolist()
        unique_operators_asstr = '['
        for i in unique_operators:
            unique_operators_asstr += str(i)
        unique_operators_asstr += ']'
        kernel_types_asstr = re.split(unique_operators_asstr, kernel_string)

    for kernel in kernel_types_asstr:
        kernel_ = getattr(sklearn.gaussian_process.kernels, kernel)
        kernel_types_ascls.append(kernel_())

    # Case for single kernel
    if len(kernel_types_ascls) == 1:
        kernel = kernel_types_ascls[0]

    kernel_count = 0
    for i, operator in enumerate(kernel_operators_used):
        if i+1 <= len(kernel_operators_used):
            if operator == "+":
                if kernel_count == 0:
                    kernel = kernel_types_ascls[kernel_count] + kernel_types_ascls[kernel_count+1]
                else:
                    kernel += kernel_types_ascls[kernel_count+1]
            elif operator == "*":
                if kernel_count == 0:
                    kernel = kernel_types_ascls[kernel_count] * kernel_types_ascls[kernel_count+1]
                else:
                    kernel *= kernel_types_ascls[kernel_count+1]
            else:
                print('Warning: You have chosen an invalid operator to construct a composite kernel. Please choose'
                              ' either "+" or "*".')
            kernel_count += 1

    return kernel
