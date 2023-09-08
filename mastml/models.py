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

CrabNetModel:
    Class that provides an implementation of PyTorch-based CrabNet regressor model based on the following work:
    Wang, A., Kauwe, S., Murdock, R., Sparks, T. "Compositionally restricted attention-based network for
    " materials property predictions", npj Computational Materials (2021) (https://www.nature.com/articles/s41524-021-00545-1)


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
import os
import subprocess

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
try:
    from lineartree import LinearTreeRegressor, LinearForestRegressor, LinearBoostingRegressor
except:
    print('linear-tree is an optional dependency, enabling use of Linear tree, forest, and boosting models. If you want'
          ' to use this model, do "pip install linear-tree"')
try:
    from gplearn.genetic import SymbolicRegressor
except:
    print('gplearn is an optional dependency, enabling the use of genetic programming SymbolicRegressor model. If you'
          ' want to use this model, do "pip install gplearn"')

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
        elif model == 'LinearTreeRegressor':
            self.model = LinearTreeRegressor(**kwargs)
        elif model == 'LinearForestRegressor':
            self.model = LinearForestRegressor(**kwargs)
        elif model == 'LinearBoostingRegressor':
            self.model = LinearBoostingRegressor(**kwargs)
        elif model == 'SymbolicRegressor':
            self.model = SymbolicRegressor(**kwargs)
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


class HostedModel():

    def __init__(self, container_name):
        self.container_name = container_name

    def predict(self, df):

        df.to_csv('/mnt/test.csv', index=False)

        command = 'udocker --allow-root run -v '
        command += '{}:/mnt '.format(os.getcwd())
        command += self.container_name

        subprocess.check_output(
                                command,
                                shell=True
                                )

        df = pd.read_csv('/mnt/prediction.csv')

        os.remove('/mnt/test.csv')
        os.remove('/mnt/prediction.csv')

        return df

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


class CrabNetModel(BaseEstimator, TransformerMixin):
    '''
    Implementation of PyTorch-based CrabNet regressor model based on the following work:
        Wang, A., Kauwe, S., Murdock, R., Sparks, T. "Compositionally restricted attention-based network for
        " materials property predictions", npj Computational Materials (2021) (https://www.nature.com/articles/s41524-021-00545-1)

    The code to run CrabNet was integrated into MAST-ML based on source files available in this repository: https://github.com/anthony-wang/CrabNet

    Args:
        composition_column (str): string denoting the name of an input column containing materials compositions

        epochs (int): number of training epochs

        val_frac (float): fraction of input data to use as validation (0 to 1). Note this can be set to 0 for most MAST-ML
            runs as the splitting is done outside of this model.

        drop_unary (bool): whether or not to drop compositions containing one element. Should probably be False always. If true,
            can cause some array length mismatches.

        savepath (str): path to save the data to. This is set automatically in the data_splitters routine.

    Methods:
        get_model: method that prepares the CrabNet model. Called when the CrabNetModel class is instantiated.

        fit: method that fits the model parameters to the provided training data
            Args:
                X: (pd.DataFrame), dataframe of X data, needs to contain at least a column of material compositions. Note
                    that featurization is done internally by the model.

                y: (pd.Series), series of y target data

            Returns:
                fitted model

        predict: method that evaluates model on new data to give predictions
            Args:
                X: (pd.DataFrame), dataframe of X data, needs to contain at least a column of material compositions. Note
                    that featurization is done internally by the model.

                as_frame: (bool), whether to return data as pandas dataframe (else numpy array)

            Returns:
                series or array of predicted values
    '''
    def __init__(self, composition_column, epochs, val_frac, drop_unary=False, savepath=None):
        super(CrabNetModel, self).__init__()
        self.composition_column = composition_column
        self.epochs = epochs
        self.val_frac = val_frac
        self.drop_unary = drop_unary
        self.savepath = savepath

        self.get_model()

    def get_model(self):
        from mastml.crabnet.model import Model
        from mastml.crabnet.kingcrab import CrabNet
        from mastml.utils.get_compute_device import get_compute_device

        compute_device = get_compute_device(prefer_last=True)

        model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                      model_name='crabnet_model', verbose=True, drop_unary=self.drop_unary)

        self.model = model
        return

    def fit(self, X, y):
        # Crabnet needs to read a csv for data. The data needs compositions with "formula" as column name and targets with "target" as column name
        train_data = os.path.join(self.savepath, 'train_crabnet.csv')
        val_data = os.path.join(self.savepath, 'val_crabnet.csv')

        all_data = pd.DataFrame({'formula': X[self.composition_column].values, 'target': y.values})
        if self.val_frac > 0:
            from sklearn.model_selection import train_test_split
            trains, vals = train_test_split(all_data, test_size=self.val_frac)
        else:
            trains = all_data
        trains.to_csv(train_data, index=False)
        if self.val_frac > 0:
            vals.to_csv(val_data, index=False)

        data_size = trains.shape[0]
        batch_size = 2 ** round(np.log2(data_size) - 4)
        if batch_size < 2 ** 7:
            batch_size = 2 ** 7
        if batch_size > 2 ** 12:
            batch_size = 2 ** 12

        self.batch_size = batch_size
        self.model.load_data(train_data, batch_size=self.batch_size, train=True)

        print(f'training with batchsize {self.model.batch_size} '
              f'(2**{np.log2(self.model.batch_size):0.3f})')

        if self.val_frac > 0:
            self.model.load_data(val_data, batch_size=batch_size)

        # Set the number of epochs, decide if you want a loss curve to be plotted
        self.model.fit(epochs=self.epochs, losscurve=False)

        # Save the network (saved as f"{model_name}.pth")
        self.model.save_network(path=self.savepath)
        return

    def predict(self, X, as_frame=True):
        test_data = os.path.join(self.savepath, 'test_crabnet.csv')
        data = pd.DataFrame({'formula': X[self.composition_column].values, 'target': np.zeros(shape=X.shape[0])})
        data.to_csv(test_data, index=False)

        self.model.load_data(test_data, batch_size=self.batch_size, train=False)
        output = self.model.predict(self.model.data_loader)
        preds = output[1]

        if as_frame == True:
            return pd.DataFrame(preds, columns=['y_pred']).squeeze()
        else:
            return preds.ravel()

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
