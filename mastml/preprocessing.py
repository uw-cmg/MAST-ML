"""
This module contains methods to perform data preprocessing, such as various standardization/normalization methods

BasePreprocessor:
    Base class that adds some MAST-ML type functionality to other preprocessors. Other preprocessor classes all inherit
    this base class

SklearnPreprocessor:
    Class that wraps any preprocessor method from scikit-learn (e.g. StandardScaler) to have MAST-ML type functionality

NoPreprocessor:
    Class that performs no preprocessing. A preprocessor is needed in the MAST-ML evaluation of data splits. If no
    preprocessing is desired, then this NoPreprocessor class is invoked by default

MeanStdevScaler:
    Preprocessor class which extends scikit-learn's StandardScaler to scale the dataset to a particular user-specified
    mean and standard deviation value
"""

import sklearn.preprocessing
import pandas as pd
import os
import numpy as np
from pprint import pprint
import inspect
from datetime import datetime
import joblib

from sklearn.base import BaseEstimator, TransformerMixin

class BasePreprocessor(BaseEstimator, TransformerMixin):
    """
    Base class to provide new methods beyond sklearn fit_transform, such as dataframe support and directory management

    Args:
        preprocessor : a sklearn.preprocessor object, e.g. StandardScaler or mastml.preprocessing object

    Methods:
        fit_transform: method that fits the data to the preprocessor, then transforms it to the preprocessed data
            Args:
                X: (pd.DataFrame), dataframe of X features

                y: (pd.Series), series of y target data

            Returns:
                Transformed data (pd.DataFrame or numpy array based on self.as_frame)

        evaluate: main method to evaluate a preprocessor, build directory and save data output
            Args:
                X: (pd.DataFrame), dataframe of X features

                y: (pd.Series), series of y target data

                savepath: (str), string containing main savepath to construct splits for saving output

                file_extension: (str), must be either '.xlsx' or '.csv', determines data file type for saving

            Returns:
                Xnew (pd.DataFrame or numpy array), dataframe or array of the preprocessed X features

        help: method to output key information on class use, e.g. methods and parameters
            Args:
                None

            Returns:
                None, but outputs help to screen

        _setup_savedir: method to create a savedir based on the provided model, splitter, selector names and datetime
            Args:
                model: (mastml.models.SklearnModel or other estimator object), an estimator, e.g. KernelRidge

                selector: (mastml.feature_selectors or other selector object), a selector, e.g. EnsembleModelFeatureSelector

                savepath: (str), string designating the savepath

            Returns:
                splitdir: (str), string containing the new subdirectory to save results to
    """
    def __init__(self, preprocessor, as_frame=False):
        self.preprocessor = preprocessor
        self.as_frame = as_frame

    def fit(self, X):
        return self.preprocessor.fit(X)

    def transform(self, X):
        if self.as_frame:
            return pd.DataFrame(self.preprocessor.transform(X=X), columns=X.columns, index=X.index)
        return self.preprocessor.transform(X=X)

    def inverse_transform(self, X):
        return pd.DataFrame(self.preprocessor.inverse_transform(X), columns=X.columns, index=X.index)

    def fit_transform(self, X, y=None, **fit_params):
        if self.as_frame:
            return pd.DataFrame(self.preprocessor.fit_transform(X=X), columns=X.columns, index=X.index)
        return self.preprocessor.fit_transform(X=X)

    def evaluate(self, X, y=None, savepath=None, file_name='', make_new_dir=False, file_extension='.csv'):
        if not savepath:
            savepath = os.getcwd()
        if make_new_dir is True:
            splitdir = self._setup_savedir(savepath=savepath)
            self.splitdir = splitdir
            savepath = splitdir
        if self.as_frame:
            Xnew = pd.DataFrame(self.preprocessor.fit_transform(X=X), columns=X.columns, index=X.index)
            if file_extension == '.xlsx':
                Xnew.to_excel(os.path.join(savepath, 'data_preprocessed_'+file_name+'.xlsx'))
            elif file_extension == '.csv':
                Xnew.to_csv(os.path.join(savepath, 'data_preprocessed_' + file_name + '.csv'))
        else:
            Xnew = self.preprocessor.fit_transform(X=X)
            np.savetxt(os.path.join(savepath, 'data_preprocessed_'+file_name+'.csv'), Xnew)

        # Save the fitted preprocessor, will be needed for DLHub upload later on
        joblib.dump(self, os.path.join(savepath, str(self.preprocessor.__class__.__name__) + ".pkl"))
        self.savepath = savepath
        return Xnew

    def help(self):
        print('Documentation for', self.preprocessor)
        pprint(dict(inspect.getmembers(self.preprocessor))['__doc__'])
        print('\n')
        print('Class methods for,', self.preprocessor)
        pprint(dict(inspect.getmembers(self.preprocessor, predicate=inspect.ismethod)))
        print('\n')
        print('Class attributes for,', self.preprocessor)
        pprint(self.preprocessor.__dict__)
        return

    def _setup_savedir(self, savepath):
        now = datetime.now()
        dirname = self.preprocessor.__class__.__name__
        dirname = f"{dirname}_{now.year:02d}_{now.month:02d}_{now.day:02d}" \
                        f"_{now.hour:02d}_{now.minute:02d}_{now.second:02d}"
        if savepath == None:
            splitdir = os.getcwd()
        else:
            splitdir = os.path.join(savepath, dirname)
        if not os.path.exists(splitdir):
            os.mkdir(splitdir)
        return splitdir

class SklearnPreprocessor(BasePreprocessor):
    """
    Class to wrap any scikit-learn preprocessor, e.g. StandardScaler

    Args:
        preprocessor (str): name of a sklearn.preprocessor object, e.g. StandardScaler

        as_frame (bool): whether to return data as a dataframe

        kwargs : key word arguments for the sklearn.preprocessor object

    Methods:

        See documentation of BasePreprocessor
    """
    def __init__(self, preprocessor, as_frame=False, **kwargs):
        super(SklearnPreprocessor, self).__init__(preprocessor=preprocessor)
        self.preprocessor = getattr(sklearn.preprocessing, preprocessor)(**kwargs)
        self.as_frame = as_frame

class NoPreprocessor(BasePreprocessor):
    '''
    Class for having a "null" transform where the output is the same as the input. Needed by MAST-ML as a placeholder if
    certain workflow aspects are not performed.

    See BasePreprocessor for information on args and methods
    '''
    def __init__(self, preprocessor=None, as_frame=False):
        super(NoPreprocessor, self).__init__(preprocessor=self)
        self.as_frame = as_frame

    def fit(self, X):
        return X

    def transform(self, X):
        if self.as_frame:
            return pd.DataFrame(X, columns=X.columns, index=X.index)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        if self.as_frame:
            return pd.DataFrame(X, columns=X.columns, index=X.index)
        return X

class MeanStdevScaler(BasePreprocessor):
    """
    Class designed to normalize input data to a specified mean and standard deviation

    Args:
        mean: (int/float), specified normalized mean of the data

        stdev: (int/float), specified normalized standard deviation of the data

    Methods:
        fit: Obtains initial mean and stdev of data
            Args:
                df: (dataframe), dataframe of values to be normalized

            Returns:
                (self, the object instance)

        transform: Normalizes the data to new mean and stdev values
            Args:
                df: (dataframe), dataframe of values to be normalized

            Returns:
                (dataframe), dataframe containing re-normalized data and any data that wasn't normalized

        inverse_transform: Un-normalizes the data to the old mean and stdev values
            Args:
                df: (dataframe), dataframe of values to be un-normalized

            Returns:
                (dataframe), dataframe containing un-normalized data and any data that wasn't normalized

    """

    def __init__(self, mean=0, stdev=1, as_frame=False):
        super(MeanStdevScaler, self).__init__(preprocessor=self, as_frame=as_frame)
        self.mean = mean
        self.stdev = stdev

    def fit_transform(self, X, y=None, **fit_params):
        self.features = X.columns.tolist()
        X_trans = list()
        for feature in self.features:
            array = X[feature].values
            mean = X[feature].mean()
            stdev = X[feature].std()
            array = ((array - mean) / stdev) * self.stdev + self.mean
            X_trans.append(array)
        X_trans = pd.DataFrame(np.array(X_trans).T, columns=self.features, index=X.index)
        return X_trans



