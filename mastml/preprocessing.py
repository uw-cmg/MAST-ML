"""
This module contains a collection of classes and methods for normalizing features. Also included is connection with
scikit-learn methods. See http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing for more info.
"""
import sklearn
import pandas as pd
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pprint import pprint
import inspect
from datetime import datetime
import joblib

class SklearnPreprocessor(BaseEstimator, TransformerMixin):
    '''


    '''
    def __init__(self, preprocessor, as_frame=False, **kwargs):
        super(SklearnPreprocessor, self).__init__()
        self.preprocessor = getattr(sklearn.preprocessing, preprocessor)(**kwargs)
        self.as_frame = as_frame

    def fit_transform(self, X, y=None, **fit_params):
        if self.as_frame:
            return pd.DataFrame(self.preprocessor.fit_transform(X=X), columns=X.columns, index=X.index)
        return self.preprocessor.fit_transform(X=X)

    def evaluate(self, X, y=None, savepath=None):
        if not savepath:
            savepath = os.getcwd()
        splitdir = self._setup_savedir(savepath=savepath)
        if self.as_frame:
            Xnew = pd.DataFrame(self.preprocessor.fit_transform(X=X), columns=X.columns, index=X.index)
            Xnew.to_excel(os.path.join(splitdir, 'data_preprocessed.xlsx'))
        else:
            Xnew = self.preprocessor.fit_transform(X=X)
            np.savetxt(os.path.join(splitdir, 'data_preprocessed.csv'), Xnew)

        # Save the fitted preprocessor, will be needed for DLHub upload later on
        joblib.dump(self.preprocessor, os.path.join(splitdir, str(self.preprocessor.__class__.__name__) + ".pkl"))
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
        dirname = f"{dirname}_{now.month:02d}_{now.day:02d}" \
                        f"_{now.hour:02d}_{now.minute:02d}_{now.second:02d}"
        if savepath == None:
            splitdir = os.getcwd()
        else:
            splitdir = os.path.join(savepath, dirname)
        if not os.path.exists(splitdir):
            os.mkdir(splitdir)
        return splitdir


# TODO: add this into new method
class MeanStdevScaler(BaseEstimator, TransformerMixin):
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

    def __init__(self, features=None, mean=0, stdev=1):
        self.features = features
        self.mean = mean
        self.stdev = stdev

    def fit(self, df, y=None):
        if self.features is None:
            self.features = df.columns
        self.old_mean  = df[self.features].values.mean()
        self.old_stdev = df[self.features].values.std()
        return self

    def transform(self, df):
        array = df[self.features].values
        array = ((array - self.old_mean) / self.old_stdev) * self.stdev + self.mean
        same = df.drop(columns=self.features)
        changed = pd.DataFrame(array, columns=self.features, index=df.index) # don't forget index!!
        return pd.concat([same, changed], axis=1)

    def inverse_transform(self, df):
        array = df[self.features].values
        array = ((array - self.mean) / self.stdev) * self.old_stdev + self.old_mean
        same = df.drop(columns=self.features)
        changed = pd.DataFrame(array, columns=self.features, index=df.index)
        return pd.concat([same, changed], axis=1)



