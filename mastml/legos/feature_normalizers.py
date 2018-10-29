"""
This module contains a collection of classes and methods for normalizing features. Also included is connection with
scikit-learn methods. See http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing for more info.
"""

from functools import wraps

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, Binarizer, StandardScaler, MaxAbsScaler, Normalizer, \
QuantileTransformer, RobustScaler, OneHotEncoder


from mastml.legos import util_legos

# TODO: add all sklearn preprocessors
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

def dataframify(transform):
    """
    Method which is a decorator transforms output of scikit-learn feature normalizers from array to dataframe.
    Enables preservation of column names.

    Args:

        transform: (function), a scikit-learn feature selector that has a transform method

    Returns:

        new_transform: (function), an amended version of the transform method that returns a dataframe

    """

    @wraps(transform)
    def new_transform(self, df):
        arr = transform(self, df.values)
        return pd.DataFrame(arr, columns=df.columns, index=df.index)
    return new_transform

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

MinMaxScaler.transform = dataframify(MinMaxScaler.transform)
Binarizer.transform = dataframify(Binarizer.transform)
StandardScaler.transform = dataframify(StandardScaler.transform)
MaxAbsScaler.transform = dataframify(MaxAbsScaler.transform)
Normalizer.transform = dataframify(Normalizer.transform)
QuantileTransformer.transform = dataframify(QuantileTransformer.transform)
RobustScaler.transform = dataframify(RobustScaler.transform)
OneHotEncoder.transform = dataframify(OneHotEncoder.transform)

name_to_constructor = {
    'Binarizer': Binarizer,
    'MeanStdevScaler': MeanStdevScaler,
    'MinMaxScaler': MinMaxScaler,
    'MaxAbsScaler': MaxAbsScaler,
    'Normalizer': Normalizer,
    'QuantileTransformer': QuantileTransformer,
    'RobustScaler': RobustScaler,
    'OneHotEncoder': OneHotEncoder,
    'DoNothing': util_legos.DoNothing,
    'StandardScaler': StandardScaler
}
