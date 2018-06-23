"""
A collection of classes for normalizing features.
All classes here assume dataframe input and guarantee dataframe output.
(So no numpy arrays.)
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from . import util_legos, lego_utils

# TODO: add all sklearn preprocessors
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

class MeanStdevScaler(BaseEstimator, TransformerMixin):
    " Scales the mean and standard deviation of specified features to specified values "

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

class NoNormalize(BaseEstimator, TransformerMixin):
    " Returns X unmodified "
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

MinMaxScaler.transform = lego_utils.dataframify(MinMaxScaler.transform)

name_to_constructor = {
    'MeanStdevScaler': MeanStdevScaler,
    'MinMaxScaler': MinMaxScaler,
    'NoNormalize': NoNormalize,
    'DoNothing': util_legos.DoNothing,
}
