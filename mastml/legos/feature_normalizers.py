"""
A collection of classes for normalizing features.
All classes here assume dataframe input and guarantee dataframe output.
(So no numpy arrays.)
"""

from functools import wraps

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, Binarizer, StandardScaler

from . import util_legos

# TODO: add all sklearn preprocessors
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

def dataframify(transform):
    """
    Decorator to make a transformer's transform method work on dataframes
    Assumes columns will be preserved
    """
    @wraps(transform)
    def new_transform(self, df):
        arr = transform(self, df.values)
        return pd.DataFrame(arr, columns=df.columns, index=df.index)
    return new_transform

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

StandardScaler.transform = dataframify(StandardScaler.transform)
MinMaxScaler.transform = dataframify(MinMaxScaler.transform)
Binarizer.transform = dataframify(Binarizer.transform)

name_to_constructor = {
    'Binarizer': Binarizer,
    'MeanStdevScaler': MeanStdevScaler,
    'MinMaxScaler': MinMaxScaler,
    'DoNothing': util_legos.DoNothing,
    'StandardScaler' : StandardScaler
}
