"""
Collection of classes for debugging and control flow
"""
import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

log = logging.getLogger('mastml')

class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):
    " For unioning dataframe generators (sklearn.pipeline.FeatureUnion always puts out arrays) "
    def __init__(self, transforms):
        self.transforms = transforms
    def fit(self, X, y=None):
        for transform in self.transforms:
            transform.fit(X, y)
        return self
    def transform(self, X):
        dataframes = [transform.transform(X) for transform in self.transforms]
        return pd.concat(dataframes, axis=1)

class DoNothing(BaseEstimator, TransformerMixin):
    " Returns same input "
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
