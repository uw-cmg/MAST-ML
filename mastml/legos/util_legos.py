"""
Collection of classes for debugging and control flow
"""
import pdb
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

class Inspector(BaseEstimator, TransformerMixin):
    " Inspect the middle of the pipeline "
    def __init__(self):
        pass
    def fit(self, X, y=None):
        pdb.set_trace()
        return self
    def transform(self, X):
        pdb.set_trace()
        return X

class PrintHeadTail(BaseEstimator, TransformerMixin):
    " Print the beginning and end of input "
    def __init__(self, name="printer", head=5, tail=5):
        self.name = name
        self.head = head
        self.tail = tail
    def fit(self, data, correct=None):
        log.info()
        log.info(f"{self.name}, fit data:")
        log.info(data[:self.head])
        log.info(data[-self.tail:])
        if correct is not None:
            log.info(f"{self.name}, fit correct:")
            log.info(correct[:self.head])
            log.info(correct[-self.tail:])
        return self
    def transform(self, data):
        log.info(f"{self.name}, transform data:")
        log.info(data[:self.head])
        log.info(data[-self.tail:])
        return data
