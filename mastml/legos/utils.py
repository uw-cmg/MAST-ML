"""
Collection of classes for debugging and control flow, plus a decorator.
"""
import pdb
from functools import wraps

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):
    " For unioning dataframe generators (sklearn.pipeline.FeatureUnion always puts out arrays) "
    def __init__(self, transforms):
        self.transforms = transforms
    def fit(self, X, y=None):
        for transform in self.transforms:
            transform.fit(X,y)
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
        print()
        print(f"{self.name}, fit data:")
        print(data[:self.head])
        print(data[-self.tail:])
        if correct is not None:
            print(f"{self.name}, fit correct:")
            print(correct[:self.head])
            print(correct[-self.tail:])
        return self
    def transform(self, data):
        print(f"{self.name}, transform data:")
        print(data[:self.head])
        print(data[-self.tail:])
        return data

def dataframify(transform):
    " Decorator to make a transformer's transform method work on dataframes. "
    @wraps(transform)
    def new_transform(self, df):
        arr = transform(self, df.values)
        try:
            return pd.DataFrame(arr, columns=df.columns, index=df.values)
        except ValueError:
            return pd.DataFrame(arr, index=df.values)
    return new_transform
