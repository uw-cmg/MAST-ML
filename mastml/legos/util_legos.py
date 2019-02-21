"""
This module contains a collection of classes for debugging and control flow
"""
import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

log = logging.getLogger('mastml')

class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):
    """
    Class for unioning dataframe generators (sklearn.pipeline.FeatureUnion always puts out arrays)

    Args:

        transforms: (list), list of scikit-learn functions, i.e. objects with a .fit or .transform method

    Methods:

        fit: Applies the .fit method for each transform

        Args:

            X: (numpy array), array of X features

        transform: Transforms the output of the scikit-learn transformer into a dataframe

        Args:

            X: (numpy array), array of X features

        Returns:

            (dataframe), concatenated dataframe after all scikit-learn transforms have been completed

    """

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
    """
    Class for having a "null" transform where the output is the same as the input. Needed by MAST-ML as a placeholder if
    certain workflow aspects are not performed.

    Args:

        None

    Methods:

        fit: does nothing, just returns object instance. Needed to maintain same structure as scikit-learn classes

        Args:

            X: (numpy array), array of X features

        transform: passes the input back out, in this case the array of X features

        Args:

            X: (numpy array), array of X features

        Returns:

            X: (numpy array), array of X features

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
