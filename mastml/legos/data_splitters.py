"""
The data_splitters module contains a collection of classes for generating (train_indices, test_indices) pairs from
a dataframe or a numpy array.

For more information and a list of scikit-learn splitter classes, see:
 http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.model_selection as ms

class SplittersUnion(BaseEstimator, TransformerMixin):
    """
    Class to take the union of two separate splitting routines, so that many splitting routines can be performed at once

    Args:
        splitters: (list), a list of scikit-learn splitter objects

    Methods:
        get_n_splits: method to calculate the number of splits to perform across all splitters

            Args:
                X: (numpy array), array of X features
                y: (numpy array), array of y data
                groups: (numpy array), array of group labels

            Returns:
                (int), number of total splits to be conducted

        split: method to perform split into train indices and test indices

            Args:
                X: (numpy array), array of X features
                y: (numpy array), array of y data
                groups: (numpy array), array of group labels

            Returns:
                (numpy array), array of train and test indices

    """
    def __init__(self, splitters):
        self.splitters = splitters

    def get_n_splits(self, X, y, groups=None):
        return sum(splitter.get_n_splits(X, y, groups) for splitter in self.splitters)

    def split(self, X, y, groups=None):
        for splitter in self.splitters:
            yield from splitter.split(X, y, groups)

class NoSplit(BaseEstimator, TransformerMixin):
    """
    Class to just train the model on the training data and test it on that same data. Sometimes referred to as a "Full fit"
    or a "Single fit", equivalent to just plotting y vs. x.

    Args:
        None (only object instance)

    Methods:
        get_n_splits: method to calculate the number of splits to perform

            Args:
                None

            Returns:
                (int), always 1 as only a single split is performed

        split: method to perform split into train indices and test indices

            Args:
                X: (numpy array), array of X features

            Returns:
                (numpy array), array of train and test indices (all data used as train and test for NoSplit)

    """
    def __init__(self):
        pass

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y, groups=None):
        indices = np.arange(X.shape[0])
        return [[indices, indices]]

class JustEachGroup(BaseEstimator, TransformerMixin):
    """
    Class to train the model on one group at a time and test it on the rest of the data
    This class wraps scikit-learn's LeavePGroupsOut with P set to n-1. More information is available at:
    http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePGroupsOut.html

    Args:
        None (only object instance)

    Methods:
        get_n_splits: method to calculate the number of splits to perform

            Args:
                groups: (numpy array), array of group labels

            Returns:
                (int), number of unique groups, indicating number of splits to perform

        split: method to perform split into train indices and test indices

            Args:
                X: (numpy array), array of X features
                y: (numpy array), array of y data
                groups: (numpy array), array of group labels

            Returns:
                (numpy array), array of train and test indices

    """
    def __init__(self):
        pass

    def get_n_splits(self, X=None, y=None, groups=None):
        return np.unique(groups).shape[0]

    def split(self, X, y, groups):
        n_groups = self.get_n_splits(groups=groups)
        #print('n_groups', n_groups)
        lpgo = ms.LeavePGroupsOut(n_groups=n_groups-1)
        return lpgo.split(X, y, groups)

#class WithoutElement(BaseEstimator, TransformerMixin):
#    " Train the model without each element, then test on the rows with that element "
#    pass

name_to_constructor = {
    # sklearn splitters:
    'GroupKFold': ms.GroupKFold,
    'GroupShuffleSplit': ms.GroupShuffleSplit,
    'KFold': ms.KFold,
    'LeaveOneGroupOut': ms.LeaveOneGroupOut,
    'LeavePGroupsOut': ms.LeavePGroupsOut,
    'LeaveOneOut': ms.LeaveOneOut,
    'LeavePOut': ms.LeavePOut,
    'PredefinedSplit': ms.PredefinedSplit,
    'RepeatedKFold': ms.RepeatedKFold, # NOTE: can use for repeated leave percent out / kfold
    'RepeatedStratifiedKFold': ms.RepeatedStratifiedKFold,
    'ShuffleSplit': ms.ShuffleSplit, # NOTE: like leave percent out
    'StratifiedKFold': ms.StratifiedKFold,
    'StratifiedShuffleSplit': ms.StratifiedShuffleSplit,
    'TimeSeriesSplit': ms.TimeSeriesSplit,

    # mastml splitters
    'NoSplit': NoSplit,
    'JustEachGroup': JustEachGroup,
    #'WithoutElement': WithoutElement,
}
