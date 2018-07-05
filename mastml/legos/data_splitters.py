"""
A collection of classes for generating (train_indices, test_indices) pairs from
a dataframe or a numpy array.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.model_selection as ms

# List of sklearn splitter classes:
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

class SplittersUnion(BaseEstimator, TransformerMixin):
    " Takes a list of splitters and creates a splitter which is their union "
    def __init__(self, splitters):
        self.splitters = splitters
    def get_n_splits(self, X, y, groups=None):
        return sum(splitter.get_n_splits(X, y, groups) for splitter in self.splitters)
    def split(self, X, y, groups=None):
        for splitter in self.splitters:
            yield from splitter.split(X, y, groups)

class NoSplit(BaseEstimator, TransformerMixin):
    " Just train the model on the training data and test it on that same data "
    def __init__(self):
        pass

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y, groups=None):
        indices = np.arange(X.shape[0])
        return [[indices, indices]]

class JustEachGroup(BaseEstimator, TransformerMixin):
    """
    Train the model on one group at a time and test it on the rest of the data
    This class wraps "LeavePGroupsOut with P set to n-1.
    """

    def __init__(self):
        pass

    def get_n_splits(self, X=None, y=None, groups=None):
        return np.unique(groups).shape[0]

    def split(self, X, y, groups):
        n_groups = self.get_n_splits(groups=groups)
        print('n_groups', n_groups)
        lpgo = LeavePGroupsOut(n_groups=n_groups-1)
        return lpgo.split(X, y, groups)

class WithoutElement(BaseEstimator, TransformerMixin):
    " Train the model without each element, then test on the rows with that element "
    pass

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
    'WithoutElement': WithoutElement,
}
