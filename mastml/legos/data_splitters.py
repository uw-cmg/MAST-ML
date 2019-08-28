"""
The data_splitters module contains a collection of classes for generating (train_indices, test_indices) pairs from
a dataframe or a numpy array.

For more information and a list of scikit-learn splitter classes, see:
 http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
import sklearn.model_selection as ms
from matminer.featurizers.composition import ElementFraction
from pymatgen import Composition

from math import ceil
import warnings
from sklearn.utils import check_random_state

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

class LeaveCloseCompositionsOut(ms.BaseCrossValidator):
    """Leave-P-out where you exclude materials with compositions close to those the test set

    Computes the distance between the element fraction vectors. For example, the :math:`L_2`
    distance between Al and Cu is :math:`\sqrt{2}` and the :math:`L_1` distance between Al
    and Al0.9Cu0.1 is 0.2.

    Consequently, this splitter requires a list of compositions as the input to `split` rather
    than the features.

    Args:
        dist_threshold (float): Entries must be farther than this distance to be included in the
            training set
        nn_kwargs (dict): Keyword arguments for the scikit-learn NearestNeighbor class used
            to find nearest points
    """

    def __init__(self, dist_threshold=0.1, nn_kwargs=None):
        super(LeaveCloseCompositionsOut, self).__init__()
        if nn_kwargs is None:
            nn_kwargs = {}
        self.dist_threshold = dist_threshold
        self.nn_kwargs = nn_kwargs

    def split(self, X, y=None, groups=None):

        # Generate the composition vectors
        frac_computer = ElementFraction()
        elem_fracs = frac_computer.featurize_many(list(map(Composition, X)), pbar=False)

        # Generate the nearest-neighbor lookup tool
        neigh = NearestNeighbors(**self.nn_kwargs)
        neigh.fit(elem_fracs)

        # Generate a list of all entries
        all_inds = np.arange(0, len(X), 1)

        # Loop through each entry in X
        for i, x in enumerate(elem_fracs):

            # Get all the entries within the threshold distance of the test point
            too_close, = neigh.radius_neighbors([x], self.dist_threshold, return_distance=False)

            # Get the training set as "not these points"
            train_inds = np.setdiff1d(all_inds, too_close)

            yield train_inds, [i]

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(X)

class LeaveOutPercent(BaseEstimator, TransformerMixin):
    """
    Class to train the model using a certain percentage of data as training data

    Args:
        percent_leave_out (float): fraction of data to use in training (must be > 0 and < 1)

        n_repeats (int): number of repeated splits to perform (must be >= 1)

    Methods:
        get_n_splits: method to return the number of splits to perform

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
    def __init__(self, percent_leave_out=0.2, n_repeats=5):
        self.percent_leave_out = percent_leave_out
        self.n_repeats = n_repeats

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_repeats

    def split(self, X, y, groups=None):
        indices = range(X.shape[0])
        split = list()
        for i in range(self.n_repeats):
            trains, tests = ms.train_test_split(indices, test_size=self.percent_leave_out, random_state=np.random.randint(1, 1000), shuffle=True)
            split.append((trains, tests))
        return split

# Note: Bootstrap taken directly from sklearn Github (https://github.com/scikit-learn/scikit-learn/blob/0.11.X/sklearn/cross_validation.py)
# which was necessary as it was later removed from more recent sklearn releases
class Bootstrap(object):
    """Random sampling with replacement cross-validation iterator
    Provides train/test indices to split data in train test sets
    while resampling the input n_bootstraps times: each time a new
    random split of the data is performed and then samples are drawn
    (with replacement) on each side of the split to build the training
    and test sets.
    Note: contrary to other cross-validation strategies, bootstrapping
    will allow some samples to occur several times in each splits. However
    a sample that occurs in the train split will never occur in the test
    split and vice-versa.
    If you want each sample to occur at most once you should probably
    use ShuffleSplit cross validation instead.
    Parameters
    ----------
    n : int
        Total number of elements in the dataset.
    n_bootstraps : int (default is 3)
        Number of bootstrapping iterations
    train_size : int or float (default is 0.5)
        If int, number of samples to include in the training split
        (should be smaller than the total number of samples passed
        in the dataset).
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split.
    test_size : int or float or None (default is None)
        If int, number of samples to include in the training set
        (should be smaller than the total number of samples passed
        in the dataset).
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split.
        If None, n_test is set as the complement of n_train.
    random_state : int or RandomState
        Pseudo number generator state used for random sampling.
    Examples
    --------
    #>>> from sklearn import cross_validation
    #>>> bs = cross_validation.Bootstrap(9, random_state=0)
    #>>> len(bs)
    3
    #>>> print bs
    Bootstrap(9, n_bootstraps=3, train_size=5, test_size=4, random_state=0)
    #>>> for train_index, test_index in bs:
    ...    print "TRAIN:", train_index, "TEST:", test_index
    ...
    TRAIN: [1 8 7 7 8] TEST: [0 3 0 5]
    TRAIN: [5 4 2 4 2] TEST: [6 7 1 0]
    TRAIN: [4 7 0 1 1] TEST: [5 3 6 5]
    See also
    --------
    ShuffleSplit: cross validation using random permutations.
    """

    # Static marker to be able to introspect the CV type
    indices = True

    def __init__(self, n, n_bootstraps=3, train_size=.5, test_size=None,
                 n_train=None, n_test=None, random_state=0):
        self.n = n
        self.n_bootstraps = n_bootstraps
        if n_train is not None:
            train_size = n_train
            warnings.warn(
                "n_train is deprecated in 0.11 and scheduled for "
                "removal in 0.12, use train_size instead",
                DeprecationWarning, stacklevel=2)
        if n_test is not None:
            test_size = n_test
            warnings.warn(
                "n_test is deprecated in 0.11 and scheduled for "
                "removal in 0.12, use test_size instead",
                DeprecationWarning, stacklevel=2)
        if (isinstance(train_size, float) and train_size >= 0.0
                            and train_size <= 1.0):
            self.train_size = ceil(train_size * n)
        elif isinstance(train_size, int):
            self.train_size = train_size
        else:
            raise ValueError("Invalid value for train_size: %r" %
                             train_size)
        if self.train_size > n:
            raise ValueError("train_size=%d should not be larger than n=%d" %
                             (self.train_size, n))

        if (isinstance(test_size, float) and test_size >= 0.0
                    and test_size <= 1.0):
            self.test_size = ceil(test_size * n)
        elif isinstance(test_size, int):
            self.test_size = test_size
        elif test_size is None:
            self.test_size = self.n - self.train_size
        else:
            raise ValueError("Invalid value for test_size: %r" % test_size)
        if self.test_size > n:
            raise ValueError("test_size=%d should not be larger than n=%d" %
                             (self.test_size, n))

        self.random_state = random_state

    def __iter__(self):
        rng = check_random_state(self.random_state)
        for i in range(self.n_bootstraps):
            # random partition
            permutation = rng.permutation(self.n)
            ind_train = permutation[:self.train_size]
            ind_test = permutation[self.train_size:self.train_size
                                   + self.test_size]

            # bootstrap in each split individually
            train = rng.randint(0, self.train_size,
                                size=(self.train_size,))
            test = rng.randint(0, self.test_size,
                                size=(self.test_size,))
            yield ind_train[train], ind_test[test]

    def __repr__(self):
        return ('%s(%d, n_bootstraps=%d, train_size=%d, test_size=%d, '
                'random_state=%d)' % (
                    self.__class__.__name__,
                    self.n,
                    self.n_bootstraps,
                    self.train_size,
                    self.test_size,
                    self.random_state,
                ))

    def __len__(self):
        return self.n_bootstraps

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.__len__()

    def split(self, X, y, groups=None):
        indices = range(X.shape[0])
        split = list()
        for trains, tests in self:
            split.append((trains.tolist(), tests.tolist()))
        return split

name_to_constructor = {
    # sklearn splitters:
    'Bootstrap': Bootstrap,
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
    'LeaveCloseCompositionsOut': LeaveCloseCompositionsOut,
    'LeaveOutPercent': LeaveOutPercent,
    #'WithoutElement': WithoutElement,
}
