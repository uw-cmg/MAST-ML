import unittest
import pandas as pd
import numpy as np
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath('../../../'))

from mastml.data_splitters import NoSplit, SklearnDataSplitter, LeaveCloseCompositionsOut, LeaveOutPercent, \
    Bootstrap, JustEachGroup
from mastml.models import SklearnModel

class TestSplitters(unittest.TestCase):

    def test_nosplit(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(10, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(10,)))
        model = SklearnModel(model='LinearRegression')
        splitter = NoSplit()
        splitter.evaluate(X=X, y=y, models=[model], savepath=os.getcwd())
        for d in splitter.splitdirs:
            self.assertTrue(os.path.exists(d))
            shutil.rmtree(d)
        return

    def test_sklearnsplitter(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(10, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(10,)))
        model = SklearnModel(model='LinearRegression')
        splitter = SklearnDataSplitter(splitter='KFold', shuffle=True, n_splits=5)
        splitter.evaluate(X=X, y=y, models=[model], savepath=os.getcwd())
        for d in splitter.splitdirs:
            self.assertTrue(os.path.exists(d))
            shutil.rmtree(d)
        return

    def test_close_comps(self):
        # Make entries at a 10% spacing
        X = ['Al{}Cu{}'.format(i, 10-i) for i in range(11)]

        # Generate test splits with a 5% distance cutoff
        splitter = LeaveCloseCompositionsOut(dist_threshold=0.05)
        train_inds, test_inds = zip(*splitter.split(X))
        self.assertEqual(train_inds[0].tolist(), list(range(1, 11)))  # Everything but 0
        self.assertEqual(list(test_inds), [[i] for i in range(11)])  # Only one point

        # Generate test splits with 25% distance cutoff
        splitter.dist_threshold = 0.25
        splitter.nn_kwargs = {'metric': 'l1'}
        train_inds, test_inds = zip(*splitter.split(X))
        self.assertEqual(train_inds[0].tolist(), list(range(2, 11)))  # 1 is too close
        self.assertEqual(train_inds[1].tolist(), list(range(3, 11)))  # 0 and 2 are too close
        return

    def test_leaveoutpercent(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(25, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(25,)))
        splitter = LeaveOutPercent(percent_leave_out=0.20, n_repeats=5)
        model = SklearnModel(model='LinearRegression')
        splitter.evaluate(X=X, y=y, models=[model], groups=None)
        for d in splitter.splitdirs:
            self.assertTrue(os.path.exists(d))
            shutil.rmtree(d)
        return

    def test_bootstrap(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(25, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(25,)))
        splitter = Bootstrap(n=25, n_bootstraps=3, train_size=0.5)
        model = SklearnModel(model='LinearRegression')
        splitter.evaluate(X=X, y=y, models=[model], groups=None)
        for d in splitter.splitdirs:
            self.assertTrue(os.path.exists(d))
            shutil.rmtree(d)
        return

    def test_justeachgroup(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(5,10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(5,)))
        groups = pd.DataFrame.from_dict({'groups':[0, 1, 1, 0, 1]})
        X = pd.concat([X, groups], axis=1)
        splitter = JustEachGroup()
        model = SklearnModel(model='LinearRegression')
        splitter.evaluate(X=X, y=y, models=[model], groups=X['groups'])
        for d in splitter.splitdirs:
            self.assertTrue(os.path.exists(d))
            shutil.rmtree(d)
        return

if __name__=='__main__':
    unittest.main()