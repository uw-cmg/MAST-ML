import unittest
import pandas as pd
import numpy as np
import os
import shutil
import sys
import sklearn.datasets as sk

sys.path.insert(0, os.path.abspath('../../../'))

from mastml.models import SklearnModel
from mastml.data_splitters import NoSplit, SklearnDataSplitter, LeaveCloseCompositionsOut, LeaveOutPercent, \
    Bootstrap, JustEachGroup, LeaveOutTwinCV, LeaveOutClusterCV

parallel_run = False  # Condition to run in parallel


class TestSplitters(unittest.TestCase):

    def test_nosplit(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(10, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(10,)))
        model = SklearnModel(model='LinearRegression')
        splitter = NoSplit(parallel_run=parallel_run)
        splitter.evaluate(X=X, y=y, models=[model], savepath=os.getcwd(), plots=list())
        for d in splitter.splitdirs:
            self.assertTrue(os.path.exists(d))
            shutil.rmtree(d)
        return

    def test_sklearnsplitter(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(10, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(10,)))
        model = SklearnModel(model='LinearRegression')
        splitter = SklearnDataSplitter(splitter='RepeatedKFold', n_repeats=10, n_splits=5, parallel_run=parallel_run)
        splitter.evaluate(X=X, y=y, models=[model], savepath=os.getcwd(), plots=list())
        for d in splitter.splitdirs:
            self.assertTrue(os.path.exists(d))
            shutil.rmtree(d)
        return

    def test_close_comps(self):
        # Make entries at a 10% spacing
        composition_df = pd.DataFrame({'composition': ['Al{}Cu{}'.format(i, 10-i) for i in range(11)]})
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(11, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(11,)))

        # Generate test splits with a 5% distance cutoff
        splitter = LeaveCloseCompositionsOut(composition_df=composition_df, dist_threshold=0.5, parallel_run=parallel_run)

        model = SklearnModel(model='LinearRegression')
        splitter.evaluate(X=X, y=y, models=[model], savepath=os.getcwd(), plots=list())
        for d in splitter.splitdirs:
            self.assertTrue(os.path.exists(d))
            shutil.rmtree(d)
        return

    def test_leaveoutpercent(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(25, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(25,)))
        splitter = LeaveOutPercent(percent_leave_out=0.20, n_repeats=5, parallel_run=parallel_run)
        model = SklearnModel(model='LinearRegression')
        splitter.evaluate(X=X, y=y, models=[model], groups=None, savepath=os.getcwd(), plots=list())
        for d in splitter.splitdirs:
            self.assertTrue(os.path.exists(d))
            shutil.rmtree(d)
        return

    def test_bootstrap(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(25, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(25,)))
        splitter = Bootstrap(n=25, n_bootstraps=3, train_size=0.5, parallel_run=parallel_run)
        model = SklearnModel(model='LinearRegression')
        splitter.evaluate(X=X, y=y, models=[model], groups=None, savepath=os.getcwd(), plots=list())
        for d in splitter.splitdirs:
            self.assertTrue(os.path.exists(d))
            shutil.rmtree(d)
        return

    def test_justeachgroup(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(5, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(5,)))
        groups = pd.DataFrame.from_dict({'groups': [0, 1, 1, 0, 1]})
        X = pd.concat([X, groups], axis=1)
        splitter = JustEachGroup(parallel_run=parallel_run)
        model = SklearnModel(model='LinearRegression')
        splitter.evaluate(X=X, y=y, models=[model], groups=X['groups'], savepath=os.getcwd(), plots=list())
        for d in splitter.splitdirs:
            self.assertTrue(os.path.exists(d))
            shutil.rmtree(d)
        return

    def test_leaveoutcluster(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(5, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(5,)))
        splitter = LeaveOutClusterCV(cluster='KMeans', n_clusters=5, parallel_run=parallel_run)
        model = SklearnModel(model='LinearRegression')
        splitter.evaluate(X=X, y=y, models=[model], savepath=os.getcwd(), plots=list())
        for d in splitter.splitdirs:
            self.assertTrue(os.path.exists(d))
            shutil.rmtree(d)
        return

    def test_leaveouttwins(self):
        # SAME TEST AS OTHER SPLITTERS
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(25, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(25,)))
        splitter = LeaveOutTwinCV(threshold=0, auto_threshold=True, parallel_run=parallel_run)
        model = SklearnModel(model='LinearRegression')
        splitter.evaluate(X=X, y=y, models=[model], groups=None, savepath=os.getcwd(), plots=list())
        for d in splitter.splitdirs:
            self.assertTrue(os.path.exists(d))
            shutil.rmtree(d)

        # Setup to check exact twins
        splitter = LeaveOutTwinCV(threshold=0, auto_threshold=True)
        model = SklearnModel(model='LinearRegression')
        n_datapoints = 25
        n_features = 5

        # CASE 1: every datapoint is an exact twin, twins in both X and y
        X = pd.DataFrame(np.random.choice(range(-n_features*n_datapoints, n_features*n_datapoints), size=(n_datapoints, n_features), replace=False))  # This generates random numbers without repetitions
        # Pull out last row to be y
        y = X[n_features-1]
        X.drop(columns=n_features-1, inplace=True)
        # Create duplicates
        X = X.append(X.copy(), ignore_index=True)
        y = y.append(y.copy(), ignore_index=True)
        ret = splitter.split(X, y)
        numTwins = n_datapoints * 2
        for r in ret:
            self.assertTrue(len(r[0]) == numTwins or len(r[1]) == numTwins)  # check that everything was counted as a twin in each split

        # CASE 2: half datapoint is an exact twin, twins in both X and y
        X = pd.DataFrame(np.random.choice(range(-n_features*n_datapoints, n_features*n_datapoints), size=(n_datapoints, n_features), replace=False))
        y = X[n_features-1]
        X.drop(columns=n_features-1, inplace=True)
        X = X.append(X[0:int(n_datapoints/2)].copy(), ignore_index=True)
        y = y.append(y[0:int(n_datapoints/2)].copy(), ignore_index=True)
        ret = splitter.split(X, y)
        numTwins = int(n_datapoints/2)*2
        for r in ret:
            self.assertTrue(len(r[0]) == numTwins or len(r[1]) == numTwins)  # check correct number of twins

        # Setup to check twins generated by sci kit learn blobs
        splitter = LeaveOutTwinCV(threshold=2, debug=False, ceiling=1, auto_threshold=True)
        model = SklearnModel(model='LinearRegression')
        n_datapoints = 25
        n_features = 5

        cen = []
        for i in range(3):
            for j in range(3):
                cen.append((i*10, j*10))

        v = sk.make_blobs(n_samples=100, n_features=2, centers=cen, cluster_std=.1, shuffle=True, random_state=None)
        X = pd.DataFrame(v[0])
        y = pd.DataFrame(np.random.choice(range(-500, 500), size=(100, 1), replace=False))
        ret = splitter.split(X, y)
        numTwins = 100
        for r in ret:
            self.assertTrue(len(r[0]) == numTwins or len(r[1]) == numTwins)

        # plt.scatter(v[0][:, 0], v[0][:, 1], c=v[1])
        # plt.savefig("./test.png")
        # self.assertTrue(False, msg="In Progress")
        return


if __name__ == '__main__':
    unittest.main()
