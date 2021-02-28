from mastml.models import SklearnModel
from mastml.data_splitters import LeaveOutTwinCV, NoSplit, SklearnDataSplitter, LeaveCloseCompositionsOut, LeaveOutPercent, \
    Bootstrap, JustEachGroup
import unittest
import pandas as pd
import numpy as np
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath('../../../'))


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
        composition_df = pd.DataFrame({'composition': ['Al{}Cu{}'.format(i, 10-i) for i in range(11)]})
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(11, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(11,)))

        # Generate test splits with a 5% distance cutoff
        splitter = LeaveCloseCompositionsOut(composition_df=composition_df, dist_threshold=0.5)

        model = SklearnModel(model='LinearRegression')
        splitter.evaluate(X=X, y=y, models=[model])
        for d in splitter.splitdirs:
            self.assertTrue(os.path.exists(d))
            shutil.rmtree(d)
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
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(5, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(5,)))
        groups = pd.DataFrame.from_dict({'groups': [0, 1, 1, 0, 1]})
        X = pd.concat([X, groups], axis=1)
        splitter = JustEachGroup()
        model = SklearnModel(model='LinearRegression')
        splitter.evaluate(X=X, y=y, models=[model], groups=X['groups'])
        for d in splitter.splitdirs:
            self.assertTrue(os.path.exists(d))
            shutil.rmtree(d)
        return

    def test_leaveoutwincv(self):

        implemented = False
        log = "\n"

        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(25, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(25,)))
        splitter = LeaveOutTwinCV(threshold=1)
        model = SklearnModel(model='LinearRegression')
        splitter.evaluate(X=X, y=y, models=[model], groups=None)
        # Assert that output exists
        for d in splitter.splitdirs:
            self.assertTrue(os.path.exists(d))
            shutil.rmtree(d)

        # CASE all unique (each value different by 1, so rows /should/ be different - although if they even out it might not work)
        n_datapoints = 25
        n_features = 5
        X = pd.DataFrame(np.random.choice(range(-n_features*n_datapoints, n_features*n_datapoints), size=(n_datapoints, n_features), replace=False))
        y = X[n_features-1]
        X.drop(columns=n_features-1, inplace=True)
        ret = splitter.split(X, y)
        log += f"\n"
        for r in ret:
            log += f"\n{r[0]} | {r[1]}"

        # CASE every datapoint is an exact twin, twins in both X and y

        X = pd.DataFrame(np.random.choice(range(-n_features*n_datapoints, n_features*n_datapoints), size=(n_datapoints, n_features), replace=False))
        y = X[n_features-1]
        X.drop(columns=n_features-1, inplace=True)
        X = X.append(X.copy(), ignore_index=True)
        y = y.append(y.copy(), ignore_index=True)
        ret = splitter.split(X, y)
        log += f"\n"
        for r in ret:
            log += f"\n{r[0]} | {r[1]}"

        # CASE quadruplets
        n_datapoints = 10

        X = pd.DataFrame(np.random.choice(range(-n_features*n_datapoints, n_features*n_datapoints), size=(n_datapoints, n_features), replace=False))
        y = X[n_features-1]
        X.drop(columns=n_features-1, inplace=True)
        X = X.append(X.copy(), ignore_index=True)
        X = X.append(X.copy(), ignore_index=True)
        y = y.append(y.copy(), ignore_index=True)
        y = y.append(y.copy(), ignore_index=True)
        ret = splitter.split(X, y)
        log += f"\n"
        for r in ret:
            log += f"\n{r[0]} | {r[1]}"

        # TODO test with groups?

        # CASE 1/5 of data is exact twin

        # n_datapoints = 40
        # X = pd.DataFrame(np.random.choice(range(-n_features*n_datapoints, n_features*n_datapoints), size=(n_datapoints, n_features), replace=False))
        # y = X[n_features-1]
        # X = X.append(X.head(10), ignore_index=True)
        # y = y.append(y.head(10), ignore_index=True)

        # CASE one twin exists

        # CASE data spread to threshold x - 1/2 exist within threshold y (test exact, then different threshold)

        # CASE random data (relatively spread out) + added noise = likely data twins within (threshold)

        n_datapoints = 20
        n_features = 5

        threshold = 10
        X = pd.DataFrame(np.random.choice(range(-n_features*n_datapoints, n_features*n_datapoints), size=(n_datapoints, n_features), replace=False))
        y = X[n_features-1]
        X = X.append(X.head(10), ignore_index=True)
        y = y.append(y.head(10), ignore_index=True)

        df = X.head(10)
        df[0] = df[0] + threshold

        ret = splitter.split(X, y)
        log += f"\n"
        for r in ret:
            log += f"\n{r[0]} | {r[1]}"

        # unique = pd.DataFrame(np.random.choice(range(0, n_features*n_datapoints*threshold, threshold), size=(n_datapoints,n_features), replace=False))
        # noise = pd.DataFrame(np.random.uniform(low=-threshold//2, high=threshold//2, size=(n_datapoints,n_features)))
        # print(unique)
        # print(unique + noise)

        # CASE you want to allow twins in training (always get removed from test)

        # CASE don't allow twins in training (always get removed from test)

        # splitter = LeaveOutTwinCV(threshold=1, )
        # for s in splitter.split(X, y, ):
        #     print(f"{s[0]} | {s[1]}\n")

        self.assertTrue(implemented, msg=log)

        return

    def create_test_suite():
        suite = unittest.TestSuite()
        suite.addTest(TestSplitters('test_leaveoutwincv'))
        return suite


if __name__ == '__main__':
    unittest.main()
