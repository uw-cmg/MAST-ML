'''
This module contains baseline test for models
'''
import os

import numpy as np
import pandas as pd
import scipy as sp
import mastml
from mastml.data_splitters import SklearnDataSplitter
from mastml.metrics import Metrics

data_path = os.path.join(mastml.__path__[0], 'data')


class Baseline_tests():
    '''
    Methods:
        test_mean: Compares the score of the model with a constant test value
            Args:
                X: (dataframe), dataframe of X features

                y: (dataframe), dataframe of y data

                metrics: (list), list of metric names to evaluate true vs. pred data in each split


        test_permuted: Compares the score of the model with a permuted test value
            Args:
                X: (dataframe), dataframe of X features

                y: (dataframe), dataframe of y data

                metrics: (list), list of metric names to evaluate true vs. pred data in each split

        test_mean: Compares the score of the model with the test value of the nearest neighbour
            Args:
                X: (dataframe), dataframe of X features

                y: (dataframe), dataframe of y data

                metrics: (list), list of metric names to evaluate true vs. pred data in each split

    '''

    def test_mean(self, X, y, model, metrics=["mean_absolute_error"]):
        splitter = SklearnDataSplitter(splitter='RepeatedKFold', n_repeats=1, n_splits=5)
        X_splits, y_splits, train_inds, test_inds = splitter.split_asframe(X, y)
        for Xs, ys, train_ind, test_ind in zip(X_splits, y_splits, train_inds, test_inds):
            X_train = Xs[0]
            X_test = Xs[1]
            y_train = pd.Series(np.array(ys[0]).ravel(), name='y_train')
            y_test = pd.Series(np.array(ys[1]).ravel(), name='y_test')

        # Let y mean of all the y-data. So, it is just like pretending the predicted value is a constant,
        # equal to the mean
        constant = y_test.mean()
        arr = [constant for i in range(len(y_test))]
        fake_test = pd.Series(arr)

        y_predict = model.predict(X_test)

        real_score = Metrics(metrics_list=metrics).evaluate(y_true=y_test, y_pred=y_predict)

        naive_score = Metrics(metrics_list=metrics).evaluate(y_true=fake_test, y_pred=y_predict)

        for (k, v), (k2, v2) in zip(real_score.items(), naive_score.items()):
            print(k, "score:")
            print("Real -", v)
            print("Fake -", v2, "\n")
        return

    def test_permuted(self, X, y, model, metrics=["mean_absolute_error"]):
        splitter = SklearnDataSplitter(splitter='RepeatedKFold', n_repeats=1, n_splits=5)
        X_splits, y_splits, train_inds, test_inds = splitter.split_asframe(X, y)
        for Xs, ys, train_ind, test_ind in zip(X_splits, y_splits, train_inds, test_inds):
            X_train = Xs[0]
            X_test = Xs[1]
            y_train = pd.Series(np.array(ys[0]).ravel(), name='y_train')
            y_test = pd.Series(np.array(ys[1]).ravel(), name='y_test')

        # Shuffling the y-data values to make it so that the X features do not correspond to the correct y data.
        fake_test = y_test.sample(frac=1)
        y_predict = model.predict(X_test)

        real_score = Metrics(metrics_list=metrics).evaluate(y_true=y_test, y_pred=y_predict)

        naive_score = Metrics(metrics_list=metrics).evaluate(y_true=fake_test, y_pred=y_predict)

        for (k, v), (k2, v2) in zip(real_score.items(), naive_score.items()):
            print(k, "score:")
            print("Real -", v)
            print("Fake -", v2, "\n")
        return

    def test_nearest_neighbour(self, X, y , model, metrics = ["mean_absolute_error"]):
        splitter = SklearnDataSplitter(splitter='RepeatedKFold', n_repeats=1, n_splits=5)
        X_splits, y_splits, train_inds, test_inds = splitter.split_asframe(X, y)
        for Xs, ys, train_ind, test_ind in zip(X_splits, y_splits, train_inds, test_inds):
            X_train = Xs[0]
            X_test = Xs[1]
            y_train = pd.Series(np.array(ys[0]).ravel(), name='y_train')
            y_test = pd.Series(np.array(ys[1]).ravel(), name='y_test')

        # Use the nearest neighbour datapoint's y_test instead of the actual y_test
        Xdatas = sp.spatial.cKDTree(X_train, leafsize=100)
        XresultDistance, XresultCoordinate = Xdatas.query(X_test)
        fake_test = []
        for i in XresultCoordinate:
            fake_test.append(y_train[i])
        fake_test = pd.DataFrame(fake_test)
        y_predict = model.predict(X_test)

        real_score = Metrics(metrics_list=metrics).evaluate(y_true=y_test, y_pred=y_predict)

        naive_score = Metrics(metrics_list=metrics).evaluate(y_true=fake_test, y_pred=y_predict)

        for (k,v), (k2,v2) in zip(real_score.items(), naive_score.items()):
            print(k , "score:")
            print("Real -", v)
            print("Fake -", v2, "\n")
        return
