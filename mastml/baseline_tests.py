'''
This module contains baseline test for models
'''
import os

import numpy as np
import pandas as pd
import scipy as sp
from scipy.spatial.distance import cdist
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

        test_nearest_neighbour_kdTree: Compares the score of the model with the test value of the nearest neighbour found using kdTree
            Args:
                X: (dataframe), dataframe of X features

                y: (dataframe), dataframe of y data

                metrics: (list), list of metric names to evaluate true vs. pred data in each split

        test_nearest_neighbour_cdist: Compares the score of the model with the test value of the nearest neighbour found using cdist
            Args:
                X: (dataframe), dataframe of X features

                y: (dataframe), dataframe of y data

                metrics: (list), list of metric names to evaluate true vs. pred data in each split

                d_metric: Metric to use to calculate the distance. Default is euclidean

        test_classifier_random: Compares the score of the model with a test value of a random class
            Args:
                X: (dataframe), dataframe of X features

                y: (dataframe), dataframe of y data

                metrics: (list), list of metric names to evaluate true vs. pred data in each split

        test_classifier_dominant: Compares the score of the model with a test value of the dominant class (ie highest count)
            Args:
                X: (dataframe), dataframe of X features

                y: (dataframe), dataframe of y data

                metrics: (list), list of metric names to evaluate true vs. pred data in each split


        print_results: Prints the comparison between the naive score and the real score
            Args:
                real_score: The actual score of the model

                naive_score: The naive score of the model tested with fake_test

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
        self.print_results(real_score,naive_score)

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
        self.print_results(real_score,naive_score)

        return

    def test_nearest_neighbour_kdtree(self, X, y , model, metrics = ["mean_absolute_error"]):
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
        self.print_results(real_score,naive_score)

        return

    def test_nearest_neighbour_cdist(self, X, y , model, metrics = ["mean_absolute_error"], d_metric = "euclidean"):
        splitter = SklearnDataSplitter(splitter='RepeatedKFold', n_repeats=1, n_splits=5)
        X_splits, y_splits, train_inds, test_inds = splitter.split_asframe(X, y)
        for Xs, ys, train_ind, test_ind in zip(X_splits, y_splits, train_inds, test_inds):
            X_train = Xs[0]
            X_test = Xs[1]
            y_train = pd.Series(np.array(ys[0]).ravel(), name='y_train')
            y_test = pd.Series(np.array(ys[1]).ravel(), name='y_test')

        result = cdist(X_test, X_train, metric = d_metric)
        fake_test = []

        #Get the index of the nearest neighbour (i.e shortest distance) from the result
        for i in range(len(X_test)):
            nn_index = result[i].tolist().index(result[i].min())
            # Use the nearest neighbour datapoint's y_test instead of the actual y_test
            fake_test.append(y_train[nn_index])

        y_predict = model.predict(X_test)

        real_score = Metrics(metrics_list=metrics).evaluate(y_true=y_test, y_pred=y_predict)
        naive_score = Metrics(metrics_list=metrics).evaluate(y_true=fake_test, y_pred=y_predict)
        self.print_results(real_score,naive_score)

        return

    def test_classifier_random(self, X, y , model, metrics = ["mean_absolute_error"]):
        splitter = SklearnDataSplitter(splitter='RepeatedKFold', n_repeats=1, n_splits=5)
        X_splits, y_splits, train_inds, test_inds = splitter.split_asframe(X, y)
        for Xs, ys, train_ind, test_ind in zip(X_splits, y_splits, train_inds, test_inds):
            X_train = Xs[0]
            X_test = Xs[1]
            y_train = pd.Series(np.array(ys[0]).ravel(), name='y_train')
            y_test = pd.Series(np.array(ys[1]).ravel(), name='y_test')

        # Get the number of classes in the data and randomly pick one
        n_classes = np.unique(y).size
        constant = np.random.randint(0, n_classes)
        arr = [constant for i in range(len(y_test))]

        fake_test = pd.Series(arr)
        y_predict = model.predict(X_test)

        real_score = Metrics(metrics_list=metrics).evaluate(y_true=y_test, y_pred=y_predict)
        naive_score = Metrics(metrics_list=metrics).evaluate(y_true=fake_test, y_pred=y_predict)
        self.print_results(real_score,naive_score)

    def test_classifier_dominant(self, X, y, model, metrics=["mean_absolute_error"]):
        splitter = SklearnDataSplitter(splitter='RepeatedKFold', n_repeats=1, n_splits=5)
        X_splits, y_splits, train_inds, test_inds = splitter.split_asframe(X, y)
        for Xs, ys, train_ind, test_ind in zip(X_splits, y_splits, train_inds, test_inds):
            X_train = Xs[0]
            X_test = Xs[1]
            y_train = pd.Series(np.array(ys[0]).ravel(), name='y_train')
            y_test = pd.Series(np.array(ys[1]).ravel(), name='y_test')

        # Choose the class with the highest number of count
        # If there are classes with equal count, the first one will be chosen
        k = y.value_counts()
        theMax = k.max()
        dominant_index = k.tolist().index(theMax)
        constant = np.random.randint(k[dominant_index])
        arr = [constant for i in range(len(y_test))]

        fake_test = pd.Series(arr)
        y_predict = model.predict(X_test)

        real_score = Metrics(metrics_list=metrics).evaluate(y_true=y_test, y_pred=y_predict)
        naive_score = Metrics(metrics_list=metrics).evaluate(y_true=fake_test, y_pred=y_predict)
        self.print_results(real_score, naive_score)


    def print_results(self, real_score, naive_score):
        for (k,v), (k2,v2) in zip(real_score.items(), naive_score.items()):
            print(k , "score:")
            print("Real:", v)
            print("Fake:", v2, "\n")