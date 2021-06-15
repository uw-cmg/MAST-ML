'''
This module contains baseline test for models
'''
from mastml.datasets import LocalDatasets
from mastml.models import SklearnModel, EnsembleModel
from mastml.preprocessing import SklearnPreprocessor
from mastml.data_splitters import SklearnDataSplitter, NoSplit, LeaveOutPercent
from mastml.metrics import Metrics
import mastml
import os
import pandas as pd
import numpy as np
data_path = os.path.join(mastml.__path__[0], 'data')



class Baseline_tests():
    '''
    Methods:
        test_mean: Compares the score of the model with a constant test value
            Args:
                X: (dataframe), dataframe of X features

                y: (dataframe), dataframe of y data

                metrics: (list), list of metric names to evaluate true vs. pred data in each split



    '''
    def test_mean(self, X,y, model, metrics =["mean_absolute_error"]):
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

        print("Real Score" , real_score)
        print("Naive Score" , naive_score)
        return

    def test_permuted(self):
        return

    def test_all(self):
        return



