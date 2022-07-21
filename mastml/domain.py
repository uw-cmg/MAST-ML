from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.spatial.distance import cdist
from mastml.mastml import parallel
from pathlib import Path

import pandas as pd
import numpy as np
import os


class Domain:
    '''
    Calculate dissimilarities between splits of data.
    '''

    def __init__(self, path, parallel_run=True):
        paths = list(map(str, Path(path).rglob('*X_train*')))

        # Calculation of dissimilarity between train data and others
        def calculate(path):

            splits = path.split('/')
            path = '/'.join(splits[:-1])

            X_train = pd.read_csv(path+'/X_train.csv')
            y_train = pd.read_csv(path+'/y_train.csv')
            X_test = pd.read_csv(path+'/X_test.csv')

            # Cannot measure non-numeric data (may consider one-hot encoding)
            train = X_train
            train['target'] = y_train
            train.drop_duplicates(inplace=True)

            X_train = train.loc[:, train.columns != 'target']
            y_train = train.target

            X_train = X_train._get_numeric_data()
            y_train = y_train.values
            X_test = X_test._get_numeric_data()
            X_test.drop_duplicates(inplace=True)

            domain = domain_split()
            domain.train(X_train, y_train)

            dist_train_to_train = domain.predict(X_train)
            dist_train_to_train = pd.DataFrame(dist_train_to_train)

            out_name = os.path.join(path, 'dist_train_to_train.csv')
            dist_train_to_train.to_csv(out_name, index=False)

            dist_train_to_test = domain.predict(X_test)
            dist_train_to_test = pd.DataFrame(dist_train_to_test)

            out_name = os.path.join(path, 'dist_train_to_test.csv')
            dist_train_to_test.to_csv(out_name, index=False)

            leave_out_file = path+'/X_leaveout.csv'
            if os.path.exists(leave_out_file):

                X_leaveout = pd.read_csv(leave_out_file)
                X_leaveout = X_leaveout._get_numeric_data()
                X_leaveout.drop_duplicates(inplace=True)

                dist_train_to_leaveout = domain.predict(X_leaveout)
                dist_train_to_leaveout = pd.DataFrame(dist_train_to_leaveout)

                out_name = os.path.join(path, 'dist_train_to_leaveout.csv')
                dist_train_to_leaveout.to_csv(out_name, index=False)

        if parallel_run is True:
            parallel(calculate, paths)
        else:
            [calculate(i) for i in paths]


class domain_split:
    '''
    Trainable domain object that can calculate dissimilarities between
    sets of data.
    '''

    def train(self, X, y=None):
        self.dist_func = lambda x: distance(X, x, y)

    def predict(self, X):
        return self.dist_func(X)


def distance_link(
                  X_train,
                  X_test,
                  dist_type,
                  y_train=None,
                  ):
    '''
    Get the distances based on a metric.

    inputs:
        X_train = The features of the training set.
        X_test = The features of the test set.
        dist = The distance to consider.
        y_train = The training target when applicable.
    ouputs:
        dists = A dictionary of distances.
    '''

    dists = {}

    if dist_type == 'gpr_std':

        model = GaussianProcessRegressor()
        model.fit(X_train, y_train)
        _, dist = model.predict(X_test, return_std=True)
        dists[dist_type] = dist

    else:
        dist = cdist(X_train, X_test, dist_type)
        dists[dist_type] = np.mean(dist, axis=0)

    return dists


def distance(
             X_train,
             X_test,
             y_train=None,
             ):
    '''
    Determine the distance from set X_test to set X_train.
    '''

    # List of supported distances
    distance_list = [
                     'gpr_std',
                     'euclidean',
                     'mahalanobis',
                     ]

    dists = {}
    for distance in distance_list:

        # Compute regular distances
        dists.update(distance_link(
                                   X_train,
                                   X_test,
                                   distance,
                                   y_train=y_train,
                                   ))

    return dists
