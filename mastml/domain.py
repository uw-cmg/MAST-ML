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
        paths = 'model_errors_*_calibrated.csv'
        paths = list(map(str, Path(path).rglob(paths)))

        # Calculation of dissimilarity between train data and others
        def calculate(path):

            splits = path.split('/')
            path = '/'.join(splits[:-1])

            for i in ['train', 'test', 'leaveout']:

                # File names
                X_train = 'X_train.csv'
                y_train = 'y_train.csv'
                stdcal = 'model_errors_{}_calibrated.csv'.format(i)
                X = 'X_{}.csv'.format(i)
                y = 'y_{}.csv'.format(i)
                y_pred = 'y_pred_{}.csv'.format(i)

                # File paths
                X_train = os.path.join(path, X_train)
                y_train = os.path.join(path, y_train)
                stdcal = os.path.join(path, stdcal)
                X = os.path.join(path, X)
                y = os.path.join(path, y)
                y_pred = os.path.join(path, y_pred)

                # If required files exist
                if all([
                        os.path.exists(y),
                        os.path.exists(y_pred),
                        os.path.exists(stdcal),
                        os.path.exists(X),
                        os.path.exists(y_train),
                        os.path.exists(X_train),
                        ]):

                    X = pd.read_csv(X)
                    X_train = pd.read_csv(X_train)
                    y = pd.read_csv(y)
                    y.columns = ['y']
                    y_train = pd.read_csv(y_train)
                    y_train.columns = ['y_train']
                    y_pred = pd.read_csv(y_pred)
                    y_pred.columns = ['y_pred']
                    stdcal = pd.read_csv(stdcal)
                    stdcal.columns = ['stdcal']

                    domain = domain_split()
                    domain.train(X_train, y_train)

                    dist = domain.predict(X)
                    dist = pd.DataFrame(dist)

                    df = pd.concat([dist, stdcal, y, y_pred], axis=1)
                    out_name = 'dist_train_to_{}.csv'.format(i)
                    out_name = os.path.join(path, out_name)
                    df.to_csv(out_name, index=False)

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
