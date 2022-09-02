from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.spatial.distance import cdist

import numpy as np


def distance_link(
                  X_train,
                  X_test,
                  dist_type,
                  y_train=None,
                  y_test=None
                  ):
    '''
    Get the distances based on a metric.
    inputs:
        X_train = The features of the training set.
        X_test = The features of the test set.
        dist = The distance to consider.
        y_train = The training target when applicable.
        y_test = The testing target when applicable.
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


def distance(X_train, X_test, y_train=None, y_test=None):
    '''
    Determine the distance from set X_test to set X_train.
    '''
    # For development
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
                                   y_test=y_test
                                   ))

    return dists
