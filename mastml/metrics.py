"""
This module contains a metrics class for construction and evaluation of various regression score metrics between
true and model predicted data.

Metrics:
    Class to construct and evaluate a list of regression metrics of interest. The full list of available metrics
    can be obtained from Metrics()._metric_zoo()

"""

import numpy as np
import sklearn.metrics as sm
from sklearn.linear_model import LinearRegression

class Metrics():
    """
    Class containing access to a wide range of metrics from scikit-learn and a number of MAST-ML custom-written metrics

    Args:
        metrics_list: (list), list of strings of metric names to use

        metrics_type: (str), one of 'regression' or 'classification': whether to use set of common regression/classifier metrics

    Methods:
        evaluate: main method to evaluate the specified metrics and the provided true and pred data
            Args:
                y_true: (pd.Series), series of true y data

                y_pred: (pd.Series), series of predicted y data

            Returns:
                stats_dict: (dict), dictionary of calculated statistics for each metric

        _get_metrics: builds the metrics dict of metric names : metric instances based on the metrics specified in metrics_list
            Args:
                None

            Returns:
                None

        _metric_zoo: method to retrieve full dict of metric names : metric instance pairs
            Args:
                None

            Returns:
                all_metrics (dict), dictionary of all metric names and instances

    """
    def __init__(self, metrics_list, metrics_type='regression'):
        self.metrics_list = metrics_list
        self.metrics_type = metrics_type
        self.metrics_dict = dict()
        if self.metrics_type not in ['regression', 'classification']:
            raise AttributeError('metrics_type must be one of "regression" or "classification"')

    def evaluate(self, y_true, y_pred):
        # Evaluate all of the metrics between provided y_true and y_pred data
        stats_dict = dict()
        self._get_metrics()
        for metric_name, metric in self.metrics_dict.items():
            stats_dict[metric_name] = metric(y_true, y_pred)
        return stats_dict

    def _get_metrics(self):
        # Take the metric names and get the metric functions
        all_metrics = self._metric_zoo()
        for metric_name in self.metrics_list:
            if metric_name in all_metrics.keys():
                self.metrics_dict[metric_name] = all_metrics[metric_name][1]
            else:
                raise NameError(metric_name, 'is not a valid metric name')
        return

    def _metric_zoo(self):
        if self.metrics_type == 'regression':
            all_metrics =  {'explained_variance':     (True, sm.explained_variance_score),
                            'mean_absolute_error':    (False, sm.mean_absolute_error),
                            'mean_squared_error':     (False, sm.mean_squared_error),
                            'mean_squared_log_error': (False, sm.mean_squared_log_error),
                            'median_absolute_error':  (False, sm.median_absolute_error),
                            'r2_score': (True, sm.r2_score),
                            'r2_score_noint' : (True, r2_score_noint),
                            'r2_score_fitted' : (True, r2_score_fitted),
                            'r2_score_adjusted' : (True, r2_score_adjusted),
                            'root_mean_squared_error' : (False, root_mean_squared_error),
                            'rmse_over_stdev' : (False, rmse_over_stdev)
                            }
        elif self.metrics_type == 'classification':
            print('WARNING: new version of MAST-ML has not yet been reconfigured to handle classification tasks')
            exit()
            all_metrics = {'accuracy':           (True, sm.accuracy_score),
                            'f1_binary':          (True, lambda yt, yp: sm.f1_score(yt, yp, average='binary')),
                            'f1_macro':           (True, lambda yt, yp: sm.f1_score(yt, yp, average='macro')),
                            'f1_micro':           (True, lambda yt, yp: sm.f1_score(yt, yp, average='micro')),
                            'f1_samples':         (True, lambda yt, yp: sm.f1_score(yt, yp, average='samples')),
                            'f1_weighted':        (True, lambda yt, yp: sm.f1_score(yt, yp, average='weighted')),
                            'log_loss':           (False, sm.log_loss),
                            'precision_binary':   (True, lambda yt, yp: sm.precision_score(yt, yp, average='binary')),
                            'precision_macro':    (True, lambda yt, yp: sm.precision_score(yt, yp, average='macro')),
                            'precision_micro':    (True, lambda yt, yp: sm.precision_score(yt, yp, average='micro')),
                            'precision_samples':  (True, lambda yt, yp: sm.precision_score(yt, yp, average='samples')),
                            'precision_weighted': (True, lambda yt, yp: sm.precision_score(yt, yp, average='weighted')),
                            'recall_binary':      (True, lambda yt, yp: sm.recall_score(yt, yp, average='binary')),
                            'recall_macro':       (True, lambda yt, yp: sm.recall_score(yt, yp, average='macro')),
                            'recall_micro':       (True, lambda yt, yp: sm.recall_score(yt, yp, average='micro')),
                            'recall_samples':     (True, lambda yt, yp: sm.recall_score(yt, yp, average='samples')),
                            'recall_weighted':    (True, lambda yt, yp: sm.recall_score(yt, yp, average='weighted')),
                            'roc_auc':            (True, sm.roc_auc_score),
                        }
        return all_metrics

def r2_score_noint(y_true, y_pred):
    """
    Method that calculates the R^2 value without fitting the y-intercept

    Args:
        y_true: (numpy array), array of true y data values

        y_pred: (numpy array), array of predicted y data values

    Returns:
        (float): score of R^2 with no y-intercept

    """
    lr = LinearRegression(fit_intercept=False)
    y_true = np.array(y_true).reshape(-1,1) # turn it from an n-vector to nx1-matrix
    lr.fit(y_true, y_pred)
    return lr.score(y_true, y_pred)

def r2_score_fitted(y_true, y_pred):
    """
    Method that calculates the R^2 value

    Args:
        y_true: (numpy array), array of true y data values

        y_pred: (numpy array), array of predicted y data values

    Returns:
        (float): score of R^2

    """
    lr = LinearRegression(fit_intercept=True)
    y_true = np.array(y_true).reshape(-1,1) # turn it from an n-vector to nx1-matrix
    lr.fit(y_true, y_pred)
    return lr.score(y_true, y_pred)

def root_mean_squared_error(y_true, y_pred):
    """
    Method that calculates the root mean squared error (RMSE)

    Args:
        y_true: (numpy array), array of true y data values

        y_pred: (numpy array), array of predicted y data values

    Returns:
        (float): score of RMSE

    """
    return sm.mean_squared_error(y_true, y_pred)**0.5

def rmse_over_stdev(y_true, y_pred, train_y=None):
    """
    Method that calculates the root mean squared error (RMSE) of a set of data, divided by the standard deviation of
    the training data set.

    Args:
        y_true: (numpy array), array of true y data values

        y_pred: (numpy array), array of predicted y data values

        train_y: (numpy array), array of training y data values

    Returns:
        (float): score of RMSE divided by standard deviation of training data

    """
    if train_y is not None:
        stdev = np.std(train_y)
    else:
        stdev = np.std(y_true)
    rmse = root_mean_squared_error(y_true, y_pred)
    return rmse / stdev

def r2_score_adjusted(y_true, y_pred, n_features=None):
    """
    Method that calculates the adjusted R^2 value

    Args:
        y_true: (numpy array), array of true y data values

        y_pred: (numpy array), array of predicted y data values

        n_features: (int), number of features used in the fit

    Returns:
        (float): score of adjusted R^2

    """
    r2 = sm.r2_score(y_true, y_pred)
    # n is sample size
    n = np.array(y_true).shape[0]
    # p is number of features
    p = n_features
    try:
        r2_score_adj = 1 - (((1-r2)*(n-1))/(n-p-1))
    except:
        # No n_features given, just output NaN
        r2_score_adj = 'NaN'
    return r2_score_adj
