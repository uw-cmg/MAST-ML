"""
This module contains constructors for different model score metrics. Most model metrics are obtained from scikit-learn,
while others are custom variations.

The full list of score functions in scikit-learn can be found at: http://scikit-learn.org/stable/modules/model_evaluation.html
"""

import numpy as np
import sklearn.feature_selection as fs
import sklearn.metrics as sm
from sklearn.linear_model import LinearRegression # for r2_score_noint

classification_metrics = {
    'accuracy':           (True, sm.accuracy_score),
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

regression_metrics = {
    'explained_variance':     (True, sm.explained_variance_score),
    'mean_absolute_error':    (False, sm.mean_absolute_error),
    'mean_squared_error':     (False, sm.mean_squared_error),
    'mean_squared_log_error': (False, sm.mean_squared_log_error),
    'median_absolute_error':  (False, sm.median_absolute_error),
    'R2': (True, sm.r2_score)
}

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
regression_metrics['R2_noint'] = (True, r2_score_noint)

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
regression_metrics['R2_fitted'] = (True, r2_score_fitted)

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
regression_metrics['root_mean_squared_error'] = (False, root_mean_squared_error)

# TODO: consider two rmse/stdev metrics: one that uses stdev of full data set, and one that uses stdev of data per split
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
regression_metrics['rmse_over_stdev'] = (False, rmse_over_stdev)

def adjusted_r2_score(y_true, y_pred, n_features=None):
    """
    Method that calculates the adjusted R^2 value

    Args:
        y_true: (numpy array), array of true y data values
        y_pred: (numpy array), array of predicted y data values
        n_features: (int), number of features used in the fit

    Returns:
        (float): score of adjusted R^2

    """
    r2 = r2_score(y_true, y_pred)
    # n is sample size
    n = len(y_true)
    # p is number of features
    p = n_features
    try:
        r2_score_adj = 1 - (((1-r2)*(n-1))/(n-p-1))
    except:
        # No n_features given, just output NaN
        r2_score_adj = 'NaN'
    return r2_score_adj
regression_metrics['R2_adjusted'] = (True, adjusted_r2_score)

classification_score_funcs = {
    'chi2': fs.chi2, # Compute chi-squared stats between each non-negative feature and class.
    'f_classif': fs.f_classif, # Compute the ANOVA F-value for the provided sample.
    'mutual_info_classif': fs.mutual_info_classif, # Estimate mutual information for a discrete target variable.
}

regression_score_funcs = {
    'f_regression': fs.f_regression, # Univariate linear regression tests.
    'mutual_info_regression': fs.mutual_info_regression, # Estimate mutual information for a continuous target variable.
}

nice_names = {
    # classification:
    'accuracy': 'Accuracy',
    'f1_binary': '$F_1$',
    'f1_macro': 'f1_macro',
    'f1_micro': 'f1_micro',
    'f1_samples': 'f1_samples',
    'f1_weighted': 'f1_weighted',
    'log_loss': 'log_loss',
    'precision_binary': 'Precision',
    'precision_macro': 'prec_macro',
    'precision_micro': 'prec_micro',
    'precision_samples': 'prec_samples',
    'precision_weighted': 'prec_weighted',
    'recall_binary': 'Recall',
    'recall_macro': 'rcl_macro',
    'recall_micro': 'rcl_micro',
    'recall_samples': 'rcl_samples',
    'recall_weighted': 'rcl_weighted',
    'roc_auc': 'ROC_AUC',
    # regression:
    'explained_variance': 'expl_var',
    'mean_absolute_error': 'MAE',
    'mean_squared_error': 'MSE',
    'mean_squared_log_error': 'MSLE',
    'median_absolute_error': 'MedAE',
    'root_mean_squared_error': 'RMSE',
    'rmse_over_stdev': r'RMSE/$\sigma_y$',
    'R2': '$R^2$',
    'R2_noint': '$R^2_{noint}$',
    'R2_adjusted': '$R^2_{adjusted}$'
}

def check_and_fetch_names(metric_names, is_classification):
    """
    Method that checks whether chosen metrics to evaluate models are appropriate for user-specified models (e.g.
    classification vs. regression models)

    Args:
        metric_names: (numpy array), array of true y data values
        is_classification: (bool), whether the task is a classification task

    Returns:
        functions (dict): dict containing the appropriate metric objects (e.g. classification vs. regression metrics)

    """
    " Ensures all metrics are appropriate for task "
    task = 'classification' if is_classification else 'regression'
    metrics_dict = classification_metrics if is_classification else regression_metrics
    functions = {}
    for name in metric_names:
        if name not in metrics_dict:
            raise Exception(f"Metric '{name}' is not supported for {task}.\n"
                            f"Valid metrics for {task}: {list(metrics_dict.keys())}")
        functions[name] = metrics_dict[name]
    return functions
