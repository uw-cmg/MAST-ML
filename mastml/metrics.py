"""
A collection of functions which take a y_correct vector and y_predicted vector
and give some score of them
"""

import numpy as np
import sklearn.feature_selection as fs
import sklearn.metrics as sm
from sklearn.linear_model import LinearRegression # for r2_score_noint

# list of score functions: http://scikit-learn.org/stable/modules/model_evaluation.html
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
    'r2':                     (True, sm.r2_score),
}

def r2_score_noint(y_true, y_pred):
    " Get the R^2 coefficient without intercept "
    lr = LinearRegression(fit_intercept=False)
    y_true = np.array(y_true).reshape(-1,1) # turn it from an n-vector to nx1-matrix
    lr.fit(y_true, y_pred)
    return lr.score(y_true, y_pred)
regression_metrics['r2_noint'] = (True, r2_score_noint)

def root_mean_squared_error(y_true, y_pred):
    return sm.mean_squared_error(y_true, y_pred)**0.5
regression_metrics['root_mean_squared_error'] = (False, root_mean_squared_error)

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
    'accuracy': 'accuracy',
    'f1_binary': 'f1',
    'f1_macro': 'f1_macro',
    'f1_micro': 'f1_micro',
    'f1_samples': 'f1_samples',
    'f1_weighted': 'f1_weighted',
    'log_loss': 'log_loss',
    'precision_binary': 'precision',
    'precision_macro': 'prec_macro',
    'precision_micro': 'prec_micro',
    'precision_samples': 'prec_samples',
    'precision_weighted': 'prec_weighted',
    'recall_binary': 'rcl',
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
    'r2': 'r2',
    'r2_noint': 'r2_noint',
}

def check_and_fetch_names(metric_names, is_classification):
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
