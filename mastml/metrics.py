"""
A collection of functions which take a y_correct vector and y_predicted vector
and give some score of them
"""

import sklearn.feature_selection as fs
from sklearn.linear_model import LinearRegression # for r2_score_noint
from sklearn.metrics import scorer



# list of score functions: http://scikit-learn.org/stable/modules/model_evaluation.html
classification_metrics = [
    'accuracy',
    'average_precision',
    'f1',
    'f1_macro',
    'f1_micro',
    'f1_samples',
    'f1_weighted',
    'neg_log_loss',
    'precision',
    'precision_macro',
    'precision_micro',
    'precision_samples',
    'precision_weighted',
    'recall',
    'recall_macro',
    'recall_micro',
    'recall_samples',
    'recall_weighted',
    'roc_auc',
]
classification_metrics = {name: scorer.SCORERS[name] for name in classification_metrics}

regression_metrics = [
    'explained_variance',
    'log_loss',
    'neg_mean_absolute_error',
    'neg_mean_squared_error',
    'neg_mean_squared_log_error',
    'neg_median_absolute_error',
    'mean_absolute_error',
    'mean_squared_error',
    'median_absolute_error',
    'r2',
]
regression_metrics = {name: scorer.SCORERS[name] for name in regression_metrics}

def r2_score_noint(estimator, X, y_true):
    " Get the R^2 coefficient without intercept "
    y_pred = estimator.predict(X)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(y_true, y_pred)
    return lr.score(y_true, y_pred)
def root_mean_squared_error(estimator, X, y_true):
    return regression_metrics['mean_squared_error'](estimator, X, y_true)**0.5
def neg_root_mean_squared_error(estimator, X, y_true):
    return -regression_metrics['mean_squared_error'](estimator, X, y_true)**0.5

regression_metrics['r2_noint'] = r2_score_noint
regression_metrics['root_mean_squared_error'] = root_mean_squared_error
regression_metrics['neg_root_mean_squared_error'] = neg_root_mean_squared_error


#clustering_metrics = [
#    'adjusted_mutual_info_score',
#    'adjusted_rand_score',
#    'completeness_score',
#    'fowlkes_mallows_score',
#    'homogeneity_score',
#    'mutual_info_score',
#    'normalized_mutual_info_score',
#    'v_measure_score',
#]
#clustering_metrics = {name: scorer.SCORERS[name] for name in clustering_metrics}


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
    'average_precision': 'avg_prec',
    'f1': 'f1',
    'f1_macro': 'f1_macro',
    'f1_micro': 'f1_micro',
    'f1_samples': 'f1_samples',
    'f1_weighted': 'f1_weighted',
    'neg_log_loss': 'neg_log_loss',
    'precision': 'precision',
    'precision_macro': 'prec_macro',
    'precision_micro': 'prec_micro',
    'precision_samples': 'prec_samples',
    'precision_weighted': 'prec_weighted',
    'recall': 'recall',
    'recall_macro': 'recall_macro',
    'recall_micro': 'recall_micro',
    'recall_samples': 'recall_samples',
    'recall_weighted': 'recall_weighted',
    'roc_auc': 'roc_auc',
    # regression: '',
    'explained_variance': 'explained_var',
    'log_loss': 'log_loss',
    'neg_mean_absolute_error': '-MAE',
    'neg_mean_squared_error': '-MSE',
    'neg_mean_squared_log_error': '-MSLE',
    'neg_median_absolute_error': '-MedAE',
    'neg_root_mean_squared_error': '-RMSE',
    'mean_absolute_error': '+MAE',
    'mean_squared_error': '+MSE',
    'root_mean_squared_error': '+RMSE',
    'median_absolute_error': '+MedAE',
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
            raise Exception(f"Metric '{name}' is not supported for {task}.")
        functions[name] = metrics_dict[name]
    return functions
