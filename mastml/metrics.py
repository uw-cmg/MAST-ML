"""
A collection of functions which take a y_correct vector and y_predicted vector
and give some score of them
"""

import numpy as np
import sklearn.metrics as sm
import sklearn.feature_selection as fs
from sklearn.linear_model import LinearRegression # for r2_score_noint

def r2_score_noint(y_true, y_pred):
    " Get the R^2 coefficient without intercept "
    y_true = np.array(y_true)
    y_true = y_true.reshape(y_true.shape[0], 1)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(y_true, y_pred)
    return lr.score(y_true, y_pred)

classification_metrics = {
    'accuracy_score': sm.accuracy_score,  # Accuracy classification score.
    'classification_report': sm.classification_report,  # Build a text report showing the main classification metrics
    'confusion_matrix': sm.confusion_matrix,  # Compute confusion matrix to evaluate the accuracy of a classification
    'f1_score': sm.f1_score,  # Compute the F1 score, also known as balanced F-score or F-measure
    'fbeta_score': sm.fbeta_score,  # Compute the F-beta score
    'hamming_loss': sm.hamming_loss,  # Compute the average Hamming loss.
    'jaccard_similarity_score': sm.jaccard_similarity_score,  # Jaccard similarity coefficient score
    'log_loss': sm.log_loss,  # Log loss, aka logistic loss or cross-entropy loss.
    'matthews_corrcoef': sm.matthews_corrcoef,  # Compute the Matthews correlation coefficient (MCC)
    'precision_recall_fscore_support': sm.precision_recall_fscore_support,  # Compute precision, recall, F-measure and support for each class
    'precision_score': lambda y_true, y_pred: sm.precision_score(y_true, y_pred, average='weighted'),  # Compute the precision
    'recall_score': lambda y_true, y_pred: sm.recall_score(y_true, y_pred, average='weighted'),  # Compute the recall
    'roc_auc_score': sm.roc_auc_score,  # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    'roc_curve': sm.roc_curve,  # Compute Receiver operating characteristic (ROC)
    'zero_one_loss': sm.zero_one_loss,  # Zero-one classification loss.
}

regression_metrics = {
    'explained_variance_score': sm.explained_variance_score,  # Explained variance regression score function
    'mean_absolute_error': sm.mean_absolute_error,  # Mean absolute error regression loss
    'mean_squared_error': sm.mean_squared_error,  # Mean squared error regression loss
    'root_mean_squared_error': lambda y_true, y_pred: sm.mean_squared_error(y_true, y_pred)**0.5,
    'mean_squared_log_error': sm.mean_squared_log_error,  # Mean squared logarithmic error regression loss
    'median_absolute_error': sm.median_absolute_error,  # Median absolute error regression loss
    'r2_score': sm.r2_score,  # R^2 (coefficient of determination) regression score function.
    'r2_score_noint': r2_score_noint,
}

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
    'accuracy_score': 'Acc',
    'confusion_matrix': 'Confusion Matrix',
    'explained_variance_score': 'expl var',
    'f1_score': 'f1',
    'fbeta_score': 'fbeta score',
    'hamming_loss': 'Hamming Loss',
    'jaccard_similarity_score': 'Jaccard Similarity Score',
    'log_loss': 'Log Loss',
    'matthews_corrcoef': 'Matthews',
    'mean_absolute_error': 'MAE',
    'mean_squared_error': 'MSE',
    'mean_squared_log_error': 'MSLE',
    'median_absolute_error': 'MeAE',
    'precision_recall_fscore_support': 'pr fscore',
    'precision_score': 'Prec',
    'r2_score': '$r^2$',
    'r2_score_noint': '$r^2$ noint',
    'recall_score': 'Rec',
    'roc_auc_score': 'roc auc',
    'roc_curve': 'roc',
    'root_mean_squared_error': 'RMSE',
    'zero_one_loss': '10ss',
}

def check_names(metric_names, is_classification):
    task = 'classification' if is_classification else 'regression'
    metrics_dict = classification_metrics if is_classification else regression_metrics
    for name in metric_names:
        if name not in metrics_dict:
            raise Exception(f"Metric '{name}' is not supported for {task}.")
