"""
A collection of metrics which take a y_correct vector and y_predicted vector
and give some score of them
"""

# All metrics:
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

# Classification metrics:
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, fbeta_score, hamming_loss, jaccard_similarity_score, log_loss, matthews_corrcoef, precision_recall_fscore_support, precision_score, recall_score, roc_auc_score, roc_curve, zero_one_loss

# Regression metrics:
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score

# For r2_score_noint:
from sklearn.linear_model import LinearRegression

# for fixing featuer selectors regress/classif
from sklearn import feature_selection

def r2_score_noint(y_true, y_pred):
    " Get the R^2 coefficient without intercept "
    lr = LinearRegression(fit_intercept=False)
    lr.fit(y_true, y_pred)
    return lr.score(y_true, y_pred)


classification_metrics = {
    'accuracy_score': accuracy_score,  # Accuracy classification score.
    'classification_report': classification_report,  # Build a text report showing the main classification metrics
    'confusion_matrix': confusion_matrix,  # Compute confusion matrix to evaluate the accuracy of a classification
    'f1_score': f1_score,  # Compute the F1 score, also known as balanced F-score or F-measure
    'fbeta_score': fbeta_score,  # Compute the F-beta score
    'hamming_loss': hamming_loss,  # Compute the average Hamming loss.
    'jaccard_similarity_score': jaccard_similarity_score,  # Jaccard similarity coefficient score
    'log_loss': log_loss,  # Log loss, aka logistic loss or cross-entropy loss.
    'matthews_corrcoef': matthews_corrcoef,  # Compute the Matthews correlation coefficient (MCC)
    'precision_recall_fscore_support': precision_recall_fscore_support,  # Compute precision, recall, F-measure and support for each class
    'precision_score': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),  # Compute the precision
    'recall_score': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),  # Compute the recall
    'roc_auc_score': roc_auc_score,  # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    'roc_curve': roc_curve,  # Compute Receiver operating characteristic (ROC)
    'zero_one_loss': zero_one_loss,  # Zero-one classification loss.
}


regression_metrics = {
    'explained_variance_score': explained_variance_score,  # Explained variance regression score function
    'mean_absolute_error': mean_absolute_error,  # Mean absolute error regression loss
    'mean_squared_error': mean_squared_error,  # Mean squared error regression loss
    'root_mean_squared_error': lambda y_true, y_pred: mean_squared_error(y_true, y_pred)**0.5,
    'mean_squared_log_error': mean_squared_log_error,  # Mean squared logarithmic error regression loss
    'median_absolute_error': median_absolute_error,  # Median absolute error regression loss
    'r2_score': r2_score,  # R^2 (coefficient of determination) regression score function.
    'r2_score_noint': r2_score_noint,
}

classification_score_funcs = { 
    'chi2': feature_selection.chi2, #Compute chi-squared stats between each non-negative feature and class.
    'f_classif': feature_selection.f_classif, #Compute the ANOVA F-value for the provided sample.
    'mutual_info_classif': feature_selection.mutual_info_classif, #Estimate mutual information for a discrete target variable.
}

regression_score_funcs = { 
    'f_regression': feature_selection.f_regression, #Univariate linear regression tests.
    'mutual_info_regression': feature_selection.mutual_info_regression, #Estimate mutual information for a continuous target variable.
}




def check_names(metric_names, is_classification):
    task = 'classification' if is_classification else 'regression'
    metrics_dict = classification_metrics if is_classification else regression_metrics
    for name in metric_names:
        if name not in metrics_dict:
            raise Exception(f"Metric '{name}' is not supported for {task}.")
