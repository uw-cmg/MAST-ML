# Classification metrics:
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, fbeta_score, hamming_loss, jaccard_similarity_score, log_loss, matthews_corrcoef, precision_recall_fscore_support, precision_score, recall_score, roc_auc_score, roc_curve, zero_one_loss
# Regression metrics:
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score

# All metrics:
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

classification_metrics = {
    'accuracy_score': accuracy_score,	#Accuracy classification score.
    'classification_report': classification_report,	#Build a text report showing the main classification metrics
    'confusion_matrix': confusion_matrix,	#Compute confusion matrix to evaluate the accuracy of a classification
    'f1_score': f1_score,	#Compute the F1 score, also known as balanced F-score or F-measure
    'fbeta_score': fbeta_score,	#Compute the F-beta score
    'hamming_loss': hamming_loss,	#Compute the average Hamming loss.
    'jaccard_similarity_score': jaccard_similarity_score,	#Jaccard similarity coefficient score
    'log_loss': log_loss,	#Log loss, aka logistic loss or cross-entropy loss.
    'matthews_corrcoef': matthews_corrcoef,	#Compute the Matthews correlation coefficient (MCC)
    'precision_recall_fscore_support': precision_recall_fscore_support,	#Compute precision, recall, F-measure and support for each class
    'precision_score': precision_score,	#Compute the precision
    'recall_score': recall_score,	#Compute the recall
    'roc_auc_score': roc_auc_score,	#Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    'roc_curve': roc_curve,	#Compute Receiver operating characteristic (ROC)
    'zero_one_loss': zero_one_loss,	#Zero-one classification loss.
}

regression_metrics = {
    'explained_variance_score': explained_variance_score,	#Explained variance regression score function
    'mean_absolute_error': mean_absolute_error,	#Mean absolute error regression loss
    'mean_squared_error': mean_squared_error,	#Mean squared error regression loss
    'mean_squared_log_error': mean_squared_log_error,	#Mean squared logarithmic error regression loss
    'median_absolute_error': median_absolute_error,	#Median absolute error regression loss
    'r2_score': r2_score,	#R^2 (coefficient of determination) regression score function.
}

def check_mixed(metric_names):
    found_regression = found_classification = False
    for name in metric_names:
        if name in classification_metrics:
            found_classification = True
        if name in regression_metrics:
            found_regression = True
        fg
