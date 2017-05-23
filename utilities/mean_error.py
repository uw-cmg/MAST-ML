import numpy as np

def mean_error(prediction,actual):
    pred_len = len(prediction)
    errors = prediction - actual
    sum_errors = np.sum(errors)
    mean_error = sum_errors / pred_len
    return mean_error
