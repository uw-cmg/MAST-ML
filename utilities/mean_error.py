import numpy as np

def mean_error(prediction,actual):
    mean_error = 0;
    for i ,j in zip(np.asarray(prediction).ravel(),np.asarray(actual).ravel()):
        mean_error += i-j
    return mean_error/len(prediction.ravel())