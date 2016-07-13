from sklearn.kernel_ridge import KernelRidge

__author__ = 'haotian'


def get():

    return KernelRidge(alpha= .00518, gamma = .518, kernel='laplacian')