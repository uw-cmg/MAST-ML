from sklearn.linear_model import LinearRegression

__author__ = 'haotian'


def get():
    from warnings import warn
    warn('The usage of linear_model.py will be deprecated', DeprecationWarning)
    return LinearRegression()
