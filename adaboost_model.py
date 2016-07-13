__author__ = 'hao yuan'
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


def get():
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=12),
             n_estimators=275)
    return model
