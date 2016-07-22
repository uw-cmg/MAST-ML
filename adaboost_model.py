__author__ = 'hao yuan'
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import configuration_parser


def get():
    config = configuration_parser.parse()
    estimators = config.getint(__name__, 'estimators')
    lr = config.getfloat(__name__, 'learning rate')
    loss = config.get(__name__, 'loss function')
    exec('max_depth=' + config.get(__name__, 'max_depth'), locals(), globals())
    min_samples_split = config.getint(__name__, 'min_samples_split')
    min_samples_leaf = config.getint(__name__, 'min_samples_leaf')
    return AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth,min_samples_split=min_samples_split,
                                                   min_samples_leaf=min_samples_leaf),n_estimators=estimators,loss=loss,
                                                   learning_rate=lr)