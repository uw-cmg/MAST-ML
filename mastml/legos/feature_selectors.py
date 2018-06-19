"""
A collection of classes for selecting features
All classes here assume dataframe input and guarantee dataframe output.
(So no numpy arrays.)
"""
from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, SelectFpr, SelectFdr, SelectFwe, VarianceThreshold
from . import utils
# list of sklearn feature selectors: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection

name_to_constructor = {
    'GenericUnivariateSelect': GenericUnivariateSelect,
    'SelectPercentile': SelectPercentile,
    'SelectKBest': SelectKBest,
    'SelectFpr': SelectFpr,
    'SelectFdr': SelectFdr,
    'SelectFwe': SelectFwe,
    'VarianceThreshold': VarianceThreshold,
    'DoNothing': utils.DoNothing,
}

# TODO: this is a problem:
for constructor in name_to_constructor.values():
    constructor.transform = utils.dataframify(constructor.transform)
