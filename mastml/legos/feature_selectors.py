from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, SelectFpr, SelectFdr, SelectFwe, VarianceThreshold
from .utils import DoNothing
# list of sklearn feature selectors: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection

name_to_constructor = {
    'GenericUnivariateSelect': GenericUnivariateSelect,
    'SelectPercentile': SelectPercentile,
    'SelectKBest': SelectKBest,
    'SelectFpr': SelectFpr,
    'SelectFdr': SelectFdr,
    'SelectFwe': SelectFwe,
    'VarianceThreshold': VarianceThreshold,
    'DoNothing': DoNothing,
}

