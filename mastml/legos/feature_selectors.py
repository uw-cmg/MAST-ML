"""
A collection of classes for selecting features
All classes here assume dataframe input and guarantee dataframe output.
(So no numpy arrays.)
"""

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import sklearn.feature_selection as fs

from mlxtend.feature_selection import SequentialFeatureSelector # TODO: TEST!!

from . import util_legos, lego_utils
# list of sklearn feature selectors: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection

score_func_selectors = { 
    'GenericUnivariateSelect': fs.GenericUnivariateSelect, # Univariate feature selector with configurable strategy.
    'SelectFdr': fs.SelectFdr, # Filter: Select the p-values for an estimated false discovery rate
    'SelectFpr': fs.SelectFpr, # Filter: Select the pvalues below alpha based on a FPR test.
    'SelectFwe': fs.SelectFwe, # Filter: Select the p-values corresponding to Family-wise error rate
    'SelectKBest': fs.SelectKBest, # Select features according to the k highest scores.
    'SelectPercentile': fs.SelectPercentile, # Select features according to a percentile of the highest scores.
}

model_selectors = { # feature selectors which take a model instance as first parameter
    'RFE': fs.RFE, # Feature ranking with recursive feature elimination.
    'RFECV': fs.RFECV, # Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.
    'SelectFromModel': fs.SelectFromModel, # Meta-transformer for selecting features based on importance weights.
}

other_selectors = {
    'VarianceThreshold': fs.VarianceThreshold, # Feature selector that removes all low-variance features.
}

good_models = ['RidgeClassifier', 'LinearSVC', 'SGDClassifier', 'Perceptron',
               'PassiveAggressiveClassifier', 'BernoulliNB', 'MultinomialNB',
               'KNeighborsClassifier', 'NearestCentroid', 'RandomForestClassifier']

name_to_constructor = {
    'GenericUnivariateSelect': fs.GenericUnivariateSelect,
    'RFE': fs.RFE,
    'RFECV': fs.RFECV,
    'SelectFdr': fs.SelectFdr,
    'SelectFpr': fs.SelectFpr,
    'SelectFromModel': fs.SelectFromModel,
    'SelectFwe': fs.SelectFwe,
    'SelectKBest': fs.SelectKBest,
    'SelectPercentile': fs.SelectPercentile,
    'VarianceThreshold': fs.VarianceThreshold,
    'SequentialFeatureSelector': SequentialFeatureSelector,
}

# Modify all sklearn transform methods to return dataframes:
for constructor in name_to_constructor.values():
    constructor.old_transform = constructor.transform
    constructor.transform = lego_utils.dataframify_selector(constructor.transform)

class PassThrough(BaseEstimator, TransformerMixin):
    " Keep specific features and pass them on to the other side "
    def __init__(self, features):
        if not isinstance(features, list):
            features = [features]
        self.features = features
    def fit(self, df, y=None):
        for feature in self.features:
            if feature not in df.columns:
                raise Exception(f"Specified feature '{feature}' to PassThrough not present in data file.")
    def transform(self, df):
        return df[self.features]

# Mess with PCA stuff:
old_transform = PCA.transform
def new_transform(self, df):
    arr = old_transform(self, df)
    labels = ['pca_'+str(i) for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=labels)
PCA.transform = new_transform

# Custom selectors don't need to be dataframified
name_to_constructor.update({
    'PassThrough': PassThrough,
    'DoNothing': util_legos.DoNothing,
    'PCA': PCA,
})

# done!
