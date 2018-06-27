"""
A collection of classes for selecting features
All classes here assume dataframe input and guarantee dataframe output.
(So no numpy arrays.)
"""
import pandas as pd
from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, SelectFpr, SelectFdr, SelectFwe, VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn import feature_selection

from . import util_legos, lego_utils
# list of sklearn feature selectors: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection



score_func_selectors = { 
    'SelectKBest': feature_selection.SelectKBest, #Select features according to the k highest scores.
    'SelectFpr': feature_selection.SelectFpr, #Filter: Select the pvalues below alpha based on a FPR test.
    'SelectFdr': feature_selection.SelectFdr, #Filter: Select the p-values for an estimated false discovery rate
    'SelectFwe': feature_selection.SelectFwe, #Filter: Select the p-values corresponding to Family-wise error rate
    'GenericUnivariateSelect': feature_selection.GenericUnivariateSelect, #Univariate feature selector with configurable strategy.
    'SelectPercentile': feature_selection.SelectPercentile, #Select features according to a percentile of the highest scores.
}

model_selectors = { # feature selectors which take a model instance as first parameter
    'SelectFromModel': feature_selection.SelectFromModel, #Meta-transformer for selecting features based on importance weights.
    'RFE': feature_selection.RFE, #Feature ranking with recursive feature elimination.
    'RFECV': feature_selection.RFECV, #Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.
}

other_selectors = {
    'VarianceThreshold': feature_selection.VarianceThreshold, #Feature selector that removes all low-variance features.
}








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

name_to_constructor = {
    'GenericUnivariateSelect': GenericUnivariateSelect,
    'SelectPercentile': SelectPercentile,
    'SelectKBest': SelectKBest,
    'SelectFpr': SelectFpr,
    'SelectFdr': SelectFdr,
    'SelectFwe': SelectFwe,
    'VarianceThreshold': VarianceThreshold,
}

# Modify all sklearn transform methods to return dataframes:
for constructor in name_to_constructor.values():
    constructor.old_transform = constructor.transform
    constructor.transform = lego_utils.dataframify_selector(constructor.transform)

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
