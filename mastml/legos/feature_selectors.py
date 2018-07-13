"""
A collection of classes for selecting features
All classes here assume dataframe input and guarantee dataframe output.
(So no numpy arrays.)
"""

from functools import wraps

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import sklearn.feature_selection as fs
from mlxtend.feature_selection import SequentialFeatureSelector

from . import util_legos

# list of sklearn feature selectors:
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection

def dataframify_selector(transform):
    " Special dataframify which preserves column names for feature selectors "
    @wraps(transform)
    def new_transform(self, df):
        if isinstance(df, pd.DataFrame):
            return df[df.columns[self.get_support(indices=True)]]
        else: # just in case you try to use it with an array ;)
            return df
    return new_transform

def dataframify_new_column_names(transform, name):
    def new_transform(self, df):
        arr = transform(self, df.values)
        labels = [name+str(i) for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=labels)
    return new_transform

def fitify_just_use_values(fit):
    def new_fit(self, X_df, y_df):
        return fit(self, X_df.values, y_df.values)
    return new_fit

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

# Union together the above dicts for the primary export:
name_to_constructor = dict(**score_func_selectors, **model_selectors, **other_selectors)

# Modify all sklearn transform methods to return dataframes:
for constructor in name_to_constructor.values():
    constructor.old_transform = constructor.transform
    constructor.transform = dataframify_selector(constructor.transform)

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
PCA.transform = dataframify_new_column_names(PCA.transform, 'pca_')

# Mess with SFS stuff:
SequentialFeatureSelector.transform = dataframify_new_column_names(SequentialFeatureSelector.transform, 'sfs_')
SequentialFeatureSelector.fit = fitify_just_use_values(SequentialFeatureSelector.fit)
model_selectors['SequentialFeatureSelector'] = SequentialFeatureSelector
name_to_constructor['SequentialFeatureSelector'] = SequentialFeatureSelector


# Custom selectors don't need to be dataframified
name_to_constructor.update({
    'PassThrough': PassThrough,
    'DoNothing': util_legos.DoNothing,
    'PCA': PCA,
    'SequentialFeatureSelector': SequentialFeatureSelector,
})

# done!
