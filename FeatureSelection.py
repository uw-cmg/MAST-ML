__author__ = 'Ryan Jacobs'

from sklearn.decomposition import PCA
from DataOperations import DataframeUtilities
from FeatureOperations import FeatureIO
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

class PrincipalComponentAnalysis(object):
    """Class to conduct PCA for dimensional reduction of features. Mind that PCA produces linear combinations of features,
    and thus the resulting new features don't have physical meaning.
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    @property
    def get_original_dataframe(self):
        return self.dataframe

    def principal_component_analysis(self, x_features, y_feature):
        pca = PCA(n_components=len(x_features), svd_solver='auto')
        Xnew = pca.fit_transform(X=self.dataframe[x_features])
        dataframe = DataframeUtilities()._array_to_dataframe(array=Xnew)
        dataframe = FeatureIO(dataframe=dataframe).add_custom_features(features_to_add=[y_feature], data_to_add=self.dataframe[y_feature])
        return dataframe

class ClassificationFeatureSelection(object):
    """Class to conduct feature selection routines to reduce the number of input features for a classification problem.
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    @property
    def get_original_dataframe(self):
        return self.dataframe

    def univariate_feature_selection(self, x_features, y_feature, features_to_keep):
        selector = SelectKBest(score_func=f_classif, k=features_to_keep)
        Xnew = selector.fit_transform(X=self.dataframe[x_features], y=self.dataframe[y_feature])
        feature_indices_selected = selector.get_support(indices=True)
        # Get the names of the features based on their indices, for features selected from feature selection
        feature_names_selected = []
        for i in range(len(x_features)):
            if i in feature_indices_selected:
                feature_names_selected.append(x_features[i])
        dataframe = DataframeUtilities()._array_to_dataframe(array=Xnew)
        dataframe = DataframeUtilities()._assign_columns_as_features(dataframe=dataframe, x_features=feature_names_selected, y_feature=y_feature, remove_first_row=False)
        # Add y_feature back into the dataframe
        dataframe = FeatureIO(dataframe=dataframe).add_custom_features(features_to_add=[y_feature],data_to_add=self.dataframe[y_feature])
        return dataframe

    def recursive_feature_selection(self):
        pass

class RegressionFeatureSelection(object):
    """Class to conduct feature selection routines to reduce the number of input features for a regression problem.
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    @property
    def get_original_dataframe(self):
        return self.dataframe

    def univariate_feature_selection(self, x_features, y_feature, features_to_keep):
        selector = SelectKBest(score_func=f_regression, k=features_to_keep)
        Xnew = selector.fit_transform(X=self.dataframe[x_features], y=self.dataframe[y_feature])
        feature_indices_selected = selector.get_support(indices=True)
        # Get the names of the features based on their indices, for features selected from feature selection
        feature_names_selected = []
        for i in range(len(x_features)):
            if i in feature_indices_selected:
                feature_names_selected.append(x_features[i])
        dataframe = DataframeUtilities()._array_to_dataframe(array=Xnew)
        dataframe = DataframeUtilities()._assign_columns_as_features(dataframe=dataframe, x_features=feature_names_selected, y_feature=y_feature, remove_first_row=False)
        # Add y_feature back into the dataframe
        dataframe = FeatureIO(dataframe=dataframe).add_custom_features(features_to_add=[y_feature],data_to_add=self.dataframe[y_feature])
        return dataframe