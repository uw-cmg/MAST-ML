__author__ = 'Ryan Jacobs'

from sklearn.decomposition import PCA
from DataOperations import DataframeUtilities
from FeatureOperations import FeatureIO
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.svm import SVR, SVC
from sklearn.feature_selection import RFE
from sklearn.linear_model import RandomizedLasso
import sys
import logging

class DimensionalReduction(object):
    """Class to conduct PCA and constant feature removal for dimensional reduction of features. Mind that PCA produces linear combinations of features,
    and thus the resulting new features don't have physical meaning.
    """
    def __init__(self, dataframe, x_features, y_feature):
        self.dataframe = dataframe
        self.x_features = x_features
        self.y_feature = y_feature

    @property
    def get_original_dataframe(self):
        return self.dataframe

    def remove_constant_features(self):
        dataframe = self.dataframe.loc[:, self.dataframe.var() != 0.0]
        return dataframe

    def principal_component_analysis(self, x_features, y_feature):
        pca = PCA(n_components=len(x_features), svd_solver='auto')
        Xnew = pca.fit_transform(X=self.dataframe[x_features])
        dataframe = DataframeUtilities()._array_to_dataframe(array=Xnew)
        dataframe = FeatureIO(dataframe=dataframe).add_custom_features(features_to_add=[y_feature], data_to_add=self.dataframe[y_feature])
        return dataframe

class FeatureSelection(object):
    """Class to conduct feature selection routines to reduce the number of input features for regression and classification problems.
    """
    def __init__(self, dataframe, x_features, y_feature, selection_type='Regression'):
        self.dataframe = dataframe
        self.x_features = x_features
        self.y_feature = y_feature
        self.selection_type = selection_type

        if self.selection_type not in ['Regression', 'regression', 'Classification', 'classification']:
            logging.info('ERROR: You must specify "selection_type" as either "regression" or "classification"')
            sys.exit()

    @property
    def get_original_dataframe(self):
        return self.dataframe

    def univariate_feature_selection(self, number_features_to_keep, save_to_csv=True):
        if self.selection_type == 'Regression' or self.selection_type == 'regression':
            selector = SelectKBest(score_func=f_regression, k=number_features_to_keep)
        if self.selection_type == 'Classification' or self.selection_type == 'classification':
            selector = SelectKBest(score_func=f_classif, k=number_features_to_keep)
        Xnew = selector.fit_transform(X=self.dataframe[self.x_features], y=self.dataframe[self.y_feature])
        feature_names_selected = MiscOperations().get_selector_feature_names(selector=selector, x_features=self.x_features)
        dataframe = DataframeUtilities()._array_to_dataframe(array=Xnew)
        dataframe = DataframeUtilities()._assign_columns_as_features(dataframe=dataframe, x_features=feature_names_selected, y_feature=self.y_feature, remove_first_row=False)
        # Add y_feature back into the dataframe
        dataframe = FeatureIO(dataframe=dataframe).add_custom_features(features_to_add=[self.y_feature],data_to_add=self.dataframe[self.y_feature])
        if save_to_csv == bool(True):
            dataframe.to_csv('input_with_univariate_feature_selection.csv', index=False)
        return dataframe

    def recursive_feature_elimination(self, number_features_to_keep, save_to_csv=True):
        if self.selection_type == 'Regression' or self.selection_type == 'regression':
            estimator = SVR(kernel='linear')
        if self.selection_type == 'Classification' or self.selection_type == 'classification':
            estimator = SVC(kernel='linear')
        selector = RFE(estimator=estimator, n_features_to_select=number_features_to_keep)
        Xnew = selector.fit_transform(X=self.dataframe[self.x_features], y=self.dataframe[self.y_feature])
        feature_names_selected = MiscOperations().get_selector_feature_names(selector=selector, x_features=self.x_features)
        #feature_names_selected = MiscOperations().get_ranked_feature_names(selector=selector, x_features=self.x_features, number_features_to_keep=number_features_to_keep)
        dataframe = DataframeUtilities()._array_to_dataframe(array=Xnew)
        dataframe = DataframeUtilities()._assign_columns_as_features(dataframe=dataframe, x_features=feature_names_selected, y_feature=self.y_feature, remove_first_row=False)
        # Add y_feature back into the dataframe
        dataframe = FeatureIO(dataframe=dataframe).add_custom_features(features_to_add=[self.y_feature],data_to_add=self.dataframe[self.y_feature])
        if save_to_csv == bool(True):
            dataframe.to_csv('input_with_RFE_feature_selection.csv', index=False)
        return dataframe

    def stability_selection(self, number_features_to_keep, save_to_csv=True):
        if self.selection_type == 'Regression' or self.selection_type == 'regression':
            selector = RandomizedLasso()
            selector.fit(X=self.dataframe[self.x_features], y=self.dataframe[self.y_feature])
            feature_names_selected = MiscOperations().get_ranked_feature_names(selector=selector, x_features=self.x_features, number_features_to_keep=number_features_to_keep)
            dataframe = FeatureIO(dataframe=self.dataframe).keep_custom_features(features_to_keep=feature_names_selected, y_feature=self.y_feature)
        if self.selection_type == 'Classification' or self.selection_type == 'classification':
            print('Stability selection is currently only configured for regression tasks')
            sys.exit()
        if save_to_csv == bool(True):
            dataframe.to_csv('input_with_stability_feature_selection.csv', index=False)
        return dataframe

class MiscOperations():

    @classmethod
    def get_selector_feature_names(cls, selector, x_features):
        feature_indices_selected = selector.get_support(indices=True)
        # Get the names of the features based on their indices, for features selected from feature selection
        feature_names_selected = []
        for i in range(len(x_features)):
            if i in feature_indices_selected:
                feature_names_selected.append(x_features[i])
        return feature_names_selected

    @classmethod
    def get_ranked_feature_names(cls, selector, x_features, number_features_to_keep):
        try:
            ranked_features = sorted(zip(selector.scores_, x_features), reverse=True)
        except AttributeError:
            ranked_features = sorted(zip(selector.ranking_, x_features))
        feature_names_selected = []
        count = 0
        for i in range(len(ranked_features)):
            if count < number_features_to_keep:
                feature_names_selected.append(ranked_features[i][1])
                count += 1
        return feature_names_selected