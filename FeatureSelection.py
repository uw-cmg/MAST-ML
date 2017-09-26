__author__ = 'Ryan Jacobs'

from sklearn.decomposition import PCA
from DataOperations import DataframeUtilities
from FeatureOperations import FeatureIO
from MASTMLInitializer import ConfigFileParser
from sklearn.model_selection import learning_curve, ShuffleSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_regression, mutual_info_classif
from sklearn.svm import SVR, SVC
from sklearn.feature_selection import RFE
from sklearn.linear_model import RandomizedLasso, LinearRegression
from sklearn.kernel_ridge import KernelRidge
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import pandas as pd
import os

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
    def __init__(self, configdict, dataframe, x_features, y_feature):
        self.configdict = configdict
        self.dataframe = dataframe
        self.x_features = x_features
        self.y_feature = y_feature

    @property
    def get_original_dataframe(self):
        return self.dataframe

    def sequential_forward_selection(self, number_features_to_keep):
        sfs = SFS(KernelRidge(alpha=0.01, kernel='linear'), k_features=number_features_to_keep, forward=True, floating=False, verbose=0, scoring='r2', cv=ShuffleSplit(n_splits=5, test_size=0.2))
        sfs = sfs.fit(X=np.array(self.dataframe[self.x_features]), y=np.array(self.dataframe[self.y_feature]))
        Xnew = sfs.fit_transform(X=np.array(self.dataframe[self.x_features]), y=np.array(self.dataframe[self.y_feature]))
        feature_indices_selected = sfs.k_feature_idx_
        x_features_to_keep = []
        for index in feature_indices_selected:
            x_features_to_keep.append(self.x_features[index])

        #print('feature indices:')
        #print(feature_indices_selected)
        #for index in feature_indices_selected:
        #    print('index:', index)
        #    print('feature:', self.x_features[index])
        #print(self.x_features)

        #dataframe = DataframeUtilities()._array_to_dataframe(array=Xnew)
        feature_names_selected = MiscFeatureSelectionOperations().get_forward_selection_feature_names(feature_indices_selected=feature_indices_selected, x_features=self.x_features)
        #dataframe = DataframeUtilities()._assign_columns_as_features(dataframe=self.dataframe, x_features=feature_names_selected, y_feature=self.y_feature, remove_first_row=False)
        #dataframe = DataframeUtilities()._array_to_dataframe(array=Xnew)
        dataframe = FeatureIO(dataframe=self.dataframe).keep_custom_features(features_to_keep=x_features_to_keep)
        # Add y_feature back into the dataframe
        dataframe = FeatureIO(dataframe=dataframe).add_custom_features(features_to_add=[self.y_feature],data_to_add=self.dataframe[self.y_feature])
        dataframe = dataframe.dropna()
        # Get forward selection data
        metricdict = sfs.get_metric_dict()
        fs_dataframe = pd.DataFrame.from_dict(metricdict).T
        logging.info(("Summary of forward selection:"))
        logging.info(fs_dataframe)

        mfso = MiscFeatureSelectionOperations()
        filetag = mfso.get_feature_filetag(configdict=self.configdict, dataframe=dataframe)
        mfso.save_data_to_csv(configdict=self.configdict, dataframe=dataframe, feature_selection_str='input_with_sequential_forward_selection', filetag=filetag)
        mfso.save_data_to_csv(configdict=self.configdict, dataframe=fs_dataframe, feature_selection_str='sequential_forward_selection_data', filetag=filetag)
        mfso.get_forward_selection_learning_curve(configdict=self.configdict, metricdict=metricdict, filetag=filetag)

        return dataframe

    def univariate_feature_selection(self, number_features_to_keep, use_mutual_info):
        if 'regression' in self.y_feature:
            selection_type = 'regression'
        elif 'classification' in self.y_feature:
            selection_type = 'classification'
        else:
            print('You must specify either "regression" or "classification" in your y_feature name')
            sys.exit()

        if use_mutual_info == False or use_mutual_info == 'False':
            if selection_type == 'regression':
                selector = SelectKBest(score_func=f_regression, k=number_features_to_keep)
            elif selection_type == 'classification':
                selector = SelectKBest(score_func=f_classif, k=number_features_to_keep)
        elif use_mutual_info == True or use_mutual_info == 'True':
            if selection_type == 'regression':
                selector = SelectKBest(score_func=mutual_info_regression, k=number_features_to_keep)
            elif selection_type == 'classification':
                selector = SelectKBest(score_func=mutual_info_classif, k=number_features_to_keep)

        Xnew = selector.fit_transform(X=self.dataframe[self.x_features], y=self.dataframe[self.y_feature])

        print('feature selection scores:')
        print(selector.scores_)

        feature_names_selected = MiscFeatureSelectionOperations().get_selector_feature_names(selector=selector, x_features=self.x_features)
        dataframe = DataframeUtilities()._array_to_dataframe(array=Xnew)
        dataframe = DataframeUtilities()._assign_columns_as_features(dataframe=dataframe, x_features=feature_names_selected, y_feature=self.y_feature, remove_first_row=False)
        # Add y_feature back into the dataframe
        dataframe = FeatureIO(dataframe=dataframe).add_custom_features(features_to_add=[self.y_feature],data_to_add=self.dataframe[self.y_feature])

        mfso = MiscFeatureSelectionOperations()
        filetag = mfso.get_feature_filetag(configdict=self.configdict, dataframe=dataframe)
        mfso.save_data_to_csv(configdict=self.configdict, dataframe=dataframe, feature_selection_str='input_with_univariate_feature_selection', filetag=filetag)

        dataframe = dataframe.dropna()
        return dataframe

    """
    def recursive_feature_elimination(self, number_features_to_keep, save_to_csv=True):
        if self.selection_type == 'Regression' or self.selection_type == 'regression':
            estimator = SVR(kernel='linear')
        if self.selection_type == 'Classification' or self.selection_type == 'classification':
            estimator = SVC(kernel='linear')
        selector = RFE(estimator=estimator, n_features_to_select=number_features_to_keep)

        Xnew = selector.fit_transform(X=self.dataframe[self.x_features], y=self.dataframe[self.y_feature])
        feature_names_selected = MiscFeatureSelectionOperations().get_selector_feature_names(selector=selector, x_features=self.x_features)
        #feature_names_selected = MiscOperations().get_ranked_feature_names(selector=selector, x_features=self.x_features, number_features_to_keep=number_features_to_keep)
        dataframe = DataframeUtilities()._array_to_dataframe(array=Xnew)
        dataframe = DataframeUtilities()._assign_columns_as_features(dataframe=dataframe, x_features=feature_names_selected, y_feature=self.y_feature, remove_first_row=False)
        # Add y_feature back into the dataframe
        dataframe = FeatureIO(dataframe=dataframe).add_custom_features(features_to_add=[self.y_feature], data_to_add=self.dataframe[self.y_feature])
        if save_to_csv == bool(True):
            dataframe.to_csv('input_with_RFE_feature_selection.csv', index=False)
        return dataframe

    def stability_selection(self, number_features_to_keep, save_to_csv=True):
        # First remove features containing strings before doing feature selection
        #x_features, dataframe = MiscOperations().remove_features_containing_strings(dataframe=self.dataframe,x_features=self.x_features)
        if self.selection_type == 'Regression' or self.selection_type == 'regression':
            selector = RandomizedLasso()
            selector.fit(self.dataframe[self.x_features], y=self.dataframe[self.y_feature])
            feature_names_selected = MiscFeatureSelectionOperations().get_ranked_feature_names(selector=selector, x_features=self.x_features, number_features_to_keep=number_features_to_keep)
            dataframe = FeatureIO(dataframe=self.dataframe).keep_custom_features(features_to_keep=feature_names_selected, y_feature=self.y_feature)
        if self.selection_type == 'Classification' or self.selection_type == 'classification':
            print('Stability selection is currently only configured for regression tasks')
            sys.exit()
        if save_to_csv == bool(True):
            dataframe.to_csv('input_with_stability_feature_selection.csv', index=False)
        return dataframe
    """



class MiscFeatureSelectionOperations():

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
    def get_forward_selection_feature_names(cls, feature_indices_selected, x_features):
        # Get the names of the features based on their indices, for features selected from feature selection
        feature_names_selected = []
        for i in range(len(x_features)):
            if i in feature_indices_selected:
                feature_names_selected.append(x_features[i])
        return feature_names_selected

    @classmethod
    def get_feature_filetag(cls, configdict, dataframe):
        foundfeature = False
        for column in dataframe.columns.values:
            if column in configdict['General Setup']['target_feature']:
                filetag = column
                foundfeature = True
        if foundfeature == False:
            logging.info('Error: Could not locate y_feature in your dataframe, please ensure the y_feature names match in your csv'
                      'and input file')
            sys.exit()
        return filetag

    @classmethod
    def get_forward_selection_learning_curve(cls, configdict, metricdict, filetag):
        fig1 = plot_sfs(metric_dict=metricdict, kind='std_dev')
        plt.title('Forward selection learning curve', fontsize=18)
        plt.ylabel('R^2 correlation', fontsize=16)
        plt.xticks(fontsize=14)
        plt.xlabel('Number of features', fontsize=16)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(configdict['General Setup']['save_path']+"/"+'forward_selection_learning_curve_'+str(filetag)+'.pdf')
        return

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

    @classmethod
    def remove_features_containing_strings(cls, dataframe, x_features):
        x_features_pruned = []
        x_features_to_remove = []
        for x_feature in x_features:
            is_str = False
            for entry in dataframe[x_feature]:
                if type(entry) is str:
                    #print('found a string')
                    is_str = True
            if is_str == True:
                x_features_to_remove.append(x_feature)

        for x_feature in x_features:
            if x_feature not in x_features_to_remove:
                x_features_pruned.append(x_feature)

        dataframe = FeatureIO(dataframe=dataframe).remove_custom_features(features_to_remove=x_features_to_remove)
        return x_features_pruned, dataframe

    @classmethod
    def save_data_to_csv(cls, configdict, dataframe, feature_selection_str, filetag):
        dataframe.to_csv(configdict['General Setup']['save_path'] + "/" + feature_selection_str + '_' + str(filetag) + '.csv', index=False)
        return