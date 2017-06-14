__author__ = 'Ryan Jacobs'

from sklearn.decomposition import PCA
from DataOperations import DataframeUtilities
from FeatureOperations import FeatureIO
from sklearn.model_selection import learning_curve, ShuffleSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.svm import SVR, SVC
from sklearn.feature_selection import RFE
from sklearn.linear_model import RandomizedLasso, LinearRegression
from sklearn.kernel_ridge import KernelRidge
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor

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

        """
        # Do 20 LeaveOut 20% CV runs
        cv = ShuffleSplit(n_splits=20, test_size=0.2)
        #estimator = KernelRidge(alpha=0.003, coef0=1, degree=3, gamma=3.47, kernel='rbf')
        #estimator = KernelRidge(kernel='rbf')
        estimator = ExtraTreesRegressor(criterion='mse', max_depth=17, min_samples_split=2, min_samples_leaf=1, n_estimators=135)
        cv_scores = cross_val_score(estimator=estimator, X=Xnew, y=self.dataframe[self.y_feature], cv=cv, scoring='r2')
        cv_scores_mean = np.mean(cv_scores)
        cv_scores_stdev = np.std(cv_scores)
        print('CV scores:', cv_scores)
        print('CV mean:', cv_scores_mean)
        print('CV stdev:', cv_scores_stdev)
        #self.plot_learning_curve(estimator=estimator, title='Univariate feature selection learning curve', X=Xnew, y=self.dataframe[self.y_feature], cv=cv, n_jobs=1)
        """

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
            feature_names_selected = MiscOperations().get_ranked_feature_names(selector=selector, x_features=self.x_features, number_features_to_keep=number_features_to_keep)
            dataframe = FeatureIO(dataframe=self.dataframe).keep_custom_features(features_to_keep=feature_names_selected, y_feature=self.y_feature)
        if self.selection_type == 'Classification' or self.selection_type == 'classification':
            print('Stability selection is currently only configured for regression tasks')
            sys.exit()
        if save_to_csv == bool(True):
            dataframe.to_csv('input_with_stability_feature_selection.csv', index=False)
        return dataframe

    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate a simple plot of the test and training learning curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - An object to be used as a cross-validation generator.
              - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
        """
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        plt.show()
        return plt

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
