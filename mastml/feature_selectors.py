"""
This module contains a collection of classes and methods for selecting features, and interfaces with scikit-learn feature
selectors. More information on scikit-learn feature selectors is available at:

http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection
"""

from functools import wraps
import warnings
import numpy as np
import pandas as pd
import os
import copy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import sklearn.feature_selection as fs
from sklearn.model_selection import KFold

from mlxtend.feature_selection import SequentialFeatureSelector

from scipy.stats import pearsonr

from mastml.metrics import root_mean_squared_error

def dataframify_selector(transform):
    """
    Method which transforms output of scikit-learn feature selectors from array to dataframe. Enables preservation of column names.

    Args:

        transform: (function), a scikit-learn feature selector that has a transform method

    Returns:

        new_transform: (function), an amended version of the transform method that returns a dataframe

    """

    @wraps(transform)
    def new_transform(self, df):
        if isinstance(df, pd.DataFrame):
            return df[df.columns[self.get_support(indices=True)]]
        else: # just in case you try to use it with an array ;)
            return df
    return new_transform

def dataframify_new_column_names(transform, name):
    """
    Method which transforms output of scikit-learn feature selectors to dataframe, and adds column names

    Args:

        transform: (function), a scikit-learn feature selector that has a transform method

        name: (str), name of the feature selector

    Returns:

        new_transform: (function), an amended version of the transform method that returns a dataframe

    """

    def new_transform(self, df):
        arr = transform(self, df.values)
        labels = [name+str(i) for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=labels)
    return new_transform

def fitify_just_use_values(fit):
    """
    Method which enables a feature selector fit method to operate on dataframes

    Args:

        fit: (function), a scikit-learn feature selector object with a fit method

    Returns:

        new_fit: (function), an amended version of the fit method that uses dataframes as input

    """

    def new_fit(self, X_df, y_df):
        return fit(self, X_df.values, y_df.values)
    return new_fit

#TODO: need to clean this up and add ScikitlearnSelector wrapper. Will also include SequentialFeatureSelector in latest sklearn version
#TODO: update PearsonSelector and MASTMLFeatureSelector to conform to new style with evaluate() method

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

class BaseSelector(BaseEstimator, TransformerMixin):
    '''


    '''
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def evaluate(self, X, y, savepath):
        self.fit(X=X, y=y)
        X_select = self.transform(X=X)
        self.selected_features = X_select.columns.tolist()
        with open(os.path.join(savepath, 'selected_features.txt'), 'w') as f:
            for feature in self.selected_features:
                f.write(feature+'\n')
        if self.__class__.__name__ == 'EnsembleModelFeatureSelector':
            self.feature_importances_sorted.to_excel(os.path.join(savepath, 'EnsembleModelFeatureSelector_feature_importances.xlsx'))
        if self.__class__.__name__ == 'PearsonSelector':
            self.full_correlation_matrix.to_excel(os.path.join(savepath, 'PearsonSelector_fullcorrelationmatrix.xlsx'))
            self.highly_correlated_features.to_excel(os.path.join(savepath, 'PearsonSelector_highlycorrelatedfeatures.xlsx'))
            self.highly_correlated_features_flagged.to_excel(os.path.join(savepath, 'PearsonSelector_highlycorrelatedfeaturesflagged.xlsx'))
            self.features_highly_correlated_with_target.to_excel(os.path.join(savepath, 'PearsonSelector_highlycorrelatedwithtarget.xlsx'))
        return X_select

class NoSelect(BaseSelector):
    """
    Class for having a "null" transform where the output is the same as the input. Needed by MAST-ML as a placeholder if
    certain workflow aspects are not performed.

    Args:

        None

    Methods:

        fit: does nothing, just returns object instance. Needed to maintain same structure as scikit-learn classes

        Args:

            X: (numpy array), array of X features

        transform: passes the input back out, in this case the array of X features

        Args:

            X: (numpy array), array of X features

        Returns:

            X: (numpy array), array of X features

    """

    def __init__(self):
        super(NoSelect, self).__init__()

class EnsembleModelFeatureSelector(BaseSelector):
    """
    Class custom-written for MAST-ML to conduct selection of features with ensemble model feature importances

    Args:

        model: (mastml.models object), a MAST-ML compatiable model

        k_features: (int), the number of features to select

    Methods:

        fit: performs feature selection

            Args:

                X: (dataframe), dataframe of X features

                y: (dataframe), dataframe of y data


            Returns:

                None

        transform: performs the transform to generate output of only selected features

            Args:

                X: (dataframe), dataframe of X features

            Returns:

                dataframe: (dataframe), dataframe of selected X features

    """
    def __init__(self, model, n_features_to_select):
        super(EnsembleModelFeatureSelector, self).__init__()
        self.model = model
        self.n_features_to_select = n_features_to_select
        # Check that a correct model was passed in
        self._check_model()
        self.selected_features = list()

    def _check_model(self):
        if self.model.__class__.__name__ == 'SklearnModel':
            if self.model.model.__class__.__name__ not in ['RandomForestRegressor', 'ExtraTreesRegressor',
                                                     'GradientBoostingRegressor']:
                raise ValueError(
                    'Models used in EnsembleModelFeatureSelector must be one of RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor')
        else:
            if self.model.__class__.__name__ not in ['RandomForestRegressor', 'ExtraTreesRegressor', 'GradientBoostingRegressor']:
                raise ValueError('Models used in EnsembleModelFeatureSelector must be one of RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor')
        return

    def fit(self, X, y):
        feature_importances = self.model.fit(X, y).feature_importances_
        feature_importance_dict = dict()
        for col, f in zip(X.columns.tolist(), feature_importances):
            feature_importance_dict[col] = f
        feature_importances_sorted = sorted(((f, col) for col, f in feature_importance_dict.items()), reverse=True)
        self.feature_importances_sorted = pd.DataFrame(feature_importances_sorted)
        sorted_features_list = [f[1] for f in feature_importances_sorted]
        self.selected_features = sorted_features_list[0:self.n_features_to_select]
        return self

    def transform(self, X):
        X_select = X[self.selected_features]
        return X_select

class PearsonSelector(BaseSelector):
    """
    Class custom-written for MAST-ML to conduct selection of features based on Pearson correlation coefficent between
    features and target. Can also be used for dimensionality reduction by removing redundant features highly correlated
    with each other.

    Args:

        threshold_between_features: (float), the threshold to decide whether redundant features are removed. Should be
        a decimal value between 0 and 1. Only used if remove_highly_correlated_features is True

        threshold_with_target: (float), the threshold to decide whether a given feature is sufficiently correlated with
        the target feature and thus kept as a selected feature. Should be a decimal value between 0 and 1.

        remove_highly_correlated_features: (bool), whether to remove features highly correlated with each other

        k_features: (int), the number of features to select

    Methods:

        fit: performs feature selection

            Args:

                X: (dataframe), dataframe of X features

                y: (dataframe), dataframe of y data


            Returns:

                None

        transform: performs the transform to generate output of only selected features

            Args:

                X: (dataframe), dataframe of X features

            Returns:

                dataframe: (dataframe), dataframe of selected X features

    """
    def __init__(self, threshold_between_features, threshold_with_target, flag_highly_correlated_features, n_features_to_select):
        super(PearsonSelector, self).__init__()
        self.threshold_between_features = threshold_between_features
        self.threshold_with_target = threshold_with_target
        self.flag_highly_correlated_features = flag_highly_correlated_features
        self.n_features_to_select = n_features_to_select
        self.selected_features = list()

    def fit(self, X, y):
        df = X
        df_features = df.columns.tolist()
        n_col = df.shape[1]

        if self.flag_highly_correlated_features == True:
            array_data = list()

            for i in range(n_col):
                col_data = df.iloc[:, i]
                col = list()
                for j in range(n_col):
                    row_data = df.iloc[:, j]
                    corr, _ = pearsonr(row_data, col_data)  # Pearson Correlation
                    col.append(corr)
                array_data.append(col)

            array_df = pd.DataFrame(array_data, index=df_features[:n_col], columns=df_features[:n_col])

            #array_df.to_excel(os.path.join(savepath, 'Full_correlation_matrix.xlsx'))
            self.full_correlation_matrix = array_df

            #### Print features highly-correlated to each other into excel
            hcorr = dict()
            highly_correlated_features = list()
            for i in range(len(array_df.iloc[0, :])):  # This includes all the data in the array_df
                # feature1 = array_df.iloc[:, i] # This does not work because feature1 is not the col name but a list of values
                feature1 = array_df.columns[i]
                for j in range(len(array_df.iloc[0, :])):  # This includes all the data in the array_df
                    # feature2 = array_df.iloc[:, j] # This does not work because feature2 is not the col name but a list of values
                    feature2 = array_df.columns[j]
                    if abs(array_df.iloc[i - 1, j - 1]) >= np.float64(self.threshold_between_features):
                        if i != j:  # Ignore diagonal features
                            if not (feature2, feature1) in hcorr:  # Ignore the same correlations
                                hcorr[(feature1, feature2)] = array_df.iloc[i - 1, j - 1]
                                highly_correlated_features.append(feature2)

            hcorr_df = pd.DataFrame(hcorr, index=["Corr"])
            #hcorr_df.to_excel(os.path.join(savepath, 'Highly_correlated_features.xlsx'))
            self.highly_correlated_features = hcorr_df

            highly_correlated_features = list(np.unique(np.array(highly_correlated_features)))

            #### Print the removed features and the new smaller dataframe with features removed
            # Deep copy the original dataframe
            removed_features_df = copy.deepcopy(X)
            new_df = copy.deepcopy(X)

            # Drop the features that can be removed
            all_features = list(new_df.columns)
            removed_features = list()
            for feature in all_features:
                if feature not in highly_correlated_features:
                    removed_features.append(feature)
                    new_df = new_df.drop(columns=feature)

            # Print the highly correlated features that were removed
            for feature in all_features:
                if feature not in removed_features:
                    removed_features_df = removed_features_df.drop(columns=feature)
            #removed_features_df.to_excel(os.path.join(savepath, "Highly_correlated_features_flagged.xlsx"),
            #                             index=False)
            self.highly_correlated_features_flagged = removed_features_df

            # Define self.selected_features
            remaining_features = list(new_df.columns)
        else:
            remaining_features = list(df.columns)

        # Compute Pearson correlations between each feature and target feature
        all_corrs = {}
        for i in range(len(remaining_features)):
            feature_name = df.columns[i]
            feature_data = df.iloc[:, i]
            corr, _ = pearsonr(y, feature_data)
            all_corrs[feature_name] = corr
        all_corrs = abs(pd.Series(all_corrs))

        self.selected_features = list(all_corrs[all_corrs > self.threshold_with_target].sort_values(
                                                ascending=False).keys())

        # Sometimes the specificed threshold is too high. Make it lower until at least 1 feature is selected
        while len(self.selected_features) < self.n_features_to_select:
            print('WARNING: Pearson selector threshold was too high to result in selecting any features, lowering threshold to get specified feature number')
            self.threshold_with_target -= 0.05
            self.selected_features = list(all_corrs[all_corrs > self.threshold_with_target].sort_values(
                ascending=False).keys())
            if len(self.selected_features) == n_col:
                print('WARNING: Pearson selector reduce the threshold such that all features were included')
                break
            print('Pearson selector selected features with an adjusted threshold value')
        if len(self.selected_features) > self.n_features_to_select:
            self.selected_features = list(all_corrs[all_corrs > self.threshold_with_target].sort_values(ascending=False).keys())[:self.n_features_to_select]


        # Create the dataframe displaying the highly correlated features and the Pearson Correlations
        hcorr_with_target_df = pd.DataFrame(all_corrs,
                                            index=list(all_corrs[all_corrs > self.threshold_with_target].sort_values(
                                                ascending=False).keys()),
                                            columns=["Pearson Correlation (absolute value)"])
        #hcorr_with_target_df.to_excel(os.path.join(savepath, 'Features_highly_correlated_with_target.xlsx'))
        self.features_highly_correlated_with_target = hcorr_with_target_df

        return self

    def transform(self, X):
        X_select = X[self.selected_features]
        return X_select

class MASTMLFeatureSelector():
    """
    Class custom-written for MAST-ML to conduct forward selection of features with flexible model and cv scheme

    Args:

        estimator: (scikit-learn model/estimator object), a scikit-learn model/estimator

        n_features_to_select: (int), the number of features to select

        cv: (scikit-learn cross-validation object), a scikit-learn cross-validation object

        manually_selected_features: (list), a list of features manually set by the user. The feature selector will first
        start from this list of features and sequentially add features until n_features_to_select is met.

    Methods:

        fit: performs feature selection

            Args:

                X: (dataframe), dataframe of X features

                y: (dataframe), dataframe of y data

                Xgroups: (dataframe), dataframe of group labels

            Returns:

                None

        transform: performs the transform to generate output of only selected features

            Args:

                X: (dataframe), dataframe of X features

            Returns:

                dataframe: (dataframe), dataframe of selected X features

    """

    def __init__(self, estimator, n_features_to_select, cv, manually_selected_features=list()):
        self.estimator = estimator
        if cv is None:
            self.cv = KFold(shuffle=True, n_splits=5)
        else:
            self.cv = cv
        self.manually_selected_features = manually_selected_features
        self.selected_feature_names = self.manually_selected_features
        self.n_features_to_select = n_features_to_select-len(self.manually_selected_features)

    def fit(self, X, y, savepath, Xgroups=None):
        if Xgroups is None:
            xgroups = np.zeros(len(y))
            Xgroups = pd.DataFrame(xgroups)

        selected_feature_avg_rmses = list()
        selected_feature_std_rmses = list()
        basic_forward_selection_dict = dict()
        num_features_selected = 0
        x_features = X.columns.tolist()
        if self.n_features_to_select >= len(x_features):
            self.n_features_to_select = len(x_features)
        while num_features_selected < self.n_features_to_select:

            # Catch pandas warnings here
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ranked_features = self._rank_features(X=X, y=y, groups=Xgroups)
                top_feature_name, top_feature_avg_rmse, top_feature_std_rmse = self._choose_top_feature(ranked_features=ranked_features)

            self.selected_feature_names.append(top_feature_name)
            if len(self.selected_feature_names) > 0:
                print('selected features')
                print(self.selected_feature_names)
            selected_feature_avg_rmses.append(top_feature_avg_rmse)
            selected_feature_std_rmses.append(top_feature_std_rmse)

            basic_forward_selection_dict[str(num_features_selected)] = dict()
            basic_forward_selection_dict[str(num_features_selected)][
                'Number of features selected'] = num_features_selected + 1
            basic_forward_selection_dict[str(num_features_selected)][
                'Top feature added this iteration'] = top_feature_name
            basic_forward_selection_dict[str(num_features_selected)][
                'Avg RMSE using top features'] = top_feature_avg_rmse
            basic_forward_selection_dict[str(num_features_selected)][
                'Stdev RMSE using top features'] = top_feature_std_rmse
            # Save for every loop of selecting features
            pd.DataFrame(basic_forward_selection_dict).to_csv(os.path.join(savepath,'MASTMLFeatureSelector_data_feature_'+str(num_features_selected)+'.csv'))
            num_features_selected += 1
        basic_forward_selection_dict[str(self.n_features_to_select - 1)][
            'Full feature set Names'] = self.selected_feature_names
        basic_forward_selection_dict[str(self.n_features_to_select - 1)][
            'Full feature set Avg RMSEs'] = selected_feature_avg_rmses
        basic_forward_selection_dict[str(self.n_features_to_select - 1)][
            'Full feature set Stdev RMSEs'] = selected_feature_std_rmses
        #self._plot_featureselected_learningcurve(selected_feature_avg_rmses=selected_feature_avg_rmses,
        #                                         selected_feature_std_rmses=selected_feature_std_rmses)

        return self

    def transform(self, X):
        dataframe = self._get_featureselected_dataframe(X=X, selected_feature_names=self.selected_feature_names)
        return dataframe

    def _rank_features(self, X, y, groups):
        y = np.array(y).reshape(-1, 1)
        ranked_features = dict()
        trains_metrics = list()
        tests_metrics = list()
        if groups is not None:
            groups = groups.iloc[:,0].tolist()
        for col in X.columns:
            if col not in self.selected_feature_names:
                X_ = X.loc[:, self.selected_feature_names]
                X__ = X.loc[:, col]
                X_ = np.array(pd.concat([X_, X__], axis=1))

                for trains, tests in self.cv.split(X_, y, groups):
                    self.estimator.fit(X_[trains], y[trains])
                    predict_tests = self.estimator.predict(X_[tests])
                    tests_metrics.append(root_mean_squared_error(y[tests], predict_tests))
                avg_rmse = np.mean(tests_metrics)

                std_rmse = np.std(tests_metrics)
                ranked_features[col] = {"avg_rmse": avg_rmse, "std_rmse": std_rmse}
        return ranked_features

    def _choose_top_feature(self, ranked_features):
        feature_names = list()
        feature_avg_rmses = list()
        feature_std_rmses = list()
        feature_names_sorted = list()
        feature_std_rmses_sorted = list()
        # Make dict of ranked features into list for sorting
        for k, v in ranked_features.items():
            feature_names.append(k)
            for kk, vv in v.items():
                if kk == 'avg_rmse':
                    feature_avg_rmses.append(vv)
                if kk == 'std_rmse':
                    feature_std_rmses.append(vv)

        # Sort feature lists so RMSE goes from min to max
        feature_avg_rmses_sorted = sorted(feature_avg_rmses)
        for feature_avg_rmse in feature_avg_rmses_sorted:
            for k, v in ranked_features.items():
                if v['avg_rmse'] == feature_avg_rmse:
                    feature_names_sorted.append(k)
                    feature_std_rmses_sorted.append(v['std_rmse'])

        top_feature_name = feature_names_sorted[0]
        top_feature_avg_rmse = feature_avg_rmses_sorted[0]
        top_feature_std_rmse = feature_std_rmses_sorted[0]

        return top_feature_name, top_feature_avg_rmse, top_feature_std_rmse

    def _get_featureselected_dataframe(self, X, selected_feature_names):
        # Return dataframe containing only selected features
        X_selected = X.loc[:, selected_feature_names]
        return X_selected


# Include Principal Component Analysis
PCA.transform = dataframify_new_column_names(PCA.transform, 'pca_')

# Include Sequential Forward Selector
SequentialFeatureSelector.transform = dataframify_new_column_names(SequentialFeatureSelector.transform, 'sfs_')
SequentialFeatureSelector.fit = fitify_just_use_values(SequentialFeatureSelector.fit)
model_selectors['SequentialFeatureSelector'] = SequentialFeatureSelector
name_to_constructor['SequentialFeatureSelector'] = SequentialFeatureSelector

# Custom selectors don't need to be dataframified
name_to_constructor.update({
    #'PassThrough': PassThrough,
    'PCA': PCA,
    'SequentialFeatureSelector': SequentialFeatureSelector,
    'MASTMLFeatureSelector' : MASTMLFeatureSelector,
    'PearsonSelector': PearsonSelector,
    'EnsembleModelFeatureSelector': EnsembleModelFeatureSelector
})
