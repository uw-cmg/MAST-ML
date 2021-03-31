"""
This module contains a collection of classes and methods for selecting features, and interfaces with scikit-learn
feature
selectors. More information on scikit-learn feature selectors is available at:

http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection
"""

import copy
import os
import warnings

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from mastml.metrics import root_mean_squared_error


class BaseSelector(BaseEstimator, TransformerMixin):
    '''
    Base class that forms foundation of MAST-ML feature selectors

    Args:

        None. See individual selector types for input arguments

    Methods:

        fit: Does nothing, present for compatibility

            Args:

                X: (dataframe), dataframe of X features

                y: (dataframe), dataframe of y data


            Returns:

                None

        transform: Does nothing, present for compatibility

            Args:

                X: (dataframe), dataframe of X features

            Returns:

                X: (dataframe), dataframe of X features

        evaluate: runs the fit and transform functions to select features, saves selector-specific files and saves
        list of selected features

            Args:

                X: (dataframe), dataframe of X features

                y: (dataframe), dataframe of y data

                savepath: (str), string denoting savepath to save selected features and associated files (if
                applicable) to.

            Returns:

                X_select (dataframe), dataframe of selected X features

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
                f.write(str(feature) + '\n')
        if self.__class__.__name__ == 'EnsembleModelFeatureSelector':
            self.feature_importances_sorted.to_excel(
                os.path.join(savepath, 'EnsembleModelFeatureSelector_feature_importances.xlsx'))
        if self.__class__.__name__ == 'PearsonSelector':
            self.full_correlation_matrix.to_excel(os.path.join(savepath, 'PearsonSelector_fullcorrelationmatrix.xlsx'))
            self.highly_correlated_features.to_excel(
                os.path.join(savepath, 'PearsonSelector_highlycorrelatedfeatures.xlsx'))
            self.highly_correlated_features_flagged.to_excel(
                os.path.join(savepath, 'PearsonSelector_highlycorrelatedfeaturesflagged.xlsx'))
            self.features_highly_correlated_with_target.to_excel(
                os.path.join(savepath, 'PearsonSelector_highlycorrelatedwithtarget.xlsx'))
        if self.__class__.__name__ == 'MASTMLFeatureSelector':
            self.mastml_forward_selection_df.to_excel(
                os.path.join(savepath, 'MASTMLFeatureSelector_featureselection_data.xlsx'))
        return X_select


class SklearnFeatureSelector(BaseSelector):
    '''
    Class that wraps scikit-learn feature selection methods with some new MAST-ML functionality

    Args:

        selector (str) : a string denoting the name of a sklearn.feature_selection object

        **kwargs: the key word arguments of the designated sklearn.feature_selection object

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

                X_select: (dataframe), dataframe of selected X features

    '''

    def __init__(self, selector, **kwargs):
        super(SklearnFeatureSelector, self).__init__()
        self.selector = getattr(sklearn.feature_selection, selector)(**kwargs)

        # TODO: map string input of estimator (e.g. for SequentialFeatureSelector) and score_func (e.g. for
        #  SelectKBest) to be objects

    def fit(self, X, y):
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        X_select = pd.DataFrame(self.selector.transform(X))
        original_cols = X.columns.tolist()
        new_cols = X_select.columns.tolist()
        new_cols_renamed = list()
        for original_col in original_cols:
            for new_col in new_cols:
                # Need to compare as array because indicies are different
                if np.array_equal(X[original_col].values, X_select[new_col].values):
                    new_cols_renamed.append(original_col)

        X_select.columns = new_cols_renamed
        return X_select


class NoSelect(BaseSelector):
    """
    Class for having a "null" transform where the output is the same as the input. Needed by MAST-ML as a placeholder if
    certain workflow aspects are not performed.

    See BaseSelector for information on args and methods

    """

    def __init__(self):
        super(NoSelect, self).__init__()


class EnsembleModelFeatureSelector(BaseSelector):
    """
    Class custom-written for MAST-ML to conduct selection of features with ensemble model feature importances

    Args:

        model: (mastml.models object), a MAST-ML compatable model

        n_features_to_select: (int), the number of features to select

        n_dummy_variable: (int), the number of dummy variable to use. default is 0 if not used

    Methods:

        fit: performs feature selection. if dummy variable is used, may print warning when number of features selected
             is not optimal (numbers of features selected ranks below the dummy variable)

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

        create_dummy_variable: Inserts n_dummy_variable of dummy variables with the same standard deviation and mean of
                               of the whole dataframe

            Args:
                X: (dataframe), dataframe of X features

            Returns:
                X: dataframe that includes dummy variables and scaled with standard scaler

    """

    def __init__(self, model, n_features_to_select, n_dummy_variable=0):
        super(EnsembleModelFeatureSelector, self).__init__()
        self.model = model
        self.n_features_to_select = n_features_to_select
        self.n_dummy_variable = n_dummy_variable
        # Check that a correct model was passed in
        self._check_model()
        self.selected_features = list()

    def _check_model(self):
        if self.model.__class__.__name__ == 'SklearnModel':
            if self.model.model.__class__.__name__ not in ['RandomForestRegressor', 'ExtraTreesRegressor',
                                                           'GradientBoostingRegressor']:
                raise ValueError(
                    'Models used in EnsembleModelFeatureSelector must be one of RandomForestRegressor, '
                    'ExtraTreesRegressor, GradientBoostingRegressor')
        else:
            if self.model.__class__.__name__ not in ['RandomForestRegressor', 'ExtraTreesRegressor',
                                                     'GradientBoostingRegressor']:
                raise ValueError(
                    'Models used in EnsembleModelFeatureSelector must be one of RandomForestRegressor, '
                    'ExtraTreesRegressor, GradientBoostingRegressor')
        return

    def create_dummy_variable(self, X):
        mean = 0
        std = 0
        num = len(X.columns)
        for i in X:
            mean += X[i].mean()
            std += X[i].std()
        # The average of the mean of every column
        mean /= num
        # The average of the std of every column
        std /= num

        # Create dummy variables
        for i in range(self.n_dummy_variable):
            dummyValue = np.random.normal(loc=mean, scale=std, size=(len(X),))
            dummyName = "dummy" + str(i)
            X.insert(loc=0, column=dummyName, value=dummyValue)
        self.Xcol = X.columns
        sc = StandardScaler()
        X = sc.fit_transform(X)
        return X

    def fit(self, X, y):
        self.Xcol = X.columns.tolist()
        # Add dummy variables into the data
        if self.n_dummy_variable != 0:
            X = self.create_dummy_variable(X)

        feature_importances = self.model.fit(X, y).feature_importances_
        feature_importance_dict = dict()
        for col, f in zip(self.Xcol, feature_importances):
            feature_importance_dict[col] = f
        feature_importances_sorted = sorted(((f, col) for col, f in feature_importance_dict.items()), reverse=True)
        self.feature_importances_sorted = pd.DataFrame(feature_importances_sorted)
        sorted_features_list = [f[1] for f in feature_importances_sorted]
        self.selected_features = sorted_features_list[0:self.n_features_to_select]

        # If dummy is used, check where it ranks and prints a warning if n_features_to_select is is more than its rank
        if self.n_dummy_variable != 0:
            feature_list = []
            for i in range(len(feature_importances_sorted)):
                feature_list.append([feature_importances_sorted[i][1], feature_importances_sorted[i][0]])
            dummy_score = [i[1] for i in feature_list if i[0][0:5] == "dummy"]
            mean_dummy_score = sum(dummy_score) / len(dummy_score)
            feature_list = [a for a in feature_list if a[0][0:5] != "dummy"]
            count = 0
            while feature_list[count][1] > mean_dummy_score:
                count += 1
            if count < self.n_features_to_select:
                print("Warning: The average of dummy variable ranks at ", count, "/", len(feature_list),
                       "but number of feature to select is", self.n_features_to_select)

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

        n_features_to_select: (int), the number of features to select

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

    def __init__(self, threshold_between_features, threshold_with_target, flag_highly_correlated_features,
            n_features_to_select):
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

            # array_df.to_excel(os.path.join(savepath, 'Full_correlation_matrix.xlsx'))
            self.full_correlation_matrix = array_df

            #### Print features highly-correlated to each other into excel
            hcorr = dict()
            highly_correlated_features = list()
            for i in range(len(array_df.iloc[0, :])):  # This includes all the data in the array_df
                # feature1 = array_df.iloc[:, i] # This does not work because feature1 is not the col name but a list
                # of values
                feature1 = array_df.columns[i]
                for j in range(len(array_df.iloc[0, :])):  # This includes all the data in the array_df
                    # feature2 = array_df.iloc[:, j] # This does not work because feature2 is not the col name but a
                    # list of values
                    feature2 = array_df.columns[j]
                    if abs(array_df.iloc[i - 1, j - 1]) >= np.float64(self.threshold_between_features):
                        if i != j:  # Ignore diagonal features
                            if not (feature2, feature1) in hcorr:  # Ignore the same correlations
                                hcorr[(feature1, feature2)] = array_df.iloc[i - 1, j - 1]
                                highly_correlated_features.append(feature2)

            hcorr_df = pd.DataFrame(hcorr, index=["Corr"])
            # hcorr_df.to_excel(os.path.join(savepath, 'Highly_correlated_features.xlsx'))
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
            # removed_features_df.to_excel(os.path.join(savepath, "Highly_correlated_features_flagged.xlsx"),
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
            print(
                'WARNING: Pearson selector threshold was too high to result in selecting any features, '
                'lowering threshold to get specified feature number')
            self.threshold_with_target -= 0.05
            self.selected_features = list(all_corrs[all_corrs > self.threshold_with_target].sort_values(
                ascending=False).keys())
            if len(self.selected_features) == n_col:
                print('WARNING: Pearson selector reduce the threshold such that all features were included')
                break
            print('Pearson selector selected features with an adjusted threshold value')
        if len(self.selected_features) > self.n_features_to_select:
            self.selected_features = list(
                all_corrs[all_corrs > self.threshold_with_target].sort_values(ascending=False).keys())[
                                     :self.n_features_to_select]

        # Create the dataframe displaying the highly correlated features and the Pearson Correlations
        hcorr_with_target_df = pd.DataFrame(all_corrs,
                                            index=list(all_corrs[all_corrs > self.threshold_with_target].sort_values(
                                                ascending=False).keys()),
                                            columns=["Pearson Correlation (absolute value)"])
        # hcorr_with_target_df.to_excel(os.path.join(savepath, 'Features_highly_correlated_with_target.xlsx'))
        self.features_highly_correlated_with_target = hcorr_with_target_df

        return self

    def transform(self, X):
        X_select = X[self.selected_features]
        return X_select


class MASTMLFeatureSelector(BaseSelector):
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

    def __init__(self, model, n_features_to_select, cv=None, manually_selected_features=list()):
        super(MASTMLFeatureSelector, self).__init__()
        self.model = model
        if cv is None:
            self.cv = KFold(shuffle=True, n_splits=5)
        else:
            self.cv = cv
        self.manually_selected_features = manually_selected_features
        self.selected_features = self.manually_selected_features
        self.n_features_to_select = n_features_to_select - len(self.manually_selected_features)

    def fit(self, X, y, Xgroups=None):
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
                top_feature_name, top_feature_avg_rmse, top_feature_std_rmse = self._choose_top_feature(
                    ranked_features=ranked_features)

            self.selected_features.append(top_feature_name)
            # if len(self.selected_feature_names) > 0:
            #    print('selected features')
            #    print(self.selected_feature_names)
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
            self.mastml_forward_selection_df = pd.DataFrame(basic_forward_selection_dict)
            # pd.DataFrame(basic_forward_selection_dict).to_csv(os.path.join(savepath,
            # 'MASTMLFeatureSelector_data_feature_'+str(num_features_selected)+'.csv'))
            num_features_selected += 1
        basic_forward_selection_dict[str(self.n_features_to_select - 1)][
            'Full feature set Names'] = self.selected_features
        basic_forward_selection_dict[str(self.n_features_to_select - 1)][
            'Full feature set Avg RMSEs'] = selected_feature_avg_rmses
        basic_forward_selection_dict[str(self.n_features_to_select - 1)][
            'Full feature set Stdev RMSEs'] = selected_feature_std_rmses
        # self._plot_featureselected_learningcurve(selected_feature_avg_rmses=selected_feature_avg_rmses,
        #                                         selected_feature_std_rmses=selected_feature_std_rmses)

        return self

    def transform(self, X):
        X_select = self._get_featureselected_dataframe(X=X, selected_feature_names=self.selected_features)
        return X_select

    def _rank_features(self, X, y, groups):
        y = np.array(y).reshape(-1, 1)
        ranked_features = dict()
        trains_metrics = list()
        tests_metrics = list()
        if groups is not None:
            groups = groups.iloc[:, 0].tolist()
        for col in X.columns:
            if col not in self.selected_features:
                X_ = X.loc[:, self.selected_features]
                X__ = X.loc[:, col]
                X_ = np.array(pd.concat([X_, X__], axis=1))

                for trains, tests in self.cv.split(X_, y, groups):
                    self.model.fit(X_[trains], y[trains])
                    predict_tests = self.model.predict(X_[tests])
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
