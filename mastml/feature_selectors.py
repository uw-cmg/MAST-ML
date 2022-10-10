"""
This module contains a collection of routines to perform feature selection.

BaseSelector:
    Base class to have MAST-ML like workflow functionality for feature selectors. All feature selection routines
    should inherit this base class

SklearnFeatureSelector:
    Class to wrap feature selectors from the scikit-learn package and make them have functionality from
    BaseSelector. Any scikit-learn feature selector from sklearn.feature_selection can be used by providing the
    name of the selector class as a string.

NoSelect:
    Class that performs no feature selection and just uses all features in the dataset. Needed as a placeholder
    when evaluating data splits in a MAST-ML run where feature selection is not performed.

EnsembleModelFeatureSelector:
    Class to selects features based on the feature importances scores obtained when fitting an ensemble-based model.
    Any model with the feature_importances_ attribute will work, e.g. sklearn's RandomForestRegressor and
    GradientBoostingRegressor.

PearsonSelector:
    Class that selects features based on their Pearson correlation score with the target data. Can also be used
    to assess Pearson correlation between features for use to reduce dimensionality of the feature space.

MASTMLFeatureSelector:
    Class written for MAST-ML to perform more flexible forward selection than what can be found in scikit-learn.
    Allows the user to specify a particular model and cross validation routine for selecting features, as well as the
    ability to forcibly select certain features on the outset.

ShapFeatureSelector:
    Class to select features based on how much each of the features contribute to the model in predicting the target data.


"""

import copy
import os
import warnings
import shap
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

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

                file_extension: (str), must be either '.xlsx' or '.csv', determines data file type for saving

            Returns:

                X_select (dataframe), dataframe of selected X features

    '''

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def evaluate(self, X, y, savepath=None, make_new_dir=False, file_extension='.csv'):
        if savepath is None:
            savepath = os.getcwd()
        self.fit(X=X, y=y)
        X_select = self.transform(X=X)
        if make_new_dir is True:
            splitdir = self._setup_savedir(selector=self, savepath=savepath)
            self.splitdir = splitdir
            savepath = splitdir
        self.selected_features = X_select.columns.tolist()
        with open(os.path.join(savepath, 'selected_features.txt'), 'w') as f:
            for feature in self.selected_features:
                f.write(str(feature) + '\n')
        if file_extension == '.xlsx':
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

            if self.__class__.__name__ == 'ShapFeatureSelector':
                self.feature_imp_shap.to_excel(
                    os.path.join(savepath, 'ShapFeatureSelector_sorted_features.xlsx'))
                if (self.make_plot == True):
                    shap.plots.beeswarm(self.shap_values, max_display=self.max_display, show=False)
                    plt.savefig(os.path.join(savepath, 'SHAP_features_selected.png'), dpi = 150, bbox_inches = "tight")
        elif file_extension == '.csv':
            if self.__class__.__name__ == 'EnsembleModelFeatureSelector':
                self.feature_importances_sorted.to_csv(
                    os.path.join(savepath, 'EnsembleModelFeatureSelector_feature_importances.csv'))
            if self.__class__.__name__ == 'PearsonSelector':
                self.full_correlation_matrix.to_csv(
                    os.path.join(savepath, 'PearsonSelector_fullcorrelationmatrix.csv'))
                self.highly_correlated_features.to_csv(
                    os.path.join(savepath, 'PearsonSelector_highlycorrelatedfeatures.csv'))
                self.highly_correlated_features_flagged.to_csv(
                    os.path.join(savepath, 'PearsonSelector_highlycorrelatedfeaturesflagged.csv'))
                self.features_highly_correlated_with_target.to_csv(
                    os.path.join(savepath, 'PearsonSelector_highlycorrelatedwithtarget.csv'))
            if self.__class__.__name__ == 'MASTMLFeatureSelector':
                self.mastml_forward_selection_df.to_csv(
                    os.path.join(savepath, 'MASTMLFeatureSelector_featureselection_data.csv'))

            if self.__class__.__name__ == 'ShapFeatureSelector':
                self.feature_imp_shap.to_csv(
                    os.path.join(savepath, 'ShapFeatureSelector_sorted_features.csv'))
                if (self.make_plot == True):
                    shap.plots.beeswarm(self.shap_values, max_display=self.max_display, show=False)
                    plt.savefig(os.path.join(savepath, 'SHAP_features_selected.png'), dpi=150, bbox_inches="tight")
        if file_extension == '.xlsx':
            X_select.to_excel(os.path.join(savepath, 'selected_features.xlsx'), index=False)
        elif file_extension == '.csv':
            X_select.to_csv(os.path.join(savepath, 'selected_features.csv'), index=False)

        return X_select

    def _setup_savedir(self, selector, savepath):
        now = datetime.now()
        dirname = selector.__class__.__name__
        dirname = f"{dirname}_{now.year:02d}_{now.month:02d}_{now.day:02d}" \
                  f"_{now.hour:02d}_{now.minute:02d}_{now.second:02d}"
        if savepath == None:
            splitdir = os.getcwd()
        else:
            splitdir = os.path.join(savepath, dirname)
        if not os.path.exists(splitdir):
            os.mkdir(splitdir)
        self.splitdir = splitdir
        return splitdir


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
        for new_col in new_cols:
            for original_col in original_cols:
                if np.array_equal(X[original_col].values, X_select[new_col].values):
                    new_cols_renamed.append(original_col)
                    break

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

        n_random_dummy: (int), the number of random dummy variable to use. default is 0 if not used

        n_permuted_dummy: (int), the number of permuted dummy variable to use. default is 0 if not used


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

        create_dummy_variable: Inserts n_dummy_variable of dummy variables with the same standard deviation and mean of
                               of the whole dataframe

            Args:
                X: (dataframe), dataframe of X features

            Returns:
                X: dataframe that includes dummy variables and scaled with standard scaler

        check_dummy_ranking: If dummy variable is used, prints warning when number of features selected
                             is not optimal (numbers of features selected ranks below the dummy variable)

            Args:
                feature_importances_sorted: list of features sorted based on their importances

    """

    def __init__(self, model, n_features_to_select, n_random_dummy=0, n_permuted_dummy=0):
        super(EnsembleModelFeatureSelector, self).__init__()
        self.model = model
        self.n_features_to_select = n_features_to_select
        self.n_random_dummy = n_random_dummy
        self.n_permuted_dummy = n_permuted_dummy
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
        shapeX = X.shape
        # Create random dummy variables based on mean and standard deviation of data
        if self.n_random_dummy != 0:
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
            for i in range(self.n_random_dummy):
                dummyValue = np.random.normal(loc=mean, scale=std, size=(len(X),))
                dummyName = "r_dummy" + str(i)
                X.insert(loc=0, column=dummyName, value=dummyValue)

        # Create permuted dummy variables
        if self.n_permuted_dummy != 0:
            # Checks if the n_permuted_dummy exceeds the number of features in our data
            if self.n_permuted_dummy > shapeX[1]:
                print("Warning: User input", self.n_permuted_dummy, "permuted dummies but data only consist of ",
                      shapeX[1], "features. Using ", shapeX[1], "dummies instead")
            random_features = X.sample(self.n_permuted_dummy, axis=1)
            for i in random_features:
                dummyValue = np.random.permutation(random_features[i])
                dummyName = "p_dummy" + str(dummyValue)
                X.insert(loc=0, column=dummyName, value=dummyValue)

        self.Xcol = X.columns
        return X

    def check_dummy_ranking(self, feature_importances_sorted):
        feature_list = [(feature_importances_sorted[i][1], feature_importances_sorted[i][0]) for i in
                        range(len(feature_importances_sorted))]
        feature_list_no_dummies = [a for a in feature_list if a[0][1:7] != "_dummy"]

        # Checks for random dummy scores
        if self.n_random_dummy != 0:
            random_dummy_score = [i[1] for i in feature_list if i[0][0:7] == "r_dummy"]
            mean_random_dummy_score = sum(random_dummy_score) / len(random_dummy_score)
            count = 0
            while feature_list_no_dummies[count][1] > mean_random_dummy_score:
                count += 1
            if count < self.n_features_to_select:
                print("Warning: The average of dummy variable ranks at ", count, "/", len(feature_list_no_dummies),
                      "but number of feature to select is", self.n_features_to_select)

        # Checks for permuted dummy scores
        if self.n_permuted_dummy != 0:
            permuted_dummy_score = [i[1] for i in feature_list if i[0][0:7] == "p_dummy"]
            mean_permuted_dummy_score = sum(permuted_dummy_score) / len(permuted_dummy_score)
            count = 0
            while feature_list_no_dummies[count][1] > mean_permuted_dummy_score:
                count += 1
            if count < self.n_features_to_select:
                print("Warning: The average of permuted dummy variable ranks at ", count, "/",
                      len(feature_list_no_dummies),
                      "but number of feature to select is", self.n_features_to_select)

    def fit(self, X, y):
        # Have to save the columns name in case standard scaler is used
        self.Xcol = X.columns.tolist()
        # Add dummy variables into the data if stated by user
        if self.n_random_dummy != 0 or self.n_permuted_dummy != 0:
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
        if self.n_random_dummy != 0 or self.n_permuted_dummy != 0:
            self.check_dummy_ranking(feature_importances_sorted)

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
        threshold_between_features: (float), the threshold to decide whether redundant features are removed. Should
        be a decimal value between 0 and 1. Only used if remove_highly_correlated_features is True

        threshold_with_target: (float), the threshold to decide whether a given feature is sufficiently correlated
        with the target feature and thus kept as a selected feature. Should be a decimal value between 0 and 1.

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

        manually_selected_features: (list), a list of features manually set by the user. The feature selector will
        first start from this list of features and sequentially add features until n_features_to_select is met.

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

class ShapFeatureSelector(BaseSelector):
    """
        Class custom-written for MAST-ML to conduct selection of features with SHAP

        Args:
            model: (mastml.models object), a MAST-ML compatable model

            n_features_to_select: (int), the number of features to select

            make_plot: Saves the plot of SHAP value if True, default is False

            max_display: maximum number of feature to display in the plot

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

    def __init__(self, model, n_features_to_select, make_plot = False, max_display = 10):
        super(ShapFeatureSelector, self).__init__()
        self.model = model
        self.make_plot = make_plot
        self.n_features_to_select = n_features_to_select
        self.max_display = max_display

    def fit(self, X, y):
        Xcol = X.columns.tolist()
        self.model = self.model.fit(X,y)
        explainer = shap.Explainer(self.model)
        self.shap_values = explainer(X)

        feature_order = np.argsort(np.sum(np.abs(self.shap_values.values), axis=0))
        feature_order_reversed = [k for k in reversed(feature_order)]
        self.feature_imp_shap = []
        for i in feature_order_reversed:
            self.feature_imp_shap.append(Xcol[i])
        self.selected_features = self.feature_imp_shap[:self.n_features_to_select]
        self.feature_imp_shap = pd.DataFrame(self.feature_imp_shap)
        return self

    def transform(self, X):
        X_select = X[self.selected_features]
        return X_select

def selected_features_correlation(X, savepath, features_x_path, features_y_path):
    '''
    Function to get the correlation between two sets of features selected from two different methods of feature selection

    Args:
        X: (pd.DataFrame), dataframe of X features

        savepath: (str), string denoting the path to save output to

        features_x_path: (str), string denoting the path to the first selected_features.txt

        features_y_path: (str), string denoting the path to the second selected_features.txt

    Returns:
        None.

    '''
    with open(os.path.join(features_x_path, 'selected_features.txt')) as f:
        x_selected_features = [line.rstrip() for line in f]

    with open(os.path.join(features_y_path, 'selected_features.txt')) as f:
        y_selected_features = [line.rstrip() for line in f]

    array_data = list()
    for i in range(len(x_selected_features)):
        col_data = X[x_selected_features].iloc[:,i]
        col = list()
        for j in range(len(y_selected_features)):
            row_data = X[y_selected_features].iloc[:,j]
            corr, _ = pearsonr(row_data, col_data)
            col.append(corr)
        array_data.append(col)
    array_df = pd.DataFrame(array_data, index=x_selected_features[:len(x_selected_features)], 
                            columns=y_selected_features[:len(y_selected_features)])
    array_df.to_excel(os.path.join(savepath, 'pearson')+'.xlsx', index=True)
    hCorr = dict()
    same_features = list()
    for i in range(len(array_df)):
        for j in range(len(array_df.iloc[0, :])):
            if (abs(array_df.iloc[i][j]) > 0.7):
                if (array_df.index[i] != array_df.columns[j]):
                    if ((array_df.columns[j], array_df.index[i]) not in hCorr):
                        hCorr[(array_df.index[i], array_df.columns[j])] = array_df.iloc[i][j]
                else:
                    if array_df.index[i] not in same_features:
                        same_features.append(array_df.index[i])

    hCorr_sorted = sorted(hCorr.items(), key=lambda x: (x[1]), reverse=True)
    arr = []
    for i in hCorr_sorted:
        arr.append((i[0][0], i[0][1], i[1]))
    arr_df = pd.DataFrame(arr, columns=['feature_1', 'feature_2', 'correlation'])
    arr_df.to_excel(os.path.join(savepath, 'related_features') + '.xlsx', index=True)

    with open(os.path.join(savepath, 'same_features.txt'), 'w') as f:
        for feature in same_features:
            f.write(str(feature) + '\n')