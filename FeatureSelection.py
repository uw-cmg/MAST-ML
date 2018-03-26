__author__ = 'Ryan Jacobs'
__maintainer__ = 'Ryan Jacobs'
__version__ = '1.0'
__email__ = 'rjacobs3@wisc.edu'
__date__ = 'October 14th, 2017'

import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve, KFold
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error, r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from DataOperations import DataframeUtilities, DataParser
from FeatureOperations import FeatureIO
from MASTMLInitializer import ModelTestConstructor
from SingleFit import timeit, SingleFit
from KFoldCV import KFoldCV
from DataHandler import DataHandler

class DimensionalReduction(object):
    """
    Class to conduct PCA and constant feature removal for dimensional reduction of features.

    Attributes:
        dataframe <pandas dataframe> : dataframe containing x and y data and feature names
        x_features <list> : list of x feature names
        y_feature <str> : target feature name

    Methods:
        remove_constant_features : removes features that have the same value for all data entries
            args:
                None
            returns:
                dataframe <pandas dataframe> : dataframe with constant features removed

        principal_component_analysis: uses principal component analysis to reduce size of feature space
            args:
                None
            returns:
                dataframe <pandas dataframe> : dataframe with PCA-selected features
    """
    def __init__(self, dataframe, x_features, y_feature):
        self.dataframe = dataframe
        self.x_features = x_features
        self.y_feature = y_feature

    def remove_constant_features(self):
        dataframe = self.dataframe.loc[:, self.dataframe.var() != 0.0]
        return dataframe

    def principal_component_analysis(self):
        pca = PCA(n_components=len(self.x_features), svd_solver='auto')
        Xnew = pca.fit_transform(X=self.dataframe[self.x_features])
        dataframe = DataframeUtilities().array_to_dataframe(array=Xnew)
        dataframe = FeatureIO(dataframe=dataframe).add_custom_features(features_to_add=[self.y_feature], data_to_add=self.dataframe[self.y_feature])
        return dataframe

class FeatureSelection(object):
    """
    Class to conduct feature selection routines to reduce the number of input features in the feature space

    Attributes:
        configdict <dict> : MASTML configfile object as dict
        dataframe <pandas dataframe> : dataframe containing x and y data and feature names
        x_features <list> : list of x feature names
        y_feature <str> : target feature name
        model_type <kwarg> : key word model string specifying model type (obtained from input file)

    Methods:
        sequential_forward_selection :
            args:
                number_of_features_keep <int> : number of features to keep after feature selection
            returns:
                dataframe <pandas dataframe> : dataframe containing only selected features

        feature_selection :
            args:
                feature_selection_type <kwarg> : type of feature selection algorithm to use. Must choose from either
                    "univariate_feature_selection" or "recursive_feature_elimination"
                number_features_to_keep <int> : number of features to keep after feature selection
                use_mutual_info <bool> : whether or not to use mutual information between features (only applicable to
                    univariate feature selection)
            returns:
                dataframe <pandas dataframe> : dataframe containing only selected features
    """
    def __init__(self, configdict, dataframe, x_features, y_feature, model_type):
        self.configdict = configdict
        self.dataframe = dataframe
        self.x_features = x_features
        self.y_feature = y_feature
        self.model_type = model_type
        # Get model to use in feature selection. If specified model doesn't have feature_importances_ attribute, use SVR by default
        mtc = ModelTestConstructor(configdict=self.configdict)
        self.model = mtc.get_machinelearning_model(model_type=self.model_type, y_feature=self.y_feature)

    def sequential_forward_selection(self, number_features_to_keep):
        sfs = SFS(self.model, k_features=number_features_to_keep, forward=True,
                  floating=False, verbose=0, scoring='neg_mean_squared_error', cv=KFold(n_splits=5, shuffle=True, random_state=False))
        sfs = sfs.fit(X=np.array(self.dataframe[self.x_features]), y=np.array(self.dataframe[self.y_feature]))
        Xnew = sfs.fit_transform(X=np.array(self.dataframe[self.x_features]), y=np.array(self.dataframe[self.y_feature]))
        feature_indices_selected = sfs.k_feature_idx_
        x_features_to_keep = []
        for index in feature_indices_selected:
            x_features_to_keep.append(self.x_features[index])

        dataframe = FeatureIO(dataframe=self.dataframe).keep_custom_features(features_to_keep=x_features_to_keep)
        # Add y_feature back into the dataframe
        dataframe = FeatureIO(dataframe=dataframe).add_custom_features(features_to_add=[self.y_feature],
                                                                       data_to_add=self.dataframe[self.y_feature])
        dataframe = dataframe.dropna()
        # Get forward selection data
        metricdict = sfs.get_metric_dict()

        # Change avg_score metric_dict values to be positive RMSE (currently negative MSE by default)
        for featurenumber, featuredict in metricdict.items():
            for metric, metricvalue in featuredict.items():
                if metric == 'avg_score':
                    metricvalue = np.sqrt(-1*metricvalue)
                    metricdict[featurenumber][metric] = metricvalue

        fs_dataframe = pd.DataFrame.from_dict(metricdict).T
        logging.info(("Summary of forward selection:"))
        logging.info(fs_dataframe)

        mfso = MiscFeatureSelectionOperations()
        filetag = mfso.get_feature_filetag(configdict=self.configdict, dataframe=dataframe)
        mfso.save_data_to_csv(configdict=self.configdict, dataframe=dataframe,
                              feature_selection_str='input_with_sequential_forward_selection', filetag=filetag)
        mfso.save_data_to_csv(configdict=self.configdict, dataframe=fs_dataframe,
                              feature_selection_str='sequential_forward_selection_data', filetag=filetag)
        learningcurve = LearningCurve(configdict=self.configdict, dataframe=dataframe, model_type=self.model_type)
        learningcurve.get_sequential_forward_selection_learning_curve(metricdict=metricdict, filetag=filetag)

        return dataframe

    def feature_selection(self, feature_selection_type, number_features_to_keep, use_mutual_info):
        if 'regression' in self.y_feature:
            selection_type = 'regression'
        elif 'classification' in self.y_feature:
            selection_type = 'classification'
        else:
            print('You must specify either "regression" or "classification" in your y_feature name')
            sys.exit()

        mtc = ModelTestConstructor(configdict=self.configdict)
        if feature_selection_type == 'recursive_feature_elimination' and self.model_type not in \
                ["linear_model_regressor", "linear_model_lasso_regressor", "support_vector_machine_regressor", "randomforest_model_regressor"]:
            self.model = SVR(kernel='linear')
            logging.info('You have specified a model type for feature selection that does not have a feature_importances_ or coef_ attribute.'
                         'The RFE method requires one of these to function.'
                         'Therefore, the model type has defaulted to an SVR model. Results should still be ok.')
        else:
            self.model = mtc.get_machinelearning_model(model_type=self.model_type, y_feature=self.y_feature)

        if use_mutual_info == False or use_mutual_info == 'False':
            if selection_type == 'regression':
                if feature_selection_type == 'univariate_feature_selection':
                    selector = SelectKBest(score_func=f_regression, k=number_features_to_keep)
                elif feature_selection_type == 'recursive_feature_elimination':
                    selector = RFE(estimator=self.model, n_features_to_select=number_features_to_keep)
                elif feature_selection_type == 'basic_forward_selection':
                    bfs = BasicForwardSelection(dataframe=self.dataframe, x_features=self.x_features, y_feature=self.y_feature,
                                                model=self.model, number_features_to_keep=number_features_to_keep, configdict=self.configdict)
                    dataframe = bfs.run_basic_forward_selection()
                else:
                    logging.info('You must specify feature_selection_type as either "univariate_feature_selection" or "recursive_feature_elimination"')
            elif selection_type == 'classification':
                if feature_selection_type == 'univariate_feature_selection':
                    selector = SelectKBest(score_func=f_classif, k=number_features_to_keep)
                elif feature_selection_type == 'recursive_feature_elimination':
                    selector = RFE(estimator=self.model, n_features_to_select=number_features_to_keep)
                else:
                    logging.info('You must specify feature_selection_type as either "univariate_feature_selection" or "recursive_feature_elimination"')
        elif use_mutual_info == True or use_mutual_info == 'True':
            if selection_type == 'regression':
                if feature_selection_type == 'univariate_feature_selection':
                    selector = SelectKBest(score_func=mutual_info_regression, k=number_features_to_keep)
                elif feature_selection_type == 'recursive_feature_elimination':
                    logging.info('Important Note: You have specified recursive feature elimination with mutual information. '
                                 'Mutual information is only used for univariate feature selection. Feature selection will still run OK')
                    selector = RFE(estimator=self.model, n_features_to_select=number_features_to_keep)
                elif feature_selection_type == 'basic_forward_selection':
                    logging.info('Important Note: You have specified basic forward selection with mutual information. '
                                 'Mutual information is only used for univariate feature selection. Feature selection will still run OK')
                    bfs = BasicForwardSelection(dataframe=self.dataframe, x_features=self.x_features,
                                            y_feature=self.y_feature, model=self.model,
                                            number_features_to_keep=number_features_to_keep,
                                            configdict=self.configdict)
                    dataframe = bfs.run_basic_forward_selection()
                else:
                    logging.info('You must specify feature_selection_type as either "univariate_feature_selection" or "recursive_feature_elimination"')
            elif selection_type == 'classification':
                if feature_selection_type == 'univariate_feature_selection':
                    selector = SelectKBest(score_func=mutual_info_classif, k=number_features_to_keep)
                elif feature_selection_type == 'recursive_feature_elimination':
                    logging.info('Important Note: You have specified recursive feature elimination with mutual information. '
                                 'Mutual information is only used for univariate feature selection. Feature selection will still run OK')
                    selector = RFE(estimator=self.model, n_features_to_select=number_features_to_keep)
                else:
                    logging.info('You must specify feature_selection_type as either "univariate_feature_selection" or "recursive_feature_elimination"')

        mfso = MiscFeatureSelectionOperations()
        if feature_selection_type != 'basic_forward_selection':
            Xnew = selector.fit_transform(X=self.dataframe[self.x_features], y=self.dataframe[self.y_feature])
            feature_indices_selected, feature_names_selected = mfso.get_selector_feature_names(selector=selector, x_features=self.x_features)
            dataframe = DataframeUtilities().array_to_dataframe(array=Xnew)
            dataframe = DataframeUtilities().assign_columns_as_features(dataframe=dataframe, x_features=feature_names_selected, y_feature=self.y_feature, remove_first_row=False)
            # Add y_feature back into the dataframe
            dataframe = FeatureIO(dataframe=dataframe).add_custom_features(features_to_add=[self.y_feature],data_to_add=self.dataframe[self.y_feature])
            dataframe = dataframe.dropna()

        # Only report the features selected and save csv file when number_features_to_keep is equal to value
        # specified in input file (so that many files aren't generated when making feature learning curve).
        if number_features_to_keep == int(self.configdict['Feature Selection']['number_of_features_to_keep']):
            filetag = mfso.get_feature_filetag(configdict=self.configdict, dataframe=dataframe)
            mfso.save_data_to_csv(configdict=self.configdict, dataframe=dataframe,
                                  feature_selection_str='input_with_'+feature_selection_type, filetag=filetag)

        return dataframe

class BasicForwardSelection(object):

    def __init__(self, dataframe, x_features, y_feature, model, number_features_to_keep, configdict):
        self.dataframe = dataframe
        self.x_features = x_features
        self.y_feature = y_feature
        self.model = model
        self.number_features_to_keep = number_features_to_keep
        self.configdict = configdict

    def run_basic_forward_selection(self):
        selected_feature_names = list()
        selected_feature_avg_rmses = list()
        selected_feature_std_rmses = list()
        basic_forward_selection_dict = dict()
        num_features_selected = 0
        x_features = self.x_features
        while num_features_selected < self.number_features_to_keep:
            # Catch pandas warnings here
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ranked_features = self._rank_features(x_features=x_features, other_features_to_keep=selected_feature_names)
                top_feature_name, top_feature_avg_rmse, top_feature_std_rmse = self._choose_top_feature(ranked_features=ranked_features)

            selected_feature_names.append(top_feature_name)
            selected_feature_avg_rmses.append(top_feature_avg_rmse)
            selected_feature_std_rmses.append(top_feature_std_rmse)

            x_features.remove(top_feature_name)
            basic_forward_selection_dict[str(num_features_selected)] = dict()
            basic_forward_selection_dict[str(num_features_selected)]['Number of features selected'] = num_features_selected+1
            basic_forward_selection_dict[str(num_features_selected)]['Top feature added this iteration'] = top_feature_name
            basic_forward_selection_dict[str(num_features_selected)]['Avg RMSE using top features'] = top_feature_avg_rmse
            basic_forward_selection_dict[str(num_features_selected)]['Stdev RMSE using top features'] = top_feature_std_rmse
            num_features_selected += 1
        basic_forward_selection_dict[str(self.number_features_to_keep-1)]['Full feature set Names'] = selected_feature_names
        basic_forward_selection_dict[str(self.number_features_to_keep - 1)]['Full feature set Avg RMSEs'] = selected_feature_avg_rmses
        basic_forward_selection_dict[str(self.number_features_to_keep - 1)]['Full feature set Stdev RMSEs'] = selected_feature_std_rmses
        self._save_featureselected_data(basic_forward_selection_dict=basic_forward_selection_dict)
        self._plot_featureselected_learningcurve(selected_feature_avg_rmses=selected_feature_avg_rmses, selected_feature_std_rmses=selected_feature_std_rmses)
        dataframe = self._get_featureselected_dataframe(selected_feature_names=selected_feature_names)

        return dataframe

    def _rank_features(self, x_features, other_features_to_keep):
        ranked_features = dict()
        for x_feature in x_features:
            features_to_keep = list()
            for feature in other_features_to_keep:
                features_to_keep.append(feature)
            features_to_keep.append(x_feature)
            dataframe = FeatureIO(dataframe=self.dataframe).keep_custom_features(features_to_keep=features_to_keep)
            dh = DataHandler(data=self.dataframe, input_data=dataframe, target_data=self.dataframe[self.y_feature],
                             input_features=x_feature, target_feature=self.y_feature)
            #sf = SingleFit(training_dataset=dh, testing_dataset=dh, model=self.model)
            #sf.fit()
            #sf.predict()
            #rmse = sf.statistics['rmse']

            kfold = KFoldCV(training_dataset=dh, testing_dataset=dh, model=self.model, num_cvtests=10, num_folds=5)
            kfold.set_up_cv()
            kfold.cv_fit_and_predict()
            kfold.get_statistics()

            avg_rmse = kfold.statistics['avg_fold_avg_rmses']
            std_rmse = kfold.statistics['std_fold_avg_rmses']

            ranked_features[x_feature] = {"avg_rmse": avg_rmse, "std_rmse": std_rmse}

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

    def _get_featureselected_dataframe(self, selected_feature_names):
        # Return dataframe containing only selected features
        fio = FeatureIO(dataframe=self.dataframe)
        dataframe = fio.keep_custom_features(features_to_keep=selected_feature_names, y_feature=self.y_feature)
        return dataframe

    def _save_featureselected_data(self, basic_forward_selection_dict):
        dataframe = pd.DataFrame(basic_forward_selection_dict)
        dataframe.to_csv(self.configdict['General Setup']['save_path']+'/'+'output_basic_forward_selection.csv')
        return

    def _plot_featureselected_learningcurve(self, selected_feature_avg_rmses, selected_feature_std_rmses):
        plt.figure()
        plt.title('Basic forward selection learning curve')
        plt.grid()
        savedir = self.configdict['General Setup']['save_path']
        Xdata = np.linspace(start=1, stop=self.number_features_to_keep, num=self.number_features_to_keep).tolist()
        ydata = selected_feature_avg_rmses
        yspread = selected_feature_std_rmses
        plt.plot(Xdata, ydata, '-o', color='r', label='Avg RMSE 10 tests 5-fold CV')
        plt.fill_between(Xdata, np.array(ydata) - np.array(yspread),
                         np.array(ydata) + np.array(yspread), alpha=0.1,
                         color="r")
        plt.xlabel("Number of features")
        plt.ylabel("RMSE")
        plt.legend(loc="best")
        plt.savefig(savedir + "/" + "basic_forwards_selection_learning_curve_featurenumber.pdf")
        return

class LearningCurve(object):
    """
    Class to construct learning curves to assess feature selection choices

    Attributes:
        configdict <dict> : MASTML configfile object as dict
        dataframe <pandas dataframe> : dataframe containing x and y data and feature names
        model_type <kwarg> : key word model string specifying model type (obtained from input file)

    Methods:
        generate_feature_learning_curve : generates feature-based learning curve for a specific feature selection routine
            args:
                feature_selection_algorithm <str> : name of feature selection routine
            returns:
                None

        get_univariate_RFE_training_data_learning_curve: generates training data learning curve for univariate or RFE
            feature selection routine
            args:
                estimator <sklearn model object> : an sklearn model used to assess model accuracy
                title <str> : Title for learning curve plot
                Xdata <pandas dataframe> : dataframe of Xdata
                ydata <pandas dataframe> : dataframe of ydata
                feature_selection_type <str> : name of feature selection routine
            returns:
                None

        get_univariate_RFE_feature_learning_curve: generates feature-based learning curve for univariate or RFE feature
            selection routine
            args:
                title <str> : Title for learning curve plot
                Xdata <pandas dataframe> : dataframe of Xdata
                ydata <pandas dataframe> : dataframe of ydata
                ydata_stdev <pandas dataframe> : dataframe of standard deviations of ydata
            returns:
                None

        get_sequential_forward_selection_learning_curve: generates feature-based learning curve for SFS algorithm
            args:
                metricdict <dict> : dict of feature selection metrics from SFS
                filetag <str> : name of target feature used to name save files
            returns:
                None
    """
    def __init__(self, configdict, dataframe, model_type):
        self.configdict = configdict
        self.dataframe = dataframe
        self.model_type = model_type
        self.x_features, self.y_feature = DataParser(configdict=self.configdict).get_features(dataframe=self.dataframe,
                                                    target_feature=self.configdict['General Setup']['target_feature'],
                                                    from_input_file=False)
        # Get model to use in feature selection
        mtc = ModelTestConstructor(configdict=self.configdict)
        self.model = mtc.get_machinelearning_model(model_type=self.model_type, y_feature=self.y_feature)
        self.scoring_metric = self.configdict['Feature Selection']['scoring_metric']

    @timeit
    def generate_feature_learning_curve(self, feature_selection_algorithm):
        n_features_to_keep = int(self.configdict['Feature Selection']['number_of_features_to_keep'])
        dataframe_fs_list = list()
        num_features_list = list()

        avg_train_rmse_list = list()
        avg_test_rmse_list = list()
        train_rmse_list = list()
        test_rmse_list = list()
        train_rmse_stdev_list = list()
        test_rmse_stdev_list = list()

        avg_train_mse_list = list()
        avg_test_mse_list = list()
        train_mse_list = list()
        test_mse_list = list()
        train_mse_stdev_list = list()
        test_mse_stdev_list = list()

        avg_train_mae_list = list()
        avg_test_mae_list = list()
        train_mae_list = list()
        test_mae_list = list()
        train_mae_stdev_list = list()
        test_mae_stdev_list = list()

        avg_train_r2_list = list()
        avg_test_r2_list = list()
        train_r2_list = list()
        test_r2_list = list()
        train_r2_stdev_list = list()
        test_r2_stdev_list = list()

        # Obtain dataframes of selected features for n_features ranging from 1 to number_of_features_to_keep
        for n_features in range(n_features_to_keep):
            num_features_list.append(n_features + 1)
            use_mutual_info = self.configdict['Feature Selection']['use_mutual_information']
            fs = FeatureSelection(configdict=self.configdict, dataframe=self.dataframe, x_features=self.x_features,
                                  y_feature=self.y_feature, model_type=self.model_type)
            dataframe_fs = fs.feature_selection(feature_selection_type=feature_selection_algorithm,
                                                                        number_features_to_keep=n_features + 1,
                                                                        use_mutual_info=use_mutual_info)
            dataframe_fs_list.append(dataframe_fs)

        num_cvtests = 10
        num_folds = 5
        kfoldcv = KFold(n_splits=num_folds, shuffle=True, random_state=False)
        target_feature = self.configdict['General Setup']['target_feature']

        # Loop over list of feature-selected dataframes and perform CV tests using general GKRR model. Save list of average CV scores for plotting
        for df in dataframe_fs_list:
            dfcopy = df
            cvtest_dict = dict()
            indices = np.arange(df.shape[0])
            num_features = df.shape[1] - 1

            # Set up CV splits for the dataframe being tested
            for cvtest in range(num_cvtests):
                cvtest_dict[cvtest] = dict()
                foldidx = 0
                for train, test in kfoldcv.split(indices):
                    fdict = dict()
                    fdict['train_index'] = train
                    fdict['test_index'] = test
                    cvtest_dict[cvtest][foldidx] = dict(fdict)
                    foldidx += 1

            # For each CV test, and each fold in each CV test, fit the model to the appropriate train and test data and get CV scores
            for cvtest in cvtest_dict.keys():
                fold_train_rmses = np.zeros(num_folds)
                fold_test_rmses = np.zeros(num_folds)
                fold_train_mses = np.zeros(num_folds)
                fold_test_mses = np.zeros(num_folds)
                fold_train_maes = np.zeros(num_folds)
                fold_test_maes = np.zeros(num_folds)
                fold_train_r2s = np.zeros(num_folds)
                fold_test_r2s = np.zeros(num_folds)
                for fold in cvtest_dict[cvtest].keys():
                    fdict = cvtest_dict[cvtest][fold]
                    input_train = df.iloc[fdict['train_index']]
                    if target_feature in input_train.columns:
                        del input_train[target_feature]

                    target_train = df[target_feature][fdict['train_index']]
                    input_test = df.iloc[fdict['test_index']]
                    if target_feature in input_test.columns:
                        del input_test[target_feature]
                    target_test = df[target_feature][fdict['test_index']]
                    self.model = self.model.fit(input_train, target_train)
                    predict_train = self.model.predict(input_train)
                    predict_test = self.model.predict(input_test)

                    mse_train = mean_squared_error(predict_train, target_train)
                    mse_test = mean_squared_error(predict_test, target_test)
                    rmse_test = np.sqrt(mse_test)
                    rmse_train = np.sqrt(mse_train)
                    mae_train = mean_absolute_error(predict_train, target_train)
                    mae_test = mean_absolute_error(predict_test, target_test)
                    r2_train = r2_score(predict_train, target_train)
                    r2_test = r2_score(predict_test, target_test)

                    fold_train_rmses[fold] = rmse_train
                    fold_test_rmses[fold] = rmse_test
                    fold_train_mses[fold] = mse_train
                    fold_test_mses[fold] = mse_test
                    fold_train_maes[fold] = mae_train
                    fold_test_maes[fold] = mae_test
                    fold_train_r2s[fold] = r2_train
                    fold_test_r2s[fold] = r2_test

                cvtest_dict[cvtest]["avg_train_rmse"] = np.mean(fold_train_rmses)
                cvtest_dict[cvtest]["train_rmse_stdev"] = np.std(fold_train_rmses)
                cvtest_dict[cvtest]["avg_test_rmse"] = np.mean(fold_test_rmses)
                cvtest_dict[cvtest]["test_rmse_stdev"] = np.std(fold_test_rmses)

                cvtest_dict[cvtest]["avg_train_mse"] = np.mean(fold_train_mses)
                cvtest_dict[cvtest]["train_mse_stdev"] = np.std(fold_train_mses)
                cvtest_dict[cvtest]["avg_test_mse"] = np.mean(fold_test_mses)
                cvtest_dict[cvtest]["test_mse_stdev"] = np.std(fold_test_mses)

                cvtest_dict[cvtest]["avg_train_mae"] = np.mean(fold_train_maes)
                cvtest_dict[cvtest]["train_mae_stdev"] = np.std(fold_train_maes)
                cvtest_dict[cvtest]["avg_test_mae"] = np.mean(fold_test_maes)
                cvtest_dict[cvtest]["test_mae_stdev"] = np.std(fold_test_maes)

                cvtest_dict[cvtest]["avg_train_r2"] = np.mean(fold_train_r2s)
                cvtest_dict[cvtest]["train_r2_stdev"] = np.std(fold_train_r2s)
                cvtest_dict[cvtest]["avg_test_r2"] = np.mean(fold_test_r2s)
                cvtest_dict[cvtest]["test_r2_stdev"] = np.std(fold_test_r2s)

            # Average scoring metric (e.g. RMSE, MAE) over all cvtests for this dataframe
            avg_train_rmse = 0
            avg_test_rmse = 0
            avg_train_mse = 0
            avg_test_mse = 0
            avg_train_mae = 0
            avg_test_mae = 0
            avg_train_rsquared = 0
            avg_test_rsquared = 0
            for cvtest in cvtest_dict.keys():
                avg_train_rmse += cvtest_dict[cvtest]["avg_train_rmse"]
                avg_test_rmse += cvtest_dict[cvtest]["avg_test_rmse"]
                avg_train_mse += cvtest_dict[cvtest]["avg_train_mse"]
                avg_test_mse += cvtest_dict[cvtest]["avg_test_mse"]
                avg_train_mae += cvtest_dict[cvtest]["avg_train_mae"]
                avg_test_mae += cvtest_dict[cvtest]["avg_test_mae"]
                avg_train_rsquared += cvtest_dict[cvtest]["avg_train_r2"]
                avg_test_rsquared += cvtest_dict[cvtest]["avg_test_r2"]
                train_rmse_list.append(cvtest_dict[cvtest]["avg_train_rmse"])
                test_rmse_list.append(cvtest_dict[cvtest]["avg_test_rmse"])
                train_mse_list.append(cvtest_dict[cvtest]["avg_train_rmse"])
                test_mse_list.append(cvtest_dict[cvtest]["avg_test_rmse"])
                train_mae_list.append(cvtest_dict[cvtest]["avg_train_rmse"])
                test_mae_list.append(cvtest_dict[cvtest]["avg_test_rmse"])
                train_r2_list.append(cvtest_dict[cvtest]["avg_train_r2"])
                test_r2_list.append(cvtest_dict[cvtest]["avg_test_r2"])
            avg_train_rmse /= num_cvtests
            avg_test_rmse /= num_cvtests
            avg_train_rmse_list.append(avg_train_rmse)
            avg_test_rmse_list.append(avg_test_rmse)
            train_rmse_stdev_list.append(np.mean(np.std(train_rmse_list)))
            test_rmse_stdev_list.append(np.mean(np.std(test_rmse_list)))

            avg_train_mse /= num_cvtests
            avg_test_mse /= num_cvtests
            avg_train_mse_list.append(avg_train_mse)
            avg_test_mse_list.append(avg_test_mse)
            train_mse_stdev_list.append(np.mean(np.std(train_mse_list)))
            test_mse_stdev_list.append(np.mean(np.std(test_mse_list)))

            avg_train_mae /= num_cvtests
            avg_test_mae /= num_cvtests
            avg_train_mae_list.append(avg_train_mae)
            avg_test_mae_list.append(avg_test_mae)
            train_mae_stdev_list.append(np.mean(np.std(train_mae_list)))
            test_mae_stdev_list.append(np.mean(np.std(test_mae_list)))

            avg_train_rsquared /= num_cvtests
            avg_test_rsquared /= num_cvtests
            avg_train_r2_list.append(avg_train_rsquared)
            avg_test_r2_list.append(avg_test_rsquared)
            train_r2_stdev_list.append(np.mean(np.std(train_r2_list)))
            test_r2_stdev_list.append(np.mean(np.std(test_r2_list)))

            # Get current df x and y data split
            ydata = FeatureIO(dataframe=dfcopy).keep_custom_features(features_to_keep=target_feature)
            Xdata = FeatureIO(dataframe=dfcopy).remove_custom_features(features_to_remove=target_feature)

            # Construct learning curve plot of CVscore vs number of training data included. Only do it once max features reached
            if num_features == n_features_to_keep:
                self.get_univariate_RFE_training_data_learning_curve(estimator=self.model, title='Training data learning curve',
                                                       Xdata=Xdata, ydata=ydata, feature_selection_type= feature_selection_algorithm, cv=5)

        # Construct learning curve plot of RMSE vs number of features included
        if self.scoring_metric == 'root_mean_squared_error':
            ydict = {"train_rmse": avg_train_rmse_list, "test_rmse": avg_test_rmse_list}
            ydict_stdev = {"train_rmse": train_rmse_stdev_list, "test_rmse": test_rmse_stdev_list}
        elif self.scoring_metric == 'mean_squared_error':
            ydict = {"train_mse": avg_train_mse_list, "test_mse": avg_test_mse_list}
            ydict_stdev = {"train_mse": train_mse_stdev_list, "test_mse": test_mse_stdev_list}
        elif self.scoring_metric == 'mean_absolute_error':
            ydict = {"train_mae": avg_train_mae_list, "test_mae": avg_test_mae_list}
            ydict_stdev = {"train_mae": train_mae_stdev_list, "test_mae": test_mae_stdev_list}
        elif self.scoring_metric == 'r2_score':
            ydict = {"train_r2": avg_train_r2_list, "test_r2": avg_test_r2_list}
            ydict_stdev = {"train_r2": train_r2_stdev_list, "test_r2": test_r2_stdev_list}
        else:
            logging.info("ERROR: you must specify a valid scoring metric for feature selection!")
            sys.exit()
        self.get_univariate_RFE_feature_learning_curve(title=feature_selection_algorithm + ' learning curve',
                                                       Xdata=num_features_list, ydata=ydict, ydata_stdev=ydict_stdev,
                                                       feature_selection_type=feature_selection_algorithm)
        return

    def get_univariate_RFE_training_data_learning_curve(self, estimator, title, Xdata, ydata, feature_selection_type, cv=None):
        plt.figure()
        plt.title(title)
        plt.grid()
        savedir = self.configdict['General Setup']['save_path']
        plt.xlabel("Number of training data points")
        if self.scoring_metric == 'root_mean_squared_error':
            plt.ylabel("RMSE")
            score_func = mean_squared_error
        elif self.scoring_metric == 'mean_squared_error':
            plt.ylabel('MSE')
            score_func = mean_squared_error
        elif self.scoring_metric == 'mean_absolute_error':
            plt.ylabel('MAE')
            score_func = mean_absolute_error
        elif self.scoring_metric == 'r2_score':
            plt.ylabel('R^2')
            score_func = r2_score

        train_sizes, train_scores, test_scores = learning_curve(estimator, Xdata, ydata, cv=cv, n_jobs=1,
                                                                scoring=make_scorer(score_func=score_func),
                                                                train_sizes=np.linspace(0.1, 1.0, 10))

        if self.scoring_metric == 'root_mean_squared_error':
            train_scores_mean = np.mean(np.sqrt(train_scores), axis=1)
            train_scores_std = np.std(np.sqrt(train_scores), axis=1)
            test_scores_mean = np.mean(np.sqrt(test_scores), axis=1)
            test_scores_std = np.std(np.sqrt(test_scores), axis=1)
        else:
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training data score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Test data score")
        plt.legend(loc="best")
        plt.savefig(savedir + "/" + feature_selection_type + "_learning_curve_trainingdata.pdf")

        return

    def get_univariate_RFE_feature_learning_curve(self, title, Xdata, ydata, ydata_stdev, feature_selection_type=None):
        plt.figure()
        plt.title(title)
        plt.grid()
        savedir = self.configdict['General Setup']['save_path']
        for ydataname, ydata in ydata.items():
            if 'train' in ydataname:
                plt.plot(Xdata, ydata, 'o-', color='r', label='Training data score')
                plt.fill_between(Xdata, np.array(ydata) - np.array(ydata_stdev[ydataname]), np.array(ydata) + np.array(ydata_stdev[ydataname]), alpha=0.1,
                                 color="r")
            if 'test' in ydataname:
                plt.plot(Xdata, ydata, 'o-', color='g', label='Test data score')
                plt.fill_between(Xdata, np.array(ydata) - np.array(ydata_stdev[ydataname]), np.array(ydata) + np.array(ydata_stdev[ydataname]), alpha=0.1,
                                color="g")
        plt.xlabel("Number of features")
        if self.scoring_metric == 'root_mean_squared_error':
            plt.ylabel("RMSE")
        elif self.scoring_metric == 'mean_squared_error':
            plt.ylabel("MSE")
        elif self.scoring_metric == 'mean_absolute_error':
            plt.ylabel("MAE")
        elif self.scoring_metric == 'r2_score':
            plt.ylabel("R^2")
        plt.legend(loc="best")
        plt.savefig(savedir + "/" + feature_selection_type + "_learning_curve_featurenumber.pdf")
        return

    def get_sequential_forward_selection_learning_curve(self, metricdict, filetag):
        fig1 = plot_sfs(metric_dict=metricdict, kind='std_dev')
        plt.title('Sequential forward selection learning curve', fontsize=18)
        plt.ylabel('RMSE', fontsize=16)
        plt.xticks(fontsize=14)
        plt.xlabel('Number of features', fontsize=16)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(self.configdict['General Setup']['save_path'] + "/" + 'sequential_forward_selection_learning_curve_' + str(
            filetag) + '.pdf')
        return

class MiscFeatureSelectionOperations():
    """
    Class containing additional functions to help with feature selection routines

    Attributes:
        None

    Methods:
        get_selector_feature_names : obtains the feature names and indices selected by a RFE algorithm
            args:
                selector <sklearn RFE object> : an instance of the RFE sklearn class
                x_features <list> : list of x feature names
            returns:
                feature_indices_selected <list> : list of feature index numbers selected
                feature_names_selected <list> : list of feature names selected

        get_forward_selection_feature_names : obtain feature names based on feature indices
            args:
                feature_indices_selected <list> : list of feature index numbers selected
                x_features <list> : list of x feature names
            returns:
                feature_names_selected <list> : list of feature names selected

        get_feature_filetag : obtain feature name to be used in saved file names
            args:
                configdict <dict> : MASTML configfile object as dict
                dataframe <pandas dataframe> : a pandas dataframe object
            return:
                filetag <str> : feature name to be used in file name

        get_ranked_feature_names : obtains ranked feature names from an RFE algorithm
            args:
                selector <sklearn RFE object> : an instance of the RFE sklearn class
                x_features <list> : list of x feature names
                number_features_to_keep <int> : number of features to keep in selected feature list
            returns:
                feature_names_selected <list> : list of feature names selected

        remove_features_containing_strings : removes feature columns whose values are strings as these can't be used in regression tasks
            args:
                dataframe <pandas dataframe> : dataframe containing data and feature names
                x_features <list> : list of x feature names
            returns:
                x_features_pruned <list> : list of x features with those features removed which contained data as strings
                dataframe <pandas dataframe> : dataframe containing data and feature names, with string features removed

        save_data_to_csv : save dataframe to csv file
            args:
                configdict <dict> : MASTML configfile object as dict
                dataframe <pandas dataframe> : a pandas dataframe object
                feature_selection_str <str> : name of feature selection routine used
                filetag <str> : name of target feature
            returns:
                None
    """
    @classmethod
    def get_selector_feature_names(cls, selector, x_features):
        feature_indices_selected = selector.get_support(indices=True)
        # Get the names of the features based on their indices, for features selected from feature selection
        feature_names_selected = []
        for i in range(len(x_features)):
            if i in feature_indices_selected:
                feature_names_selected.append(x_features[i])
        return feature_indices_selected, feature_names_selected

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