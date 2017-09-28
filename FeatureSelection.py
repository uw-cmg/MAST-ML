__author__ = 'Ryan Jacobs'

import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve, KFold
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_regression, mutual_info_classif
from sklearn.svm import SVR, SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, make_scorer
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from DataOperations import DataframeUtilities, DataParser
from FeatureOperations import FeatureIO
from MASTMLInitializer import MASTMLWrapper
from SingleFit import timeit

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
    def __init__(self, configdict, dataframe, x_features, y_feature, model_type):
        self.configdict = configdict
        self.dataframe = dataframe
        self.x_features = x_features
        self.y_feature = y_feature
        self.model_type = model_type
        # Get model to use in feature selection
        mlw = MASTMLWrapper(configdict=self.configdict)
        self.model = mlw.get_machinelearning_model(model_type=self.model_type, y_feature=self.y_feature)

    @property
    def get_original_dataframe(self):
        return self.dataframe

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

        if use_mutual_info == False or use_mutual_info == 'False':
            if selection_type == 'regression':
                if feature_selection_type == 'univariate_feature_selection':
                    selector = SelectKBest(score_func=f_regression, k=number_features_to_keep)
                elif feature_selection_type == 'recursive_feature_elimination':
                    estimator = SVR(kernel='linear')
                    selector = RFE(estimator=estimator, n_features_to_select=number_features_to_keep)
            elif selection_type == 'classification':
                if feature_selection_type == 'univariate_feature_selection':
                    selector = SelectKBest(score_func=f_classif, k=number_features_to_keep)
                elif feature_selection_type == 'recursive_feature_elimination':
                    estimator = SVC(kernel='linear')
                    selector = RFE(estimator=estimator, n_features_to_select=number_features_to_keep)
        elif use_mutual_info == True or use_mutual_info == 'True':
            if selection_type == 'regression':
                if feature_selection_type == 'univariate_feature_selection':
                    selector = SelectKBest(score_func=mutual_info_regression, k=number_features_to_keep)
                elif feature_selection_type == 'recursive_feature_elimination':
                    logging.info('Important Note: You have specified recursive feature elimination with mutual information. '
                                 'Mutual information is only used for univariate feature selection. Feature selection will still run OK')
                    estimator = SVR(kernel='linear')
                    selector = RFE(estimator=estimator, n_features_to_select=number_features_to_keep)
            elif selection_type == 'classification':
                if feature_selection_type == 'univariate_feature_selection':
                    selector = SelectKBest(score_func=mutual_info_classif, k=number_features_to_keep)
                elif feature_selection_type == 'recursive_feature_elimination':
                    logging.info('Important Note: You have specified recursive feature elimination with mutual information. '
                                 'Mutual information is only used for univariate feature selection. Feature selection will still run OK')
                    estimator = SVC(kernel='linear')
                    selector = RFE(estimator=estimator, n_features_to_select=number_features_to_keep)

        Xnew = selector.fit_transform(X=self.dataframe[self.x_features], y=self.dataframe[self.y_feature])

        mfso = MiscFeatureSelectionOperations()
        feature_indices_selected, feature_names_selected = mfso.get_selector_feature_names(selector=selector, x_features=self.x_features)
        dataframe = DataframeUtilities()._array_to_dataframe(array=Xnew)
        dataframe = DataframeUtilities()._assign_columns_as_features(dataframe=dataframe, x_features=feature_names_selected, y_feature=self.y_feature, remove_first_row=False)
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

class LearningCurve(object):

    def __init__(self, configdict, dataframe, model_type):
        self.configdict = configdict
        self.dataframe = dataframe
        self.model_type = model_type
        self.x_features, self.y_feature = DataParser(configdict=self.configdict).get_features(dataframe=self.dataframe,
                                                    target_feature=self.configdict['General Setup']['target_feature'],
                                                    from_input_file=False)
        # Get model to use in feature selection
        mlw = MASTMLWrapper(configdict=self.configdict)
        self.model = mlw.get_machinelearning_model(model_type=self.model_type, y_feature=self.y_feature)


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

                    rmse_test = np.sqrt(mean_squared_error(predict_test, target_test))
                    rmse_train = np.sqrt(mean_squared_error(predict_train, target_train))
                    fold_train_rmses[fold] = rmse_train
                    fold_test_rmses[fold] = rmse_test
                cvtest_dict[cvtest]["avg_train_rmse"] = np.mean(fold_train_rmses)
                cvtest_dict[cvtest]["train_rmse_stdev"] = np.std(fold_train_rmses)
                cvtest_dict[cvtest]["avg_test_rmse"] = np.mean(fold_test_rmses)
                cvtest_dict[cvtest]["test_rmse_stdev"] = np.std(fold_test_rmses)

            # Average rmse over all cvtests for this dataframe
            avg_train_rmse = 0
            avg_test_rmse = 0
            for cvtest in cvtest_dict.keys():
                avg_train_rmse += cvtest_dict[cvtest]["avg_train_rmse"]
                avg_test_rmse += cvtest_dict[cvtest]["avg_test_rmse"]
                train_rmse_list.append(cvtest_dict[cvtest]["avg_train_rmse"])
                test_rmse_list.append(cvtest_dict[cvtest]["avg_test_rmse"])
            avg_train_rmse /= num_cvtests
            avg_test_rmse /= num_cvtests
            avg_train_rmse_list.append(avg_train_rmse)
            avg_test_rmse_list.append(avg_test_rmse)
            train_rmse_stdev_list.append(np.mean(np.std(train_rmse_list)))
            test_rmse_stdev_list.append(np.mean(np.std(test_rmse_list)))

            # Get current df x and y data split
            ydata = FeatureIO(dataframe=dfcopy).keep_custom_features(features_to_keep=target_feature)
            Xdata = FeatureIO(dataframe=dfcopy).remove_custom_features(features_to_remove=target_feature)

            # Construct learning curve plot of CVscore vs number of training data included. Only do it once max features reached
            if num_features == n_features_to_keep:
                self.get_univariate_RFE_training_data_learning_curve(estimator=self.model, title='Training data learning curve',
                                                       Xdata=Xdata, ydata=ydata, feature_selection_type= feature_selection_algorithm, cv=5)

        # Construct learning curve plot of RMSE vs number of features included
        ydict = {"train_rmse": avg_train_rmse_list, "test_rmse": avg_test_rmse_list}
        ydict_stdev = {"train_rmse": train_rmse_stdev_list, "test_rmse": test_rmse_stdev_list}
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
        plt.ylabel("RMSE")
        train_sizes, train_scores, test_scores = learning_curve(estimator, Xdata, ydata, cv=cv, n_jobs=1,
                                                                scoring=make_scorer(score_func=mean_squared_error),
                                                                train_sizes=np.linspace(0.1, 1.0, 10))
        train_scores_mean = np.mean(np.sqrt(train_scores), axis=1)
        train_scores_std = np.std(np.sqrt(train_scores), axis=1)
        test_scores_mean = np.mean(np.sqrt(test_scores), axis=1)
        test_scores_std = np.std(np.sqrt(test_scores), axis=1)
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
            if ydataname == 'train_rmse':
                plt.plot(Xdata, ydata, 'o-', color='r', label='Training data score')
                plt.fill_between(Xdata, np.array(ydata) - np.array(ydata_stdev[ydataname]), np.array(ydata) + np.array(ydata_stdev[ydataname]), alpha=0.1,
                                 color="r")
            if ydataname == 'test_rmse':
                plt.plot(Xdata, ydata, 'o-', color='g', label='Test data score')
                plt.fill_between(Xdata, np.array(ydata) - np.array(ydata_stdev[ydataname]), np.array(ydata) + np.array(ydata_stdev[ydataname]), alpha=0.1,
                                color="g")
        plt.xlabel("Number of features")
        plt.ylabel("RMSE")
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