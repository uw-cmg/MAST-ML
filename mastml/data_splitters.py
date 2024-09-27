"""
This module contains a collection of methods to split data into different types of train/test sets. Data splitters
are the core component to evaluating model performance.

BaseSplitter:
    Base class that handles the core MAST-ML data splitting and model evaluation workflow. This class is responsible
    for looping over provided feature selectors, models, and data splits and training and evaluating the model for each
    split, then generating the necessary plots and performance statistics. All different splitter types inherit this
    base class.

SklearnDataSplitter:
    Wrapper class to enable MAST-ML workflow compatible use of any data splitter contained in scikit-learn, e.g. KFold,
    RepeatedKFold, LeaveOneGroupOut, etc.

NoSplit:
    Class that doesn't perform any data split. Equivalent to a "full fit" of the data where all data is used in training.

JustEachGroup:
    Class that splits data so each individual group is used as training with all other groups used as testing. Essentially
    the inverse of LeaveOneGroupOut, this class trains only on one group and predicts the rest, as opposed to training
    on all but one group and testing on the left-out group.

LeaveCloseCompositionsOut:
    Class to split data based on their compositional similiarity. A useful means to separate compositionally similar
    compounds into the training or testing set, so that similar materials are not contained in both sets.

LeaveOutPercent:
    Method to randomly split the data based on fraction of total data points, rather than a designated number of splits.
    Enables one to do higher than 50% leave out (this is highest leave out possible with KFold where k=2), so can do e.g.
    leave out 90% data.

LeaveOutTwinCV:
    Another method to help separate similar data from the training and testing set. This method makes use of a general
    distance metric on the provided features, and flags twins as those data points within some provided distance threshold
    in the feature space.

LeaveOutClusterCV:
    Method to use a clustering algorithm to pre-cluster data into groups. Then, these different groups are used as each
    left-out data set. Basically functions as a leave out group test where the groups are automatically obtained from
    a clustering algorithm.

LeaveMultiGroupOut:
    Class to train the model on multiple groups at a time and test it on the rest of the data

Bootstrap:
    Method to perform bootstrap resampling, i.e. random leave-out with replacement.

"""

import numpy as np
import os
import sys
import pandas as pd
from datetime import datetime
import copy
import sklearn
import inspect
from pprint import pprint
import joblib
from math import ceil
import warnings
import shutil
import itertools
from scipy.spatial.distance import minkowski
from tqdm import tqdm
import json
import dill
from mastml.mastml import NumpyEncoder

try:
    import keras
except:
    print('Keras is an optional dependency. To use keras, do pip install keras tensorflow')
try:
    from matminer.featurizers.composition import ElementFraction
    from pymatgen.core import Composition
except:
    print('matminer and pymatgen are optional dependencies. To use data splitter methods invoking these packages,'
          'do pip install matminer pymatgen')

import sklearn.model_selection as ms
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
import sklearn_extra.cluster

from mastml.plots import make_plots
from mastml.feature_selectors import NoSelect
from mastml.error_analysis import ErrorUtils
from mastml.metrics import Metrics
from mastml.preprocessing import NoPreprocessor
from mastml.baseline_tests import Baseline_tests
from mastml.domain import Domain
from mastml.mastml import parallel
try:
    import transfernet as trans
    import torch
except:
    print('transfernet is an optional dependency. If you want to use transfer learning pytorch NNs, do'
          ' "pip install transfernet"')


class BaseSplitter(ms.BaseCrossValidator):
    """
    Class functioning as a base splitter with methods for organizing output and evaluating any mastml data splitter

    Args:
        None

    Methods:
        split_asframe: method to perform split into train indices and test indices, but return as dataframes
            Args:
                X: (pd.DataFrame), dataframe of X features

                y: (pd.Series), series of y target data

                groups: (pd.Series), series of group designations

            Returns:
                X_splits: (list), list of dataframes for X splits

                y_splits: (list), list of dataframes for y splits

        evaluate: main method to evaluate a sequence of models, selectors, and hyperparameter optimizers, build directories and perform analysis and output plots
            Args:
                X: (pd.DataFrame), dataframe of X features

                y: (pd.Series), series of y target data

                models: (list), list containing mastml.models instances

                preprocessor: (mastml.preprocessor), mastml.preprocessor object to normalize the training data in each split.

                groups: (pd.Series), series of group designations

                hyperopts: (list), list containing mastml.hyperopt instances. One for each provided model is needed.

                selectors: (list), list containing mastml.feature_selectors instances

                metrics: (list), list of metric names to evaluate true vs. pred data in each split

                plots: (list), list of names denoting which types of plots to make. Valid names are 'Scatter', 'Error', and 'Histogram'

                savepath: (str), string containing main savepath to construct splits for saving output

                X_extra: (pd.DataFrame), dataframe of extra X data not used in model fitting

                leaveout_inds: (list), list of arrays containing indices of data to be held out and evaluated using best model from set of train/validation splits

                best_run_metric: (str), metric name to be used to decide which model performed best. Defaults to first listed metric in metrics.

                nested_CV: (bool), whether to perform nested cross-validation. The nesting is done using the same splitter object as self.splitter

                error_method: (str), the type of model error evaluation method to perform. Only applies to certain models. Valid names are 'stdev_weak_learners' and 'jackknife_after_bootstrap'

                remove_outlier_learners: (bool), whether to remove weak learners from ensemble models whose predictions are found to be outliers. Default False.

                recalibrate_errors: (bool), whether to perform the predicted error bar recalibration method of Palmer et al. Default False.

                verbosity: (int), the output plotting verbosity. Default is 1. Valid choices are 0, 1, 2, and 3.

                baseline_test: (list), list of arrays containing base line tests to run.

                distance_metric: (str), distance metric to use in baseline_tests test_nearest_neighbour_cdist method

                file_extension: (str), must be either '.xlsx' or '.csv', determines data file type for saving

                image_dpi: (int), determines output image quality

                remove_split_dirs: (bool), whether to remove all the inner split directories after data and plots saved

                **kwargs: (str), extra argument for domain_distance, eg. minkowsi requires additional arg p

            Returns:
                None

        _evaluate_split_sets: method to evaluate a set of train/test splits. At the end of the split set, the left-out data (if any) is evaluated using the best model from the train/test splits
            Args:
                X_splits: (list), list of dataframes for X splits

                y_splits: (list), list of dataframes for y splits

                train_inds: (list), list of arrays of indices denoting the training data

                test_inds: (list), list of arrays of indices denoting the testing data

                model: (mastml.models instance), an estimator for fitting data

                model_name: (str), class name of the model being evaluated

                selector: (mastml.selector), a feature selector to select features in each split

                preprocessor: (mastml.preprocessor), mastml.preprocessor object to normalize the training data in each split.

                X_extra: (pd.DataFrame), dataframe of extra X data not used in model fitting

                groups: (pd.Series), series of group designations

                splitdir: (str), string denoting the split path in the save directory

                hyperopt: (mastml.hyperopt), mastml.hyperopt instance to perform model hyperparameter optimization in each split

                metrics: (list), list of metric names to evaluate true vs. pred data in each split

                plots: (list), list of names denoting which types of plots to make. Valid names are 'Scatter', 'Error', and 'Histogram'

                has_model_errors: (bool), whether the model used has error bars (uncertainty quantification)

                error_method: (str), the type of model error evaluation method to perform. Only applies to certain models. Valid names are 'stdev_weak_learners' and 'jackknife_after_bootstrap'

                remove_outlier_learners: (bool), whether to remove weak learners from ensemble models whose predictions are found to be outliers. Default False.

                recalibrate_errors: (bool), whether to perform the predicted error bar recalibration method of Palmer et al. Default False.

                verbosity: (int), the output plotting verbosity. Default is 1. Valid choices are 0, 1, 2, and 3.

                baseline_test: (list), list of arrays containing base line tests to run.

                distance_metric: (str), distance metric to use in baseline_test's test_nearest_neighbour_cdist method

                file_extension: (str), must be either '.xlsx' or '.csv', determines data file type for saving

                image_dpi: (int), determines output image quality

                **kwargs: (str), extra argument for domain_distance, eg. minkowsi requires additional arg p

            Returns:
                None

        _evaluate_split: method to evaluate a single data split, i.e. fit model, predict test data, and perform some plots and analysis
            Args:
                X_train: (pd.DataFrame), dataframe of X training features

                X_test: (pd.DataFrame), dataframe of X test features

                y_train: (pd.Series), series of y training features

                y_test: (pd.Series), series of y test features

                model: (mastml.models instance), an estimator for fitting data

                model_name: (str), class name of the model being evaluated

                preprocessor: (mastml.preprocessor), mastml.preprocessor object to normalize the training data in each split.

                selector: (mastml.selector), a feature selector to select features in each split

                hyperopt: (mastml.hyperopt), mastml.hyperopt instance to perform model hyperparameter optimization in each split

                metrics: (list), list of metric names to evaluate true vs. pred data in each split

                plots: (list), list of names denoting which types of plots to make. Valid names are 'Scatter', 'Error', and 'Histogram'

                groups: (str), string denoting the test group, if applicable

                splitpath:(str), string denoting the split path in the save directory

                has_model_errors: (bool), whether the model used has error bars (uncertainty quantification)

                X_extra_train: (pd.DataFrame), dataframe of the extra X data of the training split (not used in fit)

                X_extra_test: (pd.DataFrame), dataframe of the extra X data of the testing split (not used in fit)

                error_method: (str), the type of model error evaluation method to perform. Only applies to certain models. Valid names are 'stdev_weak_learners' and 'jackknife_after_bootstrap'

                remove_outlier_learners: (bool), whether to remove weak learners from ensemble models whose predictions are found to be outliers. Default False.

                verbosity: (int), the output plotting verbosity. Default is 1. Valid choices are 0, 1, 2, and 3.

                baseline_test: (list), list of arrays containing base line tests to run.

                distance_metric: (str), distance metric to use in baseline_test's test_nearest_neighbour_cdist method

                file_extension: (str), must be either '.xlsx' or '.csv', determines data file type for saving

                image_dpi: (int), determines output image quality

                **kwargs: (str), extra argument for domain_distance, eg. minkowsi requires additional arg p

            Returns:
                None

        _setup_savedir: method to create a save directory based on model/selector/preprocessor names
            Args:
                model: (mastml.models instance), an estimator for fitting data

                preprocessor: (mastml.preprocessor), mastml.preprocessor object to normalize the training data in each split.

                selector: (mastml.selector), a feature selector to select features in each split

                savepath: (str), string denoting the save path of the file

        _save_split_data: method to save the X and y split data to excel files
            Args:
                df: (pd.DataFrame), dataframe of X or y data to save to file

                filename: (str), string denoting the filename, e.g. 'Xtest'

                savepath: (str), string denoting the save path of the file

                columns: (list), list of dataframe column names, e.g. X feature names

                file_extension: (str), must be either '.xlsx' or '.csv', determines data file type for saving

            Returns:
                None

        _collect_data: method to collect all pd.Series (e.g. ytrain/ytest) data into single series over many splits (directories)
            Args:
                filename: (str), string denoting the filename, e.g. 'ytest'

                savepath: (str), string denoting the save path of the file

                file_extension: (str), must be either '.xlsx' or '.csv', determines data file type for saving

                iterdirs: (bool), whether to recursively search within directories to collect data or just look in cwd

            Returns:
                data: (list), list containing flattened array of all data of a given type over many splits, e.g. all ypred data

        _collect_df_data: method to collect all pd.DataFrame (e.g. Xtrain/Xtest) data into single dataframe over many splits (directories)
            Args:
                filename: (str), string denoting the filename, e.g. 'Xtest'

                savepath: (str), string denoting the save path of the file

                file_extension: (str), must be either '.xlsx' or '.csv', determines data file type for saving

            Returns:
                data: (list), list containing flattened array of all data of a given type over many splits, e.g. all Xtest data

        _get_best_split: method to find the best performing model in a set of train/test splits
            Args:
                savepath: (str), string denoting the save path of the file

                preprocessor: (mastml.preprocessor), mastml.preprocessor object to normalize the training data in each split.

                best_run_metric: (str), name of the metric to use to find the best performing model

                model_name: (str), class name of model being evaluated

                file_extension: (str), must be either '.xlsx' or '.csv', determines data file type for saving

            Returns:
                best_split_dict: (dict), dictionary containing the path locations of the best model and corresponding preprocessor and selected feature list

        _get_average_recalibration_params: method to get the average and standard deviation of the recalibration factors in all train/test CV sets
            Args:
                savepath: (str), string denoting the save path of the file

                data_type: (str), string denoting the type of data to examine (e.g. test or leftout)

                file_extension: (str), must be either '.xlsx' or '.csv', determines data file type for saving

            Returns:
                recalibrate_avg_dict: (dict): dictionary of average recalibration parameters

                recalibrate_stdev_dict: (dict): dictionary of stdev of recalibration parameters

        _get_recalibration_params: method to get the recalibration factors for a single evaluation
            Args:
                savepath: (str), string denoting the save path of the file

                data_type: (str), string denoting the type of data to examine (e.g. test or leftout)

                file_extension: (str), must be either '.xlsx' or '.csv', determines data file type for saving

            Returns:
                recalibrate_dict: (dict): dictionary of recalibration parameters

        help: method to output key information on class use, e.g. methods and parameters
            Args:
                None

            Returns:
                None, but outputs help to screen
    """

    def __init__(self):
        super(BaseSplitter, self).__init__()
        self.splitter = self.__class__.__name__

    def split_asframe(self, X, y, groups=None, X_force_train=None, y_force_train=None):
        split = self.split(X, y, groups)
        X_splits = list()
        y_splits = list()
        train_inds = list()
        test_inds = list()
        for train, test in split:
            if X_force_train is None and y_force_train is None:
                X_splits.append((X.iloc[train], X.iloc[test]))
                y_splits.append((y.iloc[train], y.iloc[test]))
                train_inds.append(train)
                test_inds.append(test)
            else:
                X_splits.append((pd.concat([X.iloc[train], X_force_train]), X.iloc[test]))
                y_splits.append((pd.concat([y.iloc[train], y_force_train]), y.iloc[test]))
                test_inds.append(test)
                train_inds.append(train) # Don't append the forced train inds b/c don't want to evaluate metrics with these extra data, also creates issues with groups indexing
                # Note that including the forced training data causes an issue with plotting the train data all split average parity plots
                #num_ind = X.shape[0]
                #train_inds_extra = np.arange(0+num_ind, X_force_train.shape[0]+num_ind)
                #train_inds.append(np.concatenate([train, train_inds_extra]))
        return X_splits, y_splits, train_inds, test_inds

    def evaluate(self, X, y, models, preprocessor=None, groups=None, hyperopts=None, selectors=None, metrics=None,
                 plots=None, savepath=None, X_extra=None, X_force_train=None, y_force_train=None, leaveout_inds=list(list()),
                 best_run_metric=None, nested_CV=False, error_method='stdev_weak_learners', remove_outlier_learners=False,
                 recalibrate_errors=False, verbosity=1, baseline_test=None, distance_metric="euclidean",
                 file_extension='.csv', image_dpi=250, parallel_run=False, remove_split_dirs=False,
                 X_val=None, y_val=None, X_test=None, y_test=None, force_no_model_errors=False,
                 rve_number_of_bins=15, rve_equal_sized_bins=False, domain=None, recalibrate_power=1, recalibrate_per_bin=False, **kwargs):

        if nested_CV == True:
            model_names = [m.__class__.__name__ for m in models]
            if 'CrabNetModel' in model_names:
                print('Error: Nested CV is not supported with CrabNet model')
                sys.exit()
            if self.__class__.__name__ == 'NoSplit':
                print('Warning: NoSplit does not support nested cross validation.')
            else:
                # Get set of X_leaveout, y_leaveout for testing. Append them to user-specified X_leaveout tests
                X_splits, y_splits, train_inds, test_inds = self.split_asframe(X=X, y=y, groups=groups, X_force_train=X_force_train, y_force_train=y_force_train)
                leaveout_inds_orig = leaveout_inds
                leaveout_inds = [i for i in test_inds]
                if len(leaveout_inds_orig) > 0:
                    for i in leaveout_inds_orig:
                        leaveout_inds.append(i)

        if plots is None:
            plots = ['Scatter']

        if type(models) == list:
            pass
        else:
            models = list(models)

        if selectors is None:
            selectors = [NoSelect()]
        else:
            if type(selectors) == list:
                pass
            else:
                selectors = list(selectors)

        if preprocessor is None:
            preprocessor = NoPreprocessor()

        if metrics is None:
            metrics = ['root_mean_squared_error']

        if best_run_metric is None:
            best_run_metric = metrics[0]

        if hyperopts is None:
            hyperopts = [None for l in models]

        if not savepath:
            savepath = os.getcwd()

        self.splitdirs = list()
        for model, hyperopt in zip(models, hyperopts):

            # See if the model used is amenable to uncertainty (error) analysis
            try:
                model_name = model.model.__class__.__name__
            except:
                model_name = model.__class__.__name__
            if model_name == 'NoneType':
                model_name = model.__class__.__name__

            if model_name in ['RandomForestRegressor',
                                                     'GradientBoostingRegressor',
                                                     'GaussianProcessRegressor',
                                                     'BaggingRegressor',
                                                     'ExtraTreesRegressor',
                                                  'AdaBoostRegressor']:
                has_model_errors = True
            else:
                has_model_errors = False
                if recalibrate_errors is True:
                    print('Warning: you have selected to recalibrate errors using a model that does not support '
                          'error estimation. Automatically changing to set recalibrate_errors = False')
                    recalibrate_errors = False

            if force_no_model_errors == True:
                has_model_errors = False

            for selector in selectors:
                splitdir = self._setup_savedir(model=model, selector=selector, preprocessor=preprocessor, savepath=savepath)
                self.splitdirs.append(splitdir)
                split_outer_count = 0
                split_summary_list_outer = dict()

                if len(leaveout_inds) > 0:

                    for leaveout_ind in tqdm(leaveout_inds, 'Running outer split'):
                        split_summary_list_outer[split_outer_count] = dict()
                        split_summary_list_outer[split_outer_count]['leaveout_ind'] = np.array(leaveout_ind).ravel()

                        X_subsplit = X.loc[~X.index.isin(leaveout_ind)]
                        y_subsplit = y.loc[~y.index.isin(leaveout_ind)]
                        X_leaveout = X.loc[X.index.isin(leaveout_ind)]
                        y_leaveout = y.loc[y.index.isin(leaveout_ind)]
                        if X_extra is not None:
                            X_extra_subsplit = X_extra.loc[~X_extra.index.isin(leaveout_ind)]
                            X_extra_leaveout = X_extra.loc[X_extra.index.isin(leaveout_ind)]
                        else:
                            X_extra_subsplit = None
                            X_extra_leaveout = None

                        dataset_stdev = np.std(y_subsplit)

                        if groups is not None:
                            groups_subsplit = groups.loc[~groups.index.isin(leaveout_ind)]
                            groups_leaveout = groups.loc[groups.index.isin(leaveout_ind)]
                        else:
                            groups_subsplit = None

                        X_splits, y_splits, train_inds, test_inds = self.split_asframe(X=X_subsplit, y=y_subsplit,
                                                                                       groups=groups_subsplit,
                                                                                       X_force_train=X_force_train,
                                                                                       y_force_train=y_force_train)

                        # make the individual split directory
                        splitouterpath = os.path.join(splitdir, 'split_outer_' + str(split_outer_count))
                        # make the feature selector directory for this split directory
                        os.mkdir(splitouterpath)

                        # Save the left-out data indices
                        if verbosity >= 0:
                            if file_extension == '.xlsx':
                                pd.DataFrame({'leaveout_inds': leaveout_ind}).to_excel(os.path.join(splitouterpath, 'leaveout_inds'+file_extension), index=False)
                            elif file_extension == '.csv':
                                pd.DataFrame({'leaveout_inds': leaveout_ind}).to_csv(os.path.join(splitouterpath, 'leaveout_inds' + file_extension), index=False)

                            # Save the left-out data groups
                            if groups is not None:
                                groups_leaveout.name = 'leaveout_groups'
                                if file_extension == '.xlsx':
                                    groups_leaveout.to_excel(os.path.join(splitouterpath, 'leaveout_groups'+file_extension), index=False)
                                elif file_extension == '.csv':
                                    groups_leaveout.to_csv(os.path.join(splitouterpath, 'leaveout_groups'+file_extension), index=False)

                        outerdir, split_summary_list = self._evaluate_split_sets(X_splits,
                                                             y_splits,
                                                             train_inds,
                                                             test_inds,
                                                             model,
                                                             model_name,
                                                             selector,
                                                             preprocessor,
                                                             X_extra_subsplit,
                                                             groups_subsplit, #groups subsplit??? Used to be "groups"
                                                             splitouterpath,
                                                             hyperopt,
                                                             metrics,
                                                             plots,
                                                             has_model_errors,
                                                             error_method,
                                                             remove_outlier_learners,
                                                             recalibrate_errors,
                                                             verbosity,
                                                             baseline_test,
                                                             distance_metric,
                                                             file_extension,
                                                             image_dpi,
                                                             parallel_run,
                                                             rve_number_of_bins,
                                                             rve_equal_sized_bins,
                                                             domain,
                                                             recalibrate_power,
                                                             recalibrate_per_bin,
                                                             **kwargs)

                        # Save the split summary dict
                        with open(os.path.join(splitouterpath, 'split_summary.json'), 'w') as f:
                            json.dump(split_summary_list, f, cls=NumpyEncoder)

                        split_summary_list_flat = flatten_split_summary(split_summary_list)
                        with open(os.path.join(splitouterpath, 'split_summary_allsplits.json'), 'w') as f:
                            json.dump(split_summary_list_flat, f, cls=NumpyEncoder)

                        best_split_dict = self._get_best_split(savepath=splitouterpath,
                                                               model=model,
                                                               preprocessor=preprocessor,
                                                               best_run_metric=best_run_metric,
                                                               model_name=model_name,
                                                               file_extension=file_extension)
                        # Copy the best model, selected features and preprocessor to this outer directory
                        shutil.copy(best_split_dict['preprocessor'], splitouterpath)
                        if model_name == 'KerasRegressor':
                            try:
                                shutil.copy(best_split_dict['model'], splitouterpath)
                            except:
                                print('Warning: could not move best Keras model to splitdir')
                        elif model_name == 'BaggingRegressor':
                            if model.base_estimator_ == 'KerasRegressor':
                                for m in best_split_dict['model']:
                                    shutil.move(m, splitouterpath)
                        else:
                            try:
                                shutil.copy(best_split_dict['model'], splitouterpath)
                            except:
                                print('Warning: could not copy best model to splitdir')
                        #shutil.copy(best_split_dict['model'], splitouterpath)
                        shutil.copy(best_split_dict['features'], splitouterpath)

                        # Load in the best model, preprocessor and evaluate the left-out data stats
                        if model_name == 'BaggingRegressor':
                            if model.base_estimator_ == 'KerasRegressor':
                                # Need to rebuild the ensemble of Keras models
                                import tensorflow as tf
                                keras_dirs = [d for d in os.listdir(splitouterpath) if 'keras_model' in d]
                                estimators = [tf.keras.models.load_model(os.path.join(splitouterpath, d)) for d in keras_dirs]
                                estimator_features = list()
                                for e in estimators:
                                    estimator_features.append(np.arange(0, X.shape[1]))
                                model.model.estimators_ = estimators
                                model.model.estimators_features_ = estimator_features
                                best_model = model
                            else:
                                best_model = joblib.load(best_split_dict['model'])
                        else:
                            best_model = joblib.load(best_split_dict['model'])
                        preprocessor = joblib.load(best_split_dict['preprocessor'])

                        # Need this for jacknife-after-bootstrap method.
                        try:
                            if file_extension == '.xlsx':
                                X_train_bestmodel = preprocessor.transform(pd.read_excel(best_split_dict['X_train'], engine='openpyxl')) # Need to preprocess the Xtrain data
                            elif file_extension == '.csv':
                                X_train_bestmodel = preprocessor.transform(pd.read_csv(best_split_dict['X_train']))  # Need to preprocess the Xtrain data
                        except:
                            pass

                        with open(os.path.join(splitouterpath, 'selected_features.txt')) as f:
                            selected_features = [line.rstrip() for line in f]
                        X_leaveout = X_leaveout[selected_features]
                        X_leaveout_preprocessed = preprocessor.transform(X=X_leaveout)
                        y_pred_leaveout = best_model.predict(X_leaveout_preprocessed)
                        y_pred_leaveout = y_pred_leaveout.ravel()
                        y_pred_leaveout = pd.Series(y_pred_leaveout, name='y_pred_leaveout')

                        split_summary_list_outer[split_outer_count]['preds_leaveout'] = np.array(y_pred_leaveout).ravel()
                        split_summary_list_outer[split_outer_count]['trues_leaveout'] = np.array(y_leaveout).ravel()

                        stats_dict_leaveout = Metrics(metrics_list=metrics).evaluate(y_true=y_leaveout,
                                                                                     y_pred=y_pred_leaveout)
                        df_stats_leaveout = pd.DataFrame().from_records([stats_dict_leaveout])

                        split_summary_list_outer[split_outer_count]['stats_leaveout'] = stats_dict_leaveout


                        if file_extension == '.xlsx':
                            df_stats_leaveout.to_excel(os.path.join(splitouterpath, 'leaveout_stats_summary'+file_extension), index=False)
                        elif file_extension == '.csv':
                            df_stats_leaveout.to_csv(os.path.join(splitouterpath, 'leaveout_stats_summary' + file_extension), index=False)

                        # At level of splitouterpath, do analysis over all splits (e.g. parity plot over all splits)
                        if verbosity >= 0:
                            if groups is not None:
                                groups_leaveout_all = self._collect_data(filename='leaveout_groups', savepath=splitouterpath, file_extension=file_extension, iterdirs=False)
                            else:
                                groups_leaveout_all = None
                            y_test_all = self._collect_data(filename='y_test', savepath=splitouterpath, file_extension=file_extension)
                            y_train_all = self._collect_data(filename='y_train', savepath=splitouterpath, file_extension=file_extension)
                            y_pred_all = self._collect_data(filename='y_pred', savepath=splitouterpath, file_extension=file_extension)
                            y_pred_train_all = self._collect_data(filename='y_pred_train', savepath=splitouterpath, file_extension=file_extension)
                            residuals_test_all = self._collect_data(filename='residuals_test', savepath=splitouterpath, file_extension=file_extension)
                            residuals_train_all = self._collect_data(filename='residuals_train', savepath=splitouterpath, file_extension=file_extension)
                            X_train_all = self._collect_df_data(filename='X_train', savepath=splitouterpath, file_extension=file_extension)
                            X_test_all = self._collect_df_data(filename='X_test', savepath=splitouterpath, file_extension=file_extension)

                            # Save the data gathered over all the splits
                            self._save_split_data(df=X_train_all, filename='X_train', savepath=splitouterpath, columns=X_train_all.columns.tolist(), file_extension=file_extension)
                            self._save_split_data(df=X_test_all, filename='X_test', savepath=splitouterpath, columns=X_test_all.columns.tolist(), file_extension=file_extension)
                            self._save_split_data(df=y_test_all, filename='y_test', savepath=splitouterpath, columns='y_test', file_extension=file_extension)
                            self._save_split_data(df=y_train_all, filename='y_train', savepath=splitouterpath, columns='y_train', file_extension=file_extension)
                            self._save_split_data(df=y_pred_all, filename='y_pred', savepath=splitouterpath, columns='y_pred', file_extension=file_extension)
                            self._save_split_data(df=y_pred_train_all, filename='y_pred_train', savepath=splitouterpath, columns='y_pred_train', file_extension=file_extension)
                            self._save_split_data(df=residuals_test_all, filename='residuals_test', savepath=splitouterpath, columns='residuals', file_extension=file_extension)
                            self._save_split_data(df=residuals_train_all, filename='residuals_train', savepath=splitouterpath, columns='residuals', file_extension=file_extension)

                        if has_model_errors is True:
                            if verbosity >= 0:
                                model_errors_test_all = self._collect_data(filename='model_errors_test', savepath=splitouterpath, file_extension=file_extension)
                                model_errors_train_all = self._collect_data(filename='model_errors_train', savepath=splitouterpath, file_extension=file_extension)
                                self._save_split_data(df=model_errors_test_all, filename='model_errors_test', savepath=splitouterpath, columns='model_errors', file_extension=file_extension)
                                self._save_split_data(df=model_errors_train_all, filename='model_errors_train', savepath=splitouterpath, columns='model_errors', file_extension=file_extension)
                            try:
                                model_errors_leaveout, num_removed_learners_leaveout = ErrorUtils()._get_model_errors(model=best_model,
                                                                                        X=X_leaveout_preprocessed,
                                                                                        X_test=X_leaveout_preprocessed,
                                                                                        X_train=X_train_bestmodel,
                                                                                       error_method=error_method,
                                                                                        remove_outlier_learners=remove_outlier_learners)
                            except:
                                model_errors_leaveout, num_removed_learners_leaveout = ErrorUtils()._get_model_errors(model=best_model,
                                                                                        X=X_leaveout_preprocessed,
                                                                                        X_test=X_leaveout_preprocessed,
                                                                                        X_train=None,
                                                                                        error_method=error_method,
                                                                                        remove_outlier_learners=remove_outlier_learners)
                            if verbosity >= 0:
                                self._save_split_data(df=model_errors_leaveout, filename='model_errors_leaveout',
                                                  savepath=splitouterpath, columns='model_errors', file_extension=file_extension)
                                self._save_split_data(df=num_removed_learners_leaveout, filename='num_removed_learners_leaveout',
                                                  savepath=splitouterpath, columns='num_removed_learners', file_extension=file_extension)

                            split_summary_list_outer[split_outer_count]['model_errors_leaveout'] = np.array(model_errors_leaveout).ravel()

                        else:
                            model_errors_leaveout = None

                        # Remake the leaveout y data series to reset the index
                        y_leaveout = pd.Series(np.array(y_leaveout))
                        y_pred_leaveout = pd.Series(np.array(y_pred_leaveout))

                        if verbosity >= 0:
                            self._save_split_data(df=X_leaveout, filename='X_leaveout', savepath=splitouterpath, columns=X_leaveout.columns.tolist(), file_extension=file_extension)
                            if X_extra is not None:
                                self._save_split_data(df=X_extra_leaveout, filename='X_extra_leaveout', savepath=splitouterpath, columns=X_extra_leaveout.columns.tolist(), file_extension=file_extension)
                            self._save_split_data(df=y_leaveout, filename='y_leaveout', savepath=splitouterpath, columns='y_leaveout', file_extension=file_extension)
                            self._save_split_data(df=y_pred_leaveout, filename='y_pred_leaveout', savepath=splitouterpath, columns='y_pred_leaveout', file_extension=file_extension)

                            residuals_leaveout = y_pred_leaveout-y_leaveout
                            self._save_split_data(df=residuals_leaveout, filename='residuals_leaveout', savepath=splitouterpath, columns='residuals', file_extension=file_extension)

                        if recalibrate_errors is True:
                            recalibrate_dict = self._get_recalibration_params(savepath=splitouterpath,
                                                                              data_type='test', file_extension=file_extension)

                            if recalibrate_per_bin == False:
                                recalibrate_dict = recalibrate_dict[0]
                                if recalibrate_power == 0:
                                    model_errors_leaveout_cal = model_errors_leaveout + recalibrate_dict['a']
                                elif recalibrate_power == 1:
                                    model_errors_leaveout_cal = recalibrate_dict['a']*model_errors_leaveout+recalibrate_dict['b']
                                elif recalibrate_power == 2:
                                    model_errors_leaveout_cal = recalibrate_dict['a']*model_errors_leaveout**2 + recalibrate_dict['b']*model_errors_leaveout + recalibrate_dict['c']
                                elif recalibrate_power == 3:
                                    model_errors_leaveout_cal = recalibrate_dict['a']*model_errors_leaveout**3 + recalibrate_dict['b']*model_errors_leaveout**2 + recalibrate_dict['c']*model_errors_leaveout + recalibrate_dict['d']
                            else:
                                # Need to bin the data
                                digitized, err_values, rms_residual_values, num_values_per_bin, number_of_bins, ms_residual_values, var_sq_residual_values = ErrorUtils()._parse_error_data(
                                    model_errors=model_errors_leaveout,
                                    residuals=residuals_leaveout,
                                    dataset_stdev=dataset_stdev,
                                    number_of_bins=rve_number_of_bins,
                                    equal_sized_bins=rve_equal_sized_bins)

                                # Now organize the residuals and ebars into the right bins
                                data_dict_leaveout = dict()
                                counts = 0
                                for b, r, e in zip(digitized, residuals_leaveout, model_errors_leaveout):
                                    if b not in data_dict_leaveout.keys():
                                        data_dict_leaveout[b] = {'res': list(), 'err': list(), 'z': list()}
                                    data_dict_leaveout[b]['res'].append(r)
                                    data_dict_leaveout[b]['err'].append(e)
                                    try:
                                        data_dict_leaveout[b]['z'].append(r / e)
                                    except:
                                        data_dict_leaveout[b]['z'].append(np.nan)
                                    if b < 2:
                                        counts += 1

                                # Need to apply the per-bin recal params
                                model_errors_leaveout_cal = list()
                                residuals_leaveout_reorder = list()
                                model_errors_leaveout_reorder = list()
                                for i, b in enumerate(data_dict_leaveout.keys()):
                                    # If the bin can't be found, default to the last bin
                                    try:
                                        recal = recalibrate_dict[b-1]
                                    except:
                                        recal = recalibrate_dict[-1]
                                    err = np.array(data_dict_leaveout[b]['err'])
                                    if recalibrate_power == 0:
                                        err_cal = err + recal['a']
                                    elif recalibrate_power == 1:
                                        err_cal = recal['a'] * err + \
                                                                    recal['b']
                                    elif recalibrate_power == 2:
                                        err_cal = recal['a'] * err ** 2 + \
                                                                    recal['b'] * err + \
                                                                    recal['c']
                                    elif recalibrate_power == 3:
                                        err_cal = recal['a'] * err ** 3 + \
                                                                    recal['b'] * err ** 2 + \
                                                                    recal['c'] * err + \
                                                                    recal['d']
                                    model_errors_leaveout_cal.append(err_cal)
                                    residuals_leaveout_reorder.append(data_dict_leaveout[b]['res'])
                                    model_errors_leaveout_reorder.append(data_dict_leaveout[b]['err'])

                                # TODO: issue here where model errors now in different order from before, and are detached from their respective residuals??
                                model_errors_leaveout_cal = pd.Series(list(itertools.chain.from_iterable(model_errors_leaveout_cal)), name='model_errors')
                                residuals_leaveout = pd.Series(list(itertools.chain.from_iterable(residuals_leaveout_reorder)), name='residuals')
                                model_errors_leaveout = pd.Series(list(itertools.chain.from_iterable(model_errors_leaveout_reorder)), name='model_errors')

                            if verbosity >= 0:
                                self._save_split_data(df=model_errors_leaveout_cal, filename='model_errors_leaveout_calibrated',
                                                  savepath=splitouterpath, columns='model_errors', file_extension=file_extension)
                                # Save the residuals and model errors again since they're been re-sorted
                                self._save_split_data(df=model_errors_leaveout, filename='model_errors_leaveout',
                                                  savepath=splitouterpath, columns='model_errors', file_extension=file_extension)
                                self._save_split_data(df=residuals_leaveout, filename='residuals_leaveout',
                                                  savepath=splitouterpath, columns='residuals', file_extension=file_extension)

                            split_summary_list_outer[split_outer_count]['model_errors_leaveout_cal'] = np.array(model_errors_leaveout_cal).ravel()
                        else:
                            model_errors_leaveout_cal = None

                        if verbosity >= 1:
                            make_plots(plots=plots,
                                       y_true=y_leaveout,
                                       y_pred=y_pred_leaveout,
                                       X_test=X_leaveout,
                                       groups=groups_leaveout_all,
                                       data_type='leaveout',
                                       dataset_stdev=dataset_stdev,
                                       has_model_errors=has_model_errors,
                                       metrics=metrics,
                                       model=model,
                                       model_errors=model_errors_leaveout,
                                       residuals=residuals_leaveout,
                                       savepath=splitouterpath,
                                       show_figure=False,
                                       recalibrate_errors=recalibrate_errors,
                                       model_errors_cal=model_errors_leaveout_cal,
                                       splits_summary=True,
                                       file_extension=file_extension,
                                       image_dpi=image_dpi,
                                       number_of_bins=rve_number_of_bins,
                                       equal_sized_bins=rve_equal_sized_bins)

                        # Update the MASTML metadata file to include the leftout data info
                        '''
                        if mastml is not None:
                            if verbosity >= 0:
                                mastml._update_metadata(outerdir=outerdir,
                                                        split_name='split_summary',
                                                        leaveout_stats=df_stats_leaveout,
                                                        X_leaveout=X_leaveout,
                                                        X_extra_leaveout=X_extra_leaveout,
                                                        y_leaveout=y_leaveout,
                                                        y_pred_leaveout=y_pred_leaveout,
                                                        residuals_leaveout=residuals_leaveout,
                                                        model_errors_leaveout=model_errors_leaveout,
                                                        model_errors_leaveout_cal=model_errors_leaveout_cal,
                                                        )
                                mastml._save_mastml_metadata()
                        '''

                        split_outer_count += 1

                    # At level of splitdir, collect and save all leaveout data
                    if verbosity >= 0:
                        if groups is not None:
                            groups_leaveout_all = self._collect_data(filename='leaveout_groups', savepath=splitdir, file_extension=file_extension)
                        else:
                            groups_leaveout_all = None

                        y_leaveout_all = self._collect_data(filename='y_leaveout', savepath=splitdir, file_extension=file_extension)
                        leaveout_inds_all = self._collect_data(filename='leaveout_inds', savepath=splitdir, file_extension=file_extension)
                        y_pred_leaveout_all = self._collect_data(filename='y_pred_leaveout', savepath=splitdir, file_extension=file_extension)
                        residuals_leaveout_all = self._collect_data(filename='residuals_leaveout', savepath=splitdir, file_extension=file_extension)
                        self._save_split_data(df=residuals_leaveout_all, filename='residuals_leaveout', savepath=splitdir, columns='residuals', file_extension=file_extension)
                        self._save_split_data(df=y_leaveout_all, filename='y_leaveout', savepath=splitdir, columns='y_leaveout', file_extension=file_extension)
                        self._save_split_data(df=y_pred_leaveout_all, filename='y_pred_leaveout', savepath=splitdir, columns='y_pred_leaveout', file_extension=file_extension)

                        # At level of splitodir, collect and save all train/test data
                        y_test_all = self._collect_data(filename='y_test', savepath=splitdir, file_extension=file_extension)
                        y_train_all = self._collect_data(filename='y_train', savepath=splitdir, file_extension=file_extension)
                        y_pred_all = self._collect_data(filename='y_pred', savepath=splitdir, file_extension=file_extension)
                        y_pred_train_all = self._collect_data(filename='y_pred_train', savepath=splitdir, file_extension=file_extension)
                        residuals_test_all = self._collect_data(filename='residuals_test', savepath=splitdir, file_extension=file_extension)
                        residuals_train_all = self._collect_data(filename='residuals_train', savepath=splitdir, file_extension=file_extension)
                        X_train_all = self._collect_df_data(filename='X_train', savepath=splitdir, file_extension=file_extension)
                        X_test_all = self._collect_df_data(filename='X_test', savepath=splitdir, file_extension=file_extension)
                        X_leaveout_all = self._collect_df_data(filename='X_leaveout', savepath=splitdir, file_extension=file_extension)
                        if X_extra is not None:
                            X_extra_train_all = self._collect_df_data(filename='X_extra_train', savepath=splitdir, file_extension=file_extension)
                            X_extra_test_all = self._collect_df_data(filename='X_extra_test', savepath=splitdir, file_extension=file_extension)
                            X_extra_leaveout_all = self._collect_df_data(filename='X_extra_leaveout', savepath=splitdir, file_extension=file_extension)
                        self._save_split_data(df=X_train_all, filename='X_train', savepath=splitdir, columns=X_train_all.columns.tolist(), file_extension=file_extension)
                        self._save_split_data(df=X_test_all, filename='X_test', savepath=splitdir, columns=X_test_all.columns.tolist(), file_extension=file_extension)
                        self._save_split_data(df=X_leaveout_all, filename='X_leaveout', savepath=splitdir, columns=X_leaveout_all.columns.tolist(), file_extension=file_extension)
                        if X_extra is not None:
                            self._save_split_data(df=X_extra_train_all, filename='X_train', savepath=splitdir, columns=X_extra_train_all.columns.tolist(), file_extension=file_extension)
                            self._save_split_data(df=X_extra_test_all, filename='X_test', savepath=splitdir, columns=X_extra_test_all.columns.tolist(), file_extension=file_extension)
                            self._save_split_data(df=X_extra_leaveout_all, filename='X_leaveout', savepath=splitdir, columns=X_extra_leaveout_all.columns.tolist(), file_extension=file_extension)
                        self._save_split_data(df=y_test_all, filename='y_test', savepath=splitdir, columns='y_test', file_extension=file_extension)
                        self._save_split_data(df=y_train_all, filename='y_train', savepath=splitdir,  columns='y_train', file_extension=file_extension)
                        self._save_split_data(df=y_pred_all, filename='y_pred', savepath=splitdir, columns='y_pred', file_extension=file_extension)
                        self._save_split_data(df=y_pred_train_all, filename='y_pred_train', savepath=splitdir, columns='y_pred_train', file_extension=file_extension)
                        self._save_split_data(df=residuals_test_all, filename='residuals_test', savepath=splitdir, columns='residuals', file_extension=file_extension)
                        self._save_split_data(df=residuals_train_all, filename='residuals_train', savepath=splitdir, columns='residuals', file_extension=file_extension)

                        # Remake the Series so that indices are sequential (needed for math)
                        y_leaveout_all = pd.Series(np.array(y_leaveout_all))
                        y_pred_leaveout_all = pd.Series(np.array(y_pred_leaveout_all))

                    if has_model_errors is True:
                        if verbosity >= 0:
                            model_errors_test_all = self._collect_data(filename='model_errors_test', savepath=splitdir, file_extension=file_extension)
                            model_errors_train_all = self._collect_data(filename='model_errors_train', savepath=splitdir, file_extension=file_extension)
                            model_errors_leaveout_all = self._collect_data(filename='model_errors_leaveout', savepath=splitdir, file_extension=file_extension)
                            num_removed_learners_leaveout_all = self._collect_data(filename='num_removed_learners_leaveout', savepath=splitdir, file_extension=file_extension)
                            self._save_split_data(df=model_errors_test_all, filename='model_errors_test', savepath=splitdir, columns='model_errors', file_extension=file_extension)
                            self._save_split_data(df=model_errors_train_all, filename='model_errors_train', savepath=splitdir, columns='model_errors', file_extension=file_extension)
                            self._save_split_data(df=model_errors_leaveout_all, filename='model_errors_leaveout', savepath=splitdir, columns='model_errors', file_extension=file_extension)
                            self._save_split_data(df=num_removed_learners_leaveout_all, filename='num_removed_learners_leaveout', savepath=splitdir, columns='num_removed_learners', file_extension=file_extension)
                            if recalibrate_errors is True:
                                model_errors_leaveout_all_calibrated = self._collect_data(filename='model_errors_leaveout_calibrated', savepath=splitdir, file_extension=file_extension)
                                self._save_split_data(df=model_errors_leaveout_all_calibrated, filename='model_errors_leaveout_calibrated', savepath=splitdir, columns='model_errors', file_extension=file_extension)
                            else:
                                model_errors_leaveout_all_calibrated = None
                    else:
                        model_errors_leaveout_all = None
                        model_errors_leaveout_all_calibrated = None

                    # Gather the recalibration dicts from each split set and save average and stdev of recalibrations
                    if has_model_errors is True and recalibrate_errors is True:
                        recalibrate_avg_dict, recalibrate_stdev_dict = self._get_average_recalibration_params(savepath=splitdir,
                                                                                                              data_type='test', file_extension=file_extension)
                        if file_extension == '.xlsx':
                            pd.DataFrame(recalibrate_avg_dict).to_excel(os.path.join(splitdir, 'recalibration_parameters_average_test'+file_extension))
                            pd.DataFrame(recalibrate_stdev_dict).to_excel(os.path.join(splitdir, 'recalibration_parameters_stdev_test'+file_extension))
                        elif file_extension == '.csv':
                            pd.DataFrame(recalibrate_avg_dict).to_csv(os.path.join(splitdir, 'recalibration_parameters_average_test'+file_extension))
                            pd.DataFrame(recalibrate_stdev_dict).to_csv(os.path.join(splitdir, 'recalibration_parameters_stdev_test'+file_extension))

                    # Make all leaveout data plots
                    if verbosity >= 1:
                        make_plots(plots=plots,
                                   y_true=y_leaveout_all,
                                   y_pred=y_pred_leaveout_all,
                                   X_test=X_leaveout_all,
                                   groups=groups_leaveout_all,
                                   data_type='leaveout',
                                   dataset_stdev=np.std(y_leaveout_all),
                                   has_model_errors=has_model_errors,
                                   metrics=metrics,
                                   model=model,
                                   model_errors=model_errors_leaveout_all,
                                   residuals=residuals_leaveout_all,
                                   savepath=splitdir,
                                   show_figure=False,
                                   recalibrate_errors=recalibrate_errors,
                                   model_errors_cal=model_errors_leaveout_all_calibrated,
                                   splits_summary=True,
                                   file_extension=file_extension,
                                   image_dpi=image_dpi,
                                   number_of_bins=rve_number_of_bins,
                                   equal_sized_bins=rve_equal_sized_bins)

                    # Update the MASTML metadata file
                    try:
                        if file_extension == '.xlsx':
                            df_stats_leaveout = pd.read_excel(os.path.join(splitdir, 'leaveout_average_stdev_stats_summary'+file_extension))
                        elif file_extension == '.csv':
                            df_stats_leaveout = pd.read_csv(os.path.join(splitdir, 'leaveout_average_stdev_stats_summary' + file_extension))
                    except:
                        df_stats_leaveout = None

                    '''
                    if mastml is not None:
                        outerdir = splitdir.split('/')[-1]
                        if 'split_outer' in outerdir:
                            # For nested CV or left out data runs with outer and inner splits, need the model dir one level up
                            outerdir = os.path.join(splitdir.split('/')[-2], splitdir.split('/')[-1])
                        
                        if verbosity >= 0:
                            mastml._update_metadata(outerdir=outerdir,
                                                    split_name='split_outer_summary',
                                                    model=model,
                                                    splitter=self,
                                                    preprocessor=preprocessor,
                                                    selector=selector,
                                                    hyperopt=hyperopt,
                                                    train_stats=None,
                                                    test_stats=None,
                                                    leaveout_stats=df_stats_leaveout,
                                                    leaveout_inds=leaveout_inds_all,
                                                    #X_train=pd.DataFrame(np.array(X_train_all), columns=X_train_all.columns.tolist()),
                                                    #X_test=pd.DataFrame(np.array(X_test_all), columns=X_test_all.columns.tolist()),
                                                    #X_leaveout=pd.DataFrame(np.array(X_leaveout_all), columns=X_leaveout_all.columns.tolist()),
                                                    #X_extra_train=pd.DataFrame(np.array(X_extra_train_all), columns=X_extra_train_all.columns.tolist()) if X_extra is not None else None,
                                                    #X_extra_test=pd.DataFrame(np.array(X_extra_test_all), columns=X_extra_test_all.columns.tolist()) if X_extra is not None else None,
                                                    #X_extra_leaveout=pd.DataFrame(np.array(X_extra_leaveout_all), columns=X_extra_leaveout_all.columns.tolist()) if X_extra is not None else None,
                                                    y_train=y_train_all,
                                                    y_test=y_test_all,
                                                    y_leaveout=y_leaveout_all,
                                                    y_pred_train=y_pred_train_all,
                                                    y_pred=y_pred_all,
                                                    y_pred_leaveout=y_pred_leaveout_all,
                                                    residuals_train=residuals_train_all,
                                                    residuals_test=residuals_test_all,
                                                    residuals_leaveout=residuals_leaveout_all,
                                                    model_errors_train=None,
                                                    model_errors_test=None,
                                                    model_errors_leaveout=model_errors_leaveout_all,
                                                    model_errors_train_cal=None,
                                                    model_errors_test_cal=None,
                                                    model_errors_leaveout_cal=model_errors_leaveout_all_calibrated,
                                                    dataset_stdev=None)
                            mastml._save_mastml_metadata()
                    '''

                    # Save the outer split summary file
                    # Save the split summary dict
                    with open(os.path.join(splitdir, 'split_summary_outer.json'), 'w') as f:
                        json.dump(split_summary_list_outer, f, cls=NumpyEncoder)

                    split_summary_list_outer_flat = flatten_split_summary(split_summary_list_outer)
                    with open(os.path.join(splitdir, 'split_summary_outer_allsplits.json'), 'w') as f:
                        json.dump(split_summary_list_outer_flat, f, cls=NumpyEncoder)

                    # Remove all the splitdirs if set to True
                    if remove_split_dirs == True:
                        ds = os.listdir(splitdir)
                        splitdirs = [d for d in ds if 'split_' in d and '.png' not in d and '.json' not in d]
                        for d in splitdirs:
                            shutil.rmtree(os.path.join(splitdir, d))

                else:
                    X_splits, y_splits, train_inds, test_inds = self.split_asframe(X=X, y=y, groups=groups,
                                                                                   X_force_train=X_force_train,
                                                                                   y_force_train=y_force_train)

                    outerdir, split_summary_list = self._evaluate_split_sets(X_splits,
                                              y_splits,
                                              train_inds,
                                              test_inds,
                                              model,
                                              model_name,
                                              selector,
                                              preprocessor,
                                              X_extra,
                                              groups,
                                              splitdir,
                                              hyperopt,
                                              metrics,
                                              plots,
                                              has_model_errors,
                                              error_method,
                                              remove_outlier_learners,
                                              recalibrate_errors,
                                              verbosity,
                                              baseline_test,
                                              distance_metric,
                                              file_extension,
                                              image_dpi,
                                              parallel_run,
                                              rve_number_of_bins,
                                              rve_equal_sized_bins,
                                              domain,
                                              recalibrate_power,
                                              recalibrate_per_bin,
                                              **kwargs)

                    # Save the split summary dict
                    with open(os.path.join(splitdir, 'split_summary.json'), 'w') as f:
                        json.dump(split_summary_list, f, cls=NumpyEncoder)

                    split_summary_list_flat = flatten_split_summary(split_summary_list)
                    with open(os.path.join(splitdir, 'split_summary_allsplits.json'), 'w') as f:
                        json.dump(split_summary_list_flat, f, cls=NumpyEncoder)

                    #if verbosity >= 0:
                    best_split_dict = self._get_best_split(savepath=splitdir,
                                                           model=model,
                                                           preprocessor=preprocessor,
                                                           best_run_metric=best_run_metric,
                                                           model_name=model_name,
                                                           file_extension=file_extension)
                    # Copy the best model, selected features and preprocessor to this outer directory
                    try:
                        shutil.copy(best_split_dict['preprocessor'], splitdir)
                    except:
                        print('Warning: could not copy best preprocessor to splitdir')

                    if model_name == 'KerasRegressor':
                        try:
                            shutil.copy(best_split_dict['model'], splitdir)
                        except:
                            print('Warning: could not move best Keras model to splitdir')
                    elif model_name == 'BaggingRegressor':
                        if model.base_estimator_ == 'KerasRegressor':
                            for m in best_split_dict['model']:
                                shutil.move(m, splitdir)
                        else:
                            shutil.copy(best_split_dict['model'], splitdir)
                    else:
                        try:
                            shutil.copy(best_split_dict['model'], splitdir)
                        except:
                            print('Warning: could not copy best model to splitdir')
                    try:
                        shutil.copy(best_split_dict['features'], splitdir)
                    except:
                        print('Warning: could not copy best feature set to splitdir')

                    # Remove all the splitdirs if set to True
                    if remove_split_dirs == True:
                        ds = os.listdir(splitdir)
                        splitdirs = [d for d in ds if 'split_' in d and '.png' not in d and '.json' not in d]
                        for d in splitdirs:
                            shutil.rmtree(os.path.join(splitdir, d))

        return

    def _evaluate_split_sets(self, X_splits, y_splits, train_inds, test_inds, model, model_name, selector, preprocessor,
                             X_extra, groups, splitdir, hyperopt, metrics, plots, has_model_errors, error_method,
                             remove_outlier_learners, recalibrate_errors, verbosity, baseline_test, distance_metric,
                             file_extension, image_dpi, parallel_run, number_of_bins, equal_sized_bins, domain, recalibrate_power,
                             recalibrate_per_bin):

        def _evaluate_split_sets_serial(data, verbosity, groups=None, domain=None):

            Xs, ys, train_ind, test_ind, split_count = data
            # TODO: not copying this causes issues with KerasRegressor when doing different split types. But, doing this breaks BaggingRegressor with KerasRegressor networks
            #model_orig = copy.deepcopy(model)
            selector_orig = copy.deepcopy(selector)
            preprocessor_orig = copy.deepcopy(preprocessor)
            hyperopt_orig = copy.deepcopy(hyperopt)

            X_train = Xs[0]
            X_test = Xs[1]
            y_train = pd.Series(np.array(ys[0]).ravel(), name='y_train')
            y_test = pd.Series(np.array(ys[1]).ravel(), name='y_test')  # Make it so the y_test and y_pred have same indices so can be subtracted to get residual

            if X_extra is not None:
                try:
                    X_extra_train = X_extra.loc[X_train.index.values, :]
                except:
                    print('Could not write X_extra_train with forcing X_train data- too many indices')
                    X_extra_train = None
                X_extra_test = X_extra.loc[X_test.index.values, :]
            else:
                X_extra_train = None
                X_extra_test = None

            if groups is not None:
                group = pd.Series(np.array(groups)[test_ind])
                group_train = pd.Series(np.array(groups)[train_ind])
            else:
                group = None
                group_train = None

            splitpath = os.path.join(splitdir, 'split_' + str(split_count))
            os.mkdir(splitpath)

            # Save the test data indices and train data indices
            if verbosity >= 0:
                if file_extension == '.xlsx':
                    pd.DataFrame({'test_inds': test_ind}).to_excel(os.path.join(splitpath, 'test_inds'+file_extension), index=False)
                    pd.DataFrame({'train_inds': train_ind}).to_excel(os.path.join(splitpath, 'train_inds'+file_extension), index=False)
                elif file_extension == '.csv':
                    pd.DataFrame({'test_inds': test_ind}).to_csv(os.path.join(splitpath, 'test_inds' + file_extension), index=False)
                    pd.DataFrame({'train_inds': train_ind}).to_csv(os.path.join(splitpath, 'train_inds' + file_extension), index=False)

            split_summary = self._evaluate_split(X_train, X_test, y_train, y_test, model, model_name, preprocessor_orig,
                                 selector_orig,
                                 hyperopt_orig, metrics, plots, group, group_train,
                                 splitpath, has_model_errors, X_extra_train, X_extra_test, error_method,
                                 remove_outlier_learners,
                                 verbosity, baseline_test, distance_metric, file_extension, image_dpi,
                                 domain, train_ind, test_ind)

            return split_summary, split_count

        split_counts = list(range(len(y_splits)))
        data = list(zip(X_splits, y_splits, train_inds, test_inds, split_counts))

        # Parallel
        if parallel_run is True:
            # TODO: how to get split summary list from this?
            split_summary_dict = parallel(
                                          _evaluate_split_sets_serial,
                                          x=data,
                                          verbosity=verbosity,
                                          groups=groups,
                                          domain=domain,
                                          )
        # Serial
        else:
            split_summary_dict = []
            for i in tqdm(data, 'Running splits'):
                summary = _evaluate_split_sets_serial(data=i, verbosity=verbosity, groups=groups, domain=domain)
                split_summary_dict.append(summary)

        split_summary_dict = {i[1]: i[0] for i in split_summary_dict}

        # At level of splitdir, do analysis over all splits (e.g. parity plot over all splits)
        if verbosity >= 0:
            y_test_all = self._collect_data(filename='y_test', savepath=splitdir, file_extension=file_extension)
            y_train_all = self._collect_data(filename='y_train', savepath=splitdir, file_extension=file_extension)
            y_pred_all = self._collect_data(filename='y_pred', savepath=splitdir, file_extension=file_extension)
            y_pred_train_all = self._collect_data(filename='y_pred_train', savepath=splitdir, file_extension=file_extension)
            residuals_test_all = self._collect_data(filename='residuals_test', savepath=splitdir, file_extension=file_extension)
            residuals_train_all = self._collect_data(filename='residuals_train', savepath=splitdir, file_extension=file_extension)
            X_train_all = self._collect_df_data(filename='X_train', savepath=splitdir, file_extension=file_extension)
            X_test_all = self._collect_df_data(filename='X_test', savepath=splitdir, file_extension=file_extension)

            if groups is not None:
                groups_test_all = self._collect_data(filename='test_groups', savepath=splitdir, file_extension=file_extension)
                groups_train_all = self._collect_data(filename='train_groups', savepath=splitdir, file_extension=file_extension)
            else:
                groups_test_all = None
                groups_train_all = None

            if domain is not None:
                domains_train = self._collect_df_data(filename='domains_train', savepath=splitdir, file_extension=file_extension)
                domains_test = self._collect_df_data(filename='domains_test', savepath=splitdir, file_extension=file_extension)

            if X_extra is not None:
                try:
                    X_extra_train_all = self._collect_df_data(filename='X_extra_train', savepath=splitdir, file_extension=file_extension)
                except:
                    print('No X_extra_train data to collect')
                X_extra_test_all = self._collect_df_data(filename='X_extra_test', savepath=splitdir, file_extension=file_extension)

            # Save the data gathered over all the splits
            self._save_split_data(df=X_train_all, filename='X_train', savepath=splitdir, columns=X_train_all.columns.tolist(), file_extension=file_extension)
            self._save_split_data(df=X_test_all, filename='X_test', savepath=splitdir, columns=X_test_all.columns.tolist(), file_extension=file_extension)
            if X_extra is not None:
                try:
                    self._save_split_data(df=X_extra_train_all, filename='X_extra_train', savepath=splitdir, columns=X_extra_train_all.columns.tolist(), file_extension=file_extension)
                except:
                    print('No X_extra_train data to save')
                self._save_split_data(df=X_extra_test_all, filename='X_extra_test', savepath=splitdir, columns=X_extra_test_all.columns.tolist(), file_extension=file_extension)
            self._save_split_data(df=y_test_all, filename='y_test', savepath=splitdir, columns='y_test', file_extension=file_extension)
            self._save_split_data(df=y_train_all, filename='y_train', savepath=splitdir, columns='y_train', file_extension=file_extension)
            self._save_split_data(df=y_pred_all, filename='y_pred', savepath=splitdir, columns='y_pred', file_extension=file_extension)
            self._save_split_data(df=y_pred_train_all, filename='y_pred_train', savepath=splitdir, columns='y_pred_train', file_extension=file_extension)

            if domain is not None:
                self._save_split_data(df=domains_train, filename='domains_train', savepath=splitdir, columns=domains_train.columns.tolist(), file_extension=file_extension)
                self._save_split_data(df=domains_test, filename='domains_test', savepath=splitdir, columns=domains_test.columns.tolist(), file_extension=file_extension)

            if has_model_errors is True:
                model_errors_test_all = self._collect_data(filename='model_errors_test', savepath=splitdir, file_extension=file_extension)
                model_errors_train_all = self._collect_data(filename='model_errors_train', savepath=splitdir, file_extension=file_extension)
                num_removed_learners_test_all = self._collect_data(filename='num_removed_learners_test', savepath=splitdir, file_extension=file_extension)
                num_removed_learners_train_all = self._collect_data(filename='num_removed_learners_train', savepath=splitdir, file_extension=file_extension)
                # Save all the uncalibrated model errors data
                self._save_split_data(df=model_errors_train_all, filename='model_errors_train', savepath=splitdir, columns='model_errors', file_extension=file_extension)
                self._save_split_data(df=num_removed_learners_test_all, filename='num_removed_learners_test', savepath=splitdir, columns='num_removed_learners', file_extension=file_extension)
                self._save_split_data(df=num_removed_learners_train_all, filename='num_removed_learners_train', savepath=splitdir, columns='num_removed_learners', file_extension=file_extension)
            else:
                model_errors_test_all = None
                model_errors_train_all = None

        if recalibrate_errors is True and has_model_errors is True:
            err_test_all = list()
            res_test_all = list()
            # TODO: any issue with dict ordering here?
            for i, v in split_summary_dict.items():
                err_test_all.append(v['model_errors_test'])
                res_test_all.append(v['preds']-v['trues'])
            #model_errors_test_all = [item for sublist in err_test_all for item in sublist]
            #residuals_test_all = [item for sublist in res_test_all for item in sublist]
            model_errors_test_all = np.concatenate(err_test_all)
            residuals_test_all = np.concatenate(res_test_all)

            if recalibrate_per_bin == True:
                model_errors_test_all_cal, recal_dict, bin_numbers, residuals_test_all, model_errors_test_all = ErrorUtils()._recalibrate_errors(model_errors=model_errors_test_all,
                                                                                     residuals=residuals_test_all,
                                                                                     power=recalibrate_power,
                                                                                     recal_per_bin=recalibrate_per_bin,
                                                                                     number_of_bins=number_of_bins,
                                                                                     equal_sized_bins=equal_sized_bins,
                                                                                     dataset_stdev=np.std(y_test_all))

                if type(model_errors_test_all_cal) is list and type(residuals_test_all) is list:
                    # TODO: issue here where model errors now in different order from before, and are detached from their respective residuals?? Need to do for both I think
                    model_errors_test_all_cal = pd.Series(list(itertools.chain.from_iterable(model_errors_test_all_cal)), name='model_errors')
                    residuals_test_all = pd.Series(list(itertools.chain.from_iterable(residuals_test_all)), name='residuals')
                    model_errors_test_all = pd.Series(list(itertools.chain.from_iterable(model_errors_test_all)), name='model_errors')
            else:
                model_errors_test_all_cal, recal_dict, bin_numbers, _, __ = ErrorUtils()._recalibrate_errors(
                    model_errors=model_errors_test_all,
                    residuals=residuals_test_all,
                    power=recalibrate_power,
                    recal_per_bin=recalibrate_per_bin,
                    number_of_bins=number_of_bins,
                    equal_sized_bins=equal_sized_bins,
                    dataset_stdev=np.std(y_test_all))

            # Write the recalibration values to file
            if recalibrate_per_bin == False:
                recal_df = pd.DataFrame(recal_dict, index=[0])
            else:
                recal_df = pd.DataFrame(recal_dict)
                #recal_df['bin_numbers'] = bin_numbers
            if file_extension == '.xlsx':
                recal_df.to_excel(os.path.join(splitdir, 'recalibration_parameters_'+str('test')+file_extension), index=False)
            elif file_extension == '.csv':
                recal_df.to_csv(os.path.join(splitdir, 'recalibration_parameters_' + str('test') + file_extension), index=False)

            self._save_split_data(df=model_errors_test_all_cal, filename='model_errors_test_calibrated',
                                  savepath=splitdir,
                                  columns='model_errors', file_extension=file_extension)

            # Write the calibrated model errors to file
            if verbosity >= 0:

                if recalibrate_per_bin == True:
                    model_errors_train_all_cal, recal_dict, bin_numbers, residuals_train_all, model_errors_train_all = ErrorUtils()._recalibrate_errors(model_errors=model_errors_train_all,
                                                                                           residuals=residuals_train_all,
                                                                                          power=recalibrate_power,
                                                                                          recal_per_bin=recalibrate_per_bin,
                                                                                          number_of_bins=number_of_bins,
                                                                                          equal_sized_bins=equal_sized_bins,
                                                                                          dataset_stdev=np.std(y_train_all))

                    if type(model_errors_train_all_cal) is list and type(residuals_train_all) is list:
                        # TODO: issue here where model errors now in different order from before, and are detached from their respective residuals?? Need to do for both I think
                        model_errors_train_all_cal = pd.Series(list(itertools.chain.from_iterable(model_errors_train_all_cal)), name='model_errors')
                        residuals_train_all = pd.Series(list(itertools.chain.from_iterable(residuals_train_all)), name='residuals')
                        model_errors_train_all = pd.Series(list(itertools.chain.from_iterable(model_errors_train_all)), name='model_errors')
                else:
                    model_errors_train_all_cal, recal_dict, bin_numbers, _, __ = ErrorUtils()._recalibrate_errors(
                        model_errors=model_errors_train_all,
                        residuals=residuals_train_all,
                        power=recalibrate_power,
                        recal_per_bin=recalibrate_per_bin,
                        number_of_bins=number_of_bins,
                        equal_sized_bins=equal_sized_bins,
                        dataset_stdev=np.std(y_train_all))

                # Write the recalibration values to file
                if recalibrate_per_bin == False:
                    recal_df = pd.DataFrame(recal_dict, index=[0])
                else:
                    recal_df = pd.DataFrame(recal_dict)
                    #recal_df['bin_numbers'] = bin_numbers
                if file_extension == '.xlsx':
                    recal_df.to_excel(os.path.join(splitdir, 'recalibration_parameters_' + str('train') + file_extension), index=False)
                elif file_extension == '.csv':
                    recal_df.to_csv(os.path.join(splitdir, 'recalibration_parameters_' + str('train') + file_extension), index=False)
                # Write the calibrated model errors to file
                #if recalibrate_per_bin == False:
                self._save_split_data(df=model_errors_train_all_cal, filename='model_errors_train_calibrated', savepath=splitdir,
                                      columns='model_errors', file_extension=file_extension)
                #else:
                #    for i, data in enumerate(model_errors_train_all_cal):
                #        self._save_split_data(df=data, filename='model_errors_train_calibrated_bin_'+str(i), savepath=splitdir,
                #                              columns='model_errors', file_extension=file_extension)
        else:
            model_errors_test_all_cal = None
            model_errors_train_all_cal = None

        # Make all test data plots
        if verbosity >= 1:
            dataset_stdev = np.std(np.unique(y_train_all))
            make_plots(plots=plots,
                       y_true=y_test_all,
                       y_pred=y_pred_all,
                       X_test=X_train_all,
                       groups=groups_test_all,
                       data_type='test',
                       dataset_stdev=dataset_stdev,
                       has_model_errors=has_model_errors,
                       metrics=metrics,
                       model=model,
                       model_errors=model_errors_test_all,
                       residuals=residuals_test_all,
                       savepath=splitdir,
                       show_figure=False,
                       recalibrate_errors=recalibrate_errors,
                       model_errors_cal=model_errors_test_all_cal,
                       splits_summary=True,
                       file_extension=file_extension,
                       image_dpi=image_dpi,
                       number_of_bins=number_of_bins,
                       equal_sized_bins=equal_sized_bins)

        # Make all train data plots
        if verbosity >= 1:
            dataset_stdev = np.std(np.unique(y_train_all))
            make_plots(plots=plots,
                       y_true=y_train_all,
                       y_pred=y_pred_train_all,
                       X_test=X_test_all,
                       groups=groups_train_all,
                       data_type='train',
                       dataset_stdev=dataset_stdev,
                       has_model_errors=has_model_errors,
                       metrics=metrics,
                       model=model,
                       model_errors=model_errors_train_all,
                       residuals=residuals_train_all,
                       savepath=splitdir,
                       show_figure=False,
                       recalibrate_errors=recalibrate_errors,
                       model_errors_cal=model_errors_train_all_cal,
                       splits_summary=True,
                       file_extension=file_extension,
                       image_dpi=image_dpi,
                       number_of_bins=number_of_bins,
                       equal_sized_bins=equal_sized_bins)

        try:
            if file_extension == '.xlsx':
                df_stats = pd.read_excel(os.path.join(splitdir, 'test_average_stdev_stats_summary'+file_extension))
            elif file_extension == '.csv':
                df_stats = pd.read_csv(os.path.join(splitdir, 'test_average_stdev_stats_summary' + file_extension))
        except:
            df_stats = None
        try:
            if file_extension == '.xlsx':
                df_stats_train = pd.read_excel(os.path.join(splitdir, 'train_average_stdev_stats_summary'+file_extension))
            elif file_extension == '.csv':
                df_stats_train = pd.read_csv(os.path.join(splitdir, 'train_average_stdev_stats_summary' + file_extension))
        except:
            df_stats_train = None

        # Update the MASTML metadata file
        outerdir = splitdir.split('/')[-1]
        '''
        if mastml is not None:
            if 'split_outer' in outerdir:
                # For nested CV or left out data runs with outer and inner splits, need the model dir one level up
                outerdir = os.path.join(splitdir.split('/')[-2], splitdir.split('/')[-1])
            
            if verbosity >= 0:
                mastml._update_metadata(outerdir=outerdir,
                                        split_name='split_summary',
                                        model=model,
                                        splitter=self,
                                        preprocessor=preprocessor,
                                        selector=selector,
                                        hyperopt=hyperopt,
                                        train_stats=df_stats_train,
                                        test_stats=df_stats,
                                        #X_train=pd.DataFrame(np.array(X_train_all), columns=X_train_all.columns.tolist()),
                                        #X_test=pd.DataFrame(np.array(X_test_all), columns=X_test_all.columns.tolist()),
                                        #X_extra_train=pd.DataFrame(np.array(X_extra_train_all), columns=X_extra_train_all.columns.tolist()) if X_extra is not None else None,
                                        #X_extra_test=pd.DataFrame(np.array(X_extra_test_all), columns=X_extra_test_all.columns.tolist()) if X_extra is not None else None,
                                        train_inds=train_inds,
                                        test_inds=test_inds,
                                        y_train=y_train_all,
                                        y_test=y_test_all,
                                        y_pred_train=y_pred_train_all,
                                        y_pred=y_pred_all,
                                        residuals_train=residuals_train_all,
                                        residuals_test=residuals_test_all,
                                        model_errors_train=model_errors_train_all,
                                        model_errors_test=model_errors_test_all,
                                        model_errors_train_cal=model_errors_train_all_cal,
                                        model_errors_test_cal=model_errors_test_all_cal)
                                        #dataset_stdev=dataset_stdev)
                mastml._save_mastml_metadata()
        '''
        return outerdir, split_summary_dict

    def _evaluate_split(self, X_train, X_test, y_train, y_test, model, model_name, preprocessor, selector, hyperopt,
                        metrics, plots, groups, groups_train, splitpath, has_model_errors, X_extra_train, X_extra_test,
                        error_method, remove_outlier_learners, verbosity, baseline_test, distance_metric,
                        file_extension, image_dpi, domain, train_ind, test_ind):

        split_summary = dict()
        split_summary['train_ind'] = np.array(train_ind)
        split_summary['test_ind'] = np.array(test_ind)

        X_train_orig = copy.deepcopy(X_train)
        X_test_orig = copy.deepcopy(X_test)

        preprocessor1 = copy.deepcopy(preprocessor)
        preprocessor2 = copy.deepcopy(preprocessor)

        # Preprocess the full split data
        X_train = preprocessor1.evaluate(X_train, savepath=splitpath, file_name='train', file_extension=file_extension, verbosity=verbosity)
        #X_test = preprocessor1.evaluate(X_test, savepath=splitpath, file_name='test', file_extension=file_extension, verbosity=verbosity)

        # run feature selector to get new Xtrain, Xtest
        X_train = selector.evaluate(X=X_train, y=y_train, savepath=splitpath, verbosity=verbosity)
        selected_features = selector.selected_features
        # TODO: this should be X_test scaled like X_train ?? Yes b/c domains_train and domains_test for full fit should be same and they are not
        #X_test = X_test_orig[selected_features]
        # Now preprocess the feature-selected data
        X_train = preprocessor2.evaluate(X_train_orig[selected_features], savepath=splitpath, file_name='train_selected', file_extension=file_extension, verbosity=verbosity)
        X_test = preprocessor2.transform(X_test_orig[selected_features])
        #X_test = preprocessor1.transform(X_test_orig[selected_features])

        if domain is not None:
            domains_test = list()
            domains_train = list()
            domain_checks = list()
            for domain_type in domain:
                if domain_type[0] == 'elemental':
                    check = Domain(domain_type[0])
                    check.fit(domain_type[1][train_ind])
                    domain_checks.append(check)

                    domain_test = check.predict(domain_type[1][test_ind])
                    domains_test.append(domain_test)
                    domains_train.append(check.predict(domain_type[1][train_ind]))

                    split_summary['domain_elemental'] = np.array(domain_test).ravel()

                elif domain_type == 'gpr':
                    check = Domain(domain_type)
                    check.fit(X_train, y_train)
                    domain_checks.append(check)

                    domain_test = check.predict(X_test)
                    domains_test.append(domain_test)
                    domains_train.append(check.predict(X_train))

                    split_summary['domain_gpr'] = np.array(domain_test).ravel()

                elif domain_type == 'feature_range':
                    check = Domain(domain_type)
                    check.fit(X_train, y_train)
                    domain_checks.append(check)

                    domains_test.append(check.predict(X_test))
                    domains_train.append(check.predict(X_train))

                elif domain_type[0] == 'madml':
                    check = Domain(
                                   domain_type[0],
                                   preprocessor,
                                   model,
                                   domain_type[1],
                                   splitpath
                                   )
                    check.fit(X_train, y_train)
                    domain_checks.append(check)

                    domains_test.append(check.predict(X_test))
                    domains_train.append(check.predict(X_train))

            domains_test = pd.concat(domains_test, axis=1)
            domains_train = pd.concat(domains_train, axis=1)

        split_summary['trues'] = np.array(y_test)
        if verbosity >= 0:
            self._save_split_data(df=X_train_orig[selected_features], filename='X_train', savepath=splitpath, columns=selected_features, file_extension=file_extension)
            self._save_split_data(df=X_test_orig[selected_features], filename='X_test', savepath=splitpath, columns=selected_features, file_extension=file_extension)
            self._save_split_data(df=y_train, filename='y_train', savepath=splitpath, columns='y_train', file_extension=file_extension)
            self._save_split_data(df=y_test, filename='y_test', savepath=splitpath, columns='y_test', file_extension=file_extension)

            if domain is not None:
                self._save_split_data(df=domains_train, filename='domains_train', savepath=splitpath, columns=domains_train.columns.tolist(), file_extension=file_extension)
                self._save_split_data(df=domains_test, filename='domains_test', savepath=splitpath, columns=domains_test.columns.tolist(), file_extension=file_extension)

            if X_extra_train is not None:
                self._save_split_data(df=X_extra_train, filename='X_extra_train', savepath=splitpath, columns=X_extra_train.columns.tolist(), file_extension=file_extension)
            else:
                X_extra_train = None
            if X_extra_test is not None:
                self._save_split_data(df=X_extra_test, filename='X_extra_test', savepath=splitpath, columns=X_extra_test.columns.tolist(), file_extension=file_extension)
            else:
                X_extra_test = None

        # Here evaluate hyperopt instance, if provided, and get updated model instance
        if hyperopt is not None:
            model = hyperopt.fit(X=X_train, y=y_train, model=model, cv=5, savepath=splitpath)

        if model_name in ['Model', 'CrabNetModel', 'SourceNN', 'Transfer']:
            model.savepath = splitpath
        try:
            model.fit(X_train, y_train)
        except:
            model.fit(np.array(X_train).astype(np.float32), np.array(y_train).reshape(-1, 1).astype(np.float32)) # Need y to be 2D for PyTorch model fitting

        try:
            y_pred = model.predict(X_test)
        except:
            y_pred = model.predict(np.array(X_test).astype(np.float32)) # For PyTorch model
        try:
            y_pred_train = model.predict(X_train)
        except:
            y_pred_train = model.predict(np.array(X_train).astype(np.float32))

        y_pred = pd.Series(y_pred.ravel(), name='y_pred')
        y_pred_train = pd.Series(y_pred_train.ravel(), name='y_pred_train')

        residuals_test = y_pred-y_test
        residuals_train = y_pred_train-y_train

        split_summary['preds'] = np.array(y_pred)
        if verbosity >= 0:
            self._save_split_data(df=y_pred, filename='y_pred', savepath=splitpath, columns='y_pred', file_extension=file_extension)
            self._save_split_data(df=y_pred_train, filename='y_pred_train', savepath=splitpath, columns='y_pred_train', file_extension=file_extension)
            self._save_split_data(df=residuals_test, filename='residuals_test', savepath=splitpath, columns='residuals', file_extension=file_extension)
            self._save_split_data(df=residuals_train, filename='residuals_train', savepath=splitpath, columns='residuals', file_extension=file_extension)

        if has_model_errors is True:
            model_errors_test, num_removed_learners_test = ErrorUtils()._get_model_errors(model=model,
                                                                X=X_test,
                                                                X_train=X_train,
                                                                X_test=X_test,
                                                               error_method=error_method,
                                                               remove_outlier_learners=remove_outlier_learners)
            split_summary['model_errors_test'] = np.array(model_errors_test)
            if verbosity >= 0:
                self._save_split_data(df=model_errors_test, filename='model_errors_test', savepath=splitpath, columns='model_errors', file_extension=file_extension)
                self._save_split_data(df=num_removed_learners_test, filename='num_removed_learners_test', savepath=splitpath, columns='num_removed_learners', file_extension=file_extension)

            model_errors_train, num_removed_learners_train = ErrorUtils()._get_model_errors(model=model,
                                                                X=X_train,
                                                                X_train=X_train,
                                                                X_test=X_train, #predicting on train data so test/train is same
                                                                error_method=error_method,
                                                                remove_outlier_learners=remove_outlier_learners)
            if verbosity >= 0:
                self._save_split_data(df=model_errors_train, filename='model_errors_train', savepath=splitpath, columns='model_errors', file_extension=file_extension)
                self._save_split_data(df=num_removed_learners_train, filename='num_removed_learners_train', savepath=splitpath, columns='num_removed_learners', file_extension=file_extension)
        else:
            model_errors_test = None
            model_errors_train = None
            num_removed_learners_train = None
            num_removed_learners_test = None

        # Save summary stats data for this split
        stats_dict = Metrics(metrics_list=metrics).evaluate(y_true=y_test, y_pred=y_pred)
        split_summary['stats'] = stats_dict
        df_stats = pd.DataFrame().from_records([stats_dict])
        stats_dict_train = Metrics(metrics_list=metrics).evaluate(y_true=y_train, y_pred=y_pred_train)
        df_stats_train = pd.DataFrame().from_records([stats_dict_train])

        #if verbosity >= 0:
        if file_extension == '.xlsx':
            df_stats.to_excel(os.path.join(splitpath, 'test_stats_summary' + file_extension), index=False)
            df_stats_train.to_excel(os.path.join(splitpath, 'train_stats_summary'+file_extension), index=False)
        elif file_extension == '.csv':
            df_stats.to_csv(os.path.join(splitpath, 'test_stats_summary' + file_extension), index=False)
            df_stats_train.to_csv(os.path.join(splitpath, 'train_stats_summary'+file_extension), index=False)

        # Make all test data plots
        dataset_stdev = np.std(y_train)
        if verbosity >= 2:
            make_plots(plots=plots,
                       y_true=y_test,
                       y_pred=y_pred,
                       X_test=X_test,
                       groups=groups,
                       data_type='test',
                       dataset_stdev=dataset_stdev,
                       has_model_errors=has_model_errors,
                       metrics=metrics,
                       model=model,
                       model_errors=model_errors_test,
                       residuals=residuals_test,
                       savepath=splitpath,
                       show_figure=False,
                       recalibrate_errors=False,
                       splits_summary=False,
                       file_extension=file_extension,
                       image_dpi=image_dpi)

        # Make all train data plots
        dataset_stdev = np.std(y_train)
        if verbosity >= 2:
            make_plots(plots=plots,
                       y_true=y_train,
                       y_pred=y_pred_train,
                       X_test=X_test,
                       groups=groups_train, # HERE groups_train
                       data_type='train',
                       dataset_stdev=dataset_stdev,
                       has_model_errors=has_model_errors,
                       metrics=metrics,
                       model=model,
                       model_errors=model_errors_train,
                       residuals=residuals_train,
                       savepath=splitpath,
                       show_figure=False,
                       recalibrate_errors=False,
                       splits_summary=False,
                       file_extension=file_extension,
                       image_dpi=image_dpi)

        # Write the test group to a text file,
        if groups is not None:
            unique_groups = np.unique(groups)
            groups.name = 'test_groups'
            groups_train.name = 'train_groups'
            if verbosity >= 0:
                if file_extension == '.xlsx':
                    groups.to_excel(os.path.join(splitpath, 'test_groups.xlsx'), index=False)
                    groups_train.to_excel(os.path.join(splitpath, 'train_groups.xlsx'), index=False)
                elif file_extension == '.csv':
                    groups.to_csv(os.path.join(splitpath, 'test_groups.csv'), index=False)
                    groups_train.to_csv(os.path.join(splitpath, 'train_groups.csv'), index=False)
                with open(os.path.join(splitpath, 'test_group.txt'), 'w') as f:
                    for group in unique_groups:
                        f.write(str(group)+'\n')
        else:
            groups_train = None
            groups = None

        if domain is not None:
            for check in domain_checks:
                try:
                    joblib.dump(check, open(os.path.join(splitpath, 'domain_'+str(check.check_type)+'.pkl'), 'wb'))
                except:
                    dill.dump(check, open(os.path.join(splitpath, 'domain_'+str(check.check_type)+'.pkl'), 'wb'))

        # Save the fitted model, will be needed for DLHub upload later on
        if model_name == 'KerasRegressor':
            joblib.dump(model, os.path.join(splitpath, str(model_name) + ".pkl"))
            #if not os.path.exists(os.path.join(splitpath, 'keras_model')):
            #    os.mkdir(os.path.join(splitpath, 'keras_model'))
            #model.model.save(os.path.join(splitpath, 'keras_model'))
        elif model_name == 'BaggingRegressor':
            # Edge case of ensemble of keras models- pickle won't work here. Just state warning
            if model.base_estimator_ == 'KerasRegressor':
            #if model.model.estimators_[0].__class__.__name__=='KerasRegressor':
                # Save the individual Keras models comprising the Ensemble
                count = 0
                for m in model.model.estimators_:
                    m.model_.save(filepath=os.path.join(splitpath,'keras_model_' + str(count)+'.keras'))
                    #m.model.save(filepath=os.path.join(splitpath, 'keras_model_' + str(count)))
                    count += 1
                #print('Warning: unable to save pickled model of ensemble of KerasRegressor models. Passing through...')
            else:
                joblib.dump(model, os.path.join(splitpath, str(model_name) + ".pkl"))
        elif model_name == 'NeuralNetRegressor':
            # This is a PyTorch model
            import torch
            torch.save(model, os.path.join(splitpath, str(model_name) + ".pkl"))
        elif model_name == 'Model': #Once instantiated, the CrabNet model is just called "Model"
            print('CrabNet model is already saved')
        elif model_name == 'KANModel': # Save the KAN model
            model.save_ckpt('model_ckpt')
            shutil.move('model_ckpt', os.path.join(splitpath, 'model_ckpt'))
        else:
            joblib.dump(model, os.path.join(splitpath, str(model_name) + ".pkl"))

        # If using a Keras model, need to clear the session so training multiple models doesn't slow training down
        if model_name == 'KerasRegressor':
            print('HERE CLEARING SESSION')
            keras.backend.clear_session()
        elif model_name == 'BaggingRegressor':
            if model.base_estimator_.__class__.__name__ == 'KerasRegressor':
                keras.backend.clear_session()

        if baseline_test is not None:
            baseline = Baseline_tests()
            columns = ["Metric", "Real_score", "Fake_score"]
            # ["test_mean", "test_permuted", "test_nearest_neighbour_kdTree", "test_nearest_neighbour_cdist",
            # "test_classifier_random", "test_classifier_dominant"]
            for i in baseline_test:

                if (i == "test_mean"):
                    df_res = baseline.test_mean(X_train, X_test, y_train, y_test, model, metrics)
                    self._save_split_data(df_res, "test_mean", splitpath, columns, file_extension)

                elif (i == "test_permuted"):
                    df_res = baseline.test_permuted(X_train, X_test, y_train, y_test, model, metrics)
                    self._save_split_data(df_res, "test_permuted", splitpath, columns, file_extension)

                elif (i == "test_nearest_neighbour_kdTree"):
                    df_res = baseline.test_nearest_neighbour_kdtree(X_train, X_test, y_train, y_test, model, metrics)
                    self._save_split_data(df_res, "test_nearest_neighbour_kdTree", splitpath, columns, file_extension)

                elif (i == "test_nearest_neighbour_cdist"):
                    df_res = baseline.test_nearest_neighbour_cdist(X_train, X_test, y_train, y_test, model, metrics,
                                                                   distance_metric)
                    self._save_split_data(df_res, "test_nearest_neighbour_cdist", splitpath, columns, file_extension)

                elif (i == "test_classifier_random"):
                    df_res = baseline.test_classifier_random(X_train, X_test, y_train, y_test, model, metrics)
                    self._save_split_data(df_res, "test_classifier_random", splitpath, columns, file_extension)

                elif (i == "test_classifier_dominant"):
                    df_res = baseline.test_classifier_dominant(X_train, X_test, y_train, y_test, model, metrics)
                    self._save_split_data(df_res, "test_classifier_dominant", splitpath, columns, file_extension)

        # Make combined spreadsheet that contains: y_test, y_pred, X_extra_test if user doesn't specify domain
        else:
            y_combined = []
            for i in range(len(y_test)):
                y_combined.append((y_test.iloc[i], y_pred.iloc[i], X_extra_test))
            if verbosity >= 0:
                self._save_split_data(pd.DataFrame(y_combined), "y_combined", splitpath,
                                  columns=["y_test", "y_pred", "X_extra_test"], file_extension=file_extension)

        # Update the MASTML metadata file
        '''
        if mastml is not None:
            outerdir = splitpath.split('/')[-2]
            if 'split_outer' in outerdir:
                # For nested CV or left out data runs with outer and inner splits, need the model dir one level up
                outerdir = os.path.join(splitpath.split('/')[-3], splitpath.split('/')[-2])
            
            if verbosity >= 0:
                mastml._update_metadata(outerdir=outerdir,
                                        split_name=splitpath.split('/')[-1],
                                        model=model,
                                        splitter=self,
                                        preprocessor=preprocessor,
                                        selector=selector,
                                        hyperopt=hyperopt,
                                        test_stats=df_stats,
                                        train_stats=df_stats_train,
                                        train_inds=train_ind,
                                        test_inds=test_ind,
                                        #X_train=X_train,
                                        #X_test=X_test,
                                        #X_extra_train=X_extra_train,
                                        #X_extra_test=X_extra_test,
                                        y_train=y_train,
                                        y_test=y_test,
                                        y_pred_train=y_pred_train,
                                        y_pred=y_pred,
                                        residuals_train=residuals_train,
                                        residuals_test=residuals_test,
                                        model_errors_train=model_errors_train,
                                        model_errors_test=model_errors_test,
                                        dataset_stdev=dataset_stdev,
                                        )
                mastml._save_mastml_metadata()
        '''

        return split_summary

    def _setup_savedir(self, model, selector, preprocessor, savepath):
        now = datetime.now()
        try:
            model_name = model.model.__class__.__name__
        except:
            model_name = model.__class__.__name__

        dirname = model_name+'_'+self.__class__.__name__+'_'+preprocessor.__class__.__name__+'_'+selector.__class__.__name__
        dirname = f"{dirname}_{now.year:02d}_{now.month:02d}_{now.day:02d}" \
            f"_{now.hour:02d}_{now.minute:02d}_{now.second:02d}"
        if savepath == None:
            splitdir = os.getcwd()
        else:
            splitdir = os.path.join(savepath, dirname)
        if not os.path.exists(splitdir):
            os.mkdir(splitdir)
        try:
            self.splitdir = splitdir
        except AttributeError:
            pass
        return splitdir

    def _save_split_data(self, df, filename, savepath, columns, file_extension):
        if type(df) == pd.core.frame.DataFrame:
            df.columns = columns
        if type(df) == pd.core.series.Series:
            df.name = columns
        if file_extension == '.xlsx':
            df.to_excel(os.path.join(savepath, filename)+'.xlsx', index=False)
        elif file_extension == '.csv':
            df.to_csv(os.path.join(savepath, filename) + '.csv', index=False)
        return

    def _collect_data(self, filename, savepath, file_extension, iterdirs=True):
        # Note this is only meant for collecting y-data (single column)
        if iterdirs == True:
            dirs = [d for d in os.listdir(savepath) if 'split' in d and '.png' not in d and file_extension not in d and '.json' not in d]
        else:
            dirs = ['']
        data = list()
        if 'residuals' in filename:
            col_name = 'residuals'
        elif 'model_errors' in filename:
            col_name = 'model_errors'
        elif 'num_removed_learners' in filename:
            col_name = 'num_removed_learners'
        else:
            col_name = filename

        # Condition to evaluate in parallel
        if self.parallel_run is True:
            if file_extension == '.xlsx':
                data = parallel(lambda d: np.array(pd.read_excel(os.path.join(savepath, os.path.join(d, filename)+'.xlsx'), engine='openpyxl')[col_name]), dirs)
            elif file_extension == '.csv':
                data = parallel(lambda d: np.array(pd.read_csv(os.path.join(savepath, os.path.join(d, filename) + '.csv'))[col_name]), dirs)
        else:
            for d in dirs:
                if file_extension == '.xlsx':
                    data.append(np.array(pd.read_excel(os.path.join(savepath, os.path.join(d, filename)+'.xlsx'), engine='openpyxl')[col_name]))
                elif file_extension == '.csv':
                    data.append(np.array(pd.read_csv(os.path.join(savepath, os.path.join(d, filename) + '.csv'))[col_name]))
        df = pd.Series(np.concatenate(data).ravel())
        return df

    def _collect_df_data(self, filename, savepath, file_extension):
        dirs = [d for d in os.listdir(savepath) if 'split' in d and '.png' not in d and file_extension not in d and '.json' not in d]
        data = list()

        # Condition to evaluate in parallel
        if self.parallel_run is True:
            if file_extension == '.xlsx':
                data = parallel(lambda d: pd.read_excel(os.path.join(savepath, os.path.join(d, filename)+'.xlsx'), engine='openpyxl'), dirs)
            elif file_extension == '.csv':
                data = parallel(lambda d: pd.read_csv(os.path.join(savepath, os.path.join(d, filename) + '.csv')), dirs)
        else:
            for d in dirs:
                if file_extension == '.xlsx':
                    data.append(pd.read_excel(os.path.join(savepath, os.path.join(d, filename)+'.xlsx'), engine='openpyxl'))
                elif file_extension == '.csv':
                    data.append(pd.read_csv(os.path.join(savepath, os.path.join(d, filename) + '.csv')))
        df = pd.concat(data)
        return df

    def _get_best_split(self, savepath, model, preprocessor, best_run_metric, model_name, file_extension):
        dirs = os.listdir(savepath)
        splitdirs = [d for d in dirs if 'split_' in d and '.png' not in d and 'summary' not in d]

        stats_files_dict = dict()
        for splitdir in splitdirs:
            if file_extension == '.xlsx':
                stats_files_dict[os.path.join(savepath, splitdir)] = pd.read_excel(os.path.join(os.path.join(savepath, splitdir), 'test_stats_summary.xlsx'), engine='openpyxl').to_dict('records')[0]
            elif file_extension == '.csv':
                stats_files_dict[os.path.join(savepath, splitdir)] = pd.read_csv(os.path.join(os.path.join(savepath, splitdir), 'test_stats_summary.csv')).to_dict('records')[0]
        # Find best/worst splits based on specified best run metric
        all_metrics = Metrics(metrics_list=[best_run_metric])._metric_zoo()
        greater_is_better = all_metrics[best_run_metric][0]

        if greater_is_better == False:
            metric_best = 10**20
        else:
            metric_best = 0

        # Start with best_split being the first split so doesn't throw an error later
        best_split = list(stats_files_dict.keys())[0]
        for split, stats_dict in stats_files_dict.items():
            if greater_is_better == False:
                if stats_dict[best_run_metric] < metric_best:
                    best_split = split
                    metric_best = stats_dict[best_run_metric]
            else:
                if stats_dict[best_run_metric] > metric_best:
                    best_split = split
                    metric_best = stats_dict[best_run_metric]

        # Get the preprocessor, model, and features for the best split
        best_split_dict = dict()
        preprocessor_name = preprocessor.preprocessor.__class__.__name__+'.pkl'

        bagging_with_keras = False
        #if "Keras" in model_name:
        #    model_name = 'keras_model'
        if "BaggingRegressor" in model_name:
            if model.base_estimator_ == 'KerasRegressor':
            #if model.model.estimators_[0].__class__.__name__=='KerasRegressor':
                bagging_with_keras = True
                model_name = 'keras_model'
            else:
                model_name = model_name + '.pkl'
        else:
            model_name = model_name+'.pkl'

        if not os.path.exists(os.path.join(best_split, preprocessor_name)):
            print("Warning: couldn't find preprocessor file in best split")

        if bagging_with_keras == True:
            keras_paths = [os.path.join(best_split, d) for d in os.listdir(best_split) if 'keras_model' in d]
            #print('keras paths', keras_paths)

        if bagging_with_keras == False:
            if not os.path.exists(os.path.join(best_split, model_name)):
                print("Warning: couldn't find model file in best split")

        best_split_dict['preprocessor'] = os.path.join(best_split, preprocessor_name)
        if bagging_with_keras == True:
            best_split_dict['model'] = keras_paths
        else:
            best_split_dict['model'] = os.path.join(best_split, model_name)
        best_split_dict['features'] = os.path.join(best_split, 'selected_features.txt')
        best_split_dict['X_train'] = os.path.join(best_split, 'X_train'+file_extension)

        # Remove the saved models and preprocessors from other split dirs that are not the best (save storage space)
        #for splitdir in splitdirs:
        #    if os.path.join(savepath, splitdir) != best_split:
        #        try:
        #            os.remove(os.path.join(os.path.join(savepath, splitdir), model_name))
        #        except:
        #            print("Warning: did not remove non-best models from split path")
        #        try:
        #            os.remove(os.path.join(os.path.join(savepath, splitdir), preprocessor_name))
        #        except:
        #            print("Warning: did not remove non-best preprocessors from split path")

        return best_split_dict

    def _get_average_recalibration_params(self, savepath, data_type, file_extension):
        dirs = os.listdir(savepath)
        splitdirs = [d for d in dirs if 'split_' in d and '.png' not in d and '.json' not in d]

        recalibrate_a_vals = dict()
        recalibrate_b_vals = dict()
        recalibrate_c_vals = dict()
        recalibrate_d_vals = dict()
        recalibrate_avg_dict = dict()
        recalibrate_stdev_dict = dict()
        for splitdir in splitdirs:
            if file_extension == '.xlsx':
                recalibrate_dict = pd.read_excel(os.path.join(os.path.join(savepath, splitdir),
                                                          'recalibration_parameters_'+str(data_type)+'.xlsx'), engine='openpyxl').to_dict('records')
            elif file_extension == '.csv':
                recalibrate_dict = pd.read_csv(os.path.join(os.path.join(savepath, splitdir),
                                                          'recalibration_parameters_'+str(data_type)+'.csv')).to_dict('records')
            for i, recal in enumerate(recalibrate_dict):
                if i not in recalibrate_a_vals:
                    recalibrate_a_vals[i] = list()
                if i not in recalibrate_b_vals:
                    recalibrate_b_vals[i] = list()
                if i not in recalibrate_c_vals:
                    recalibrate_c_vals[i] = list()
                if i not in recalibrate_d_vals:
                    recalibrate_d_vals[i] = list()
                if 'a' in recal.keys():
                    recalibrate_a_vals[i].append(recal['a'])
                if 'b' in recal.keys():
                    recalibrate_b_vals[i].append(recal['b'])
                if 'c' in recal.keys():
                    recalibrate_c_vals[i].append(recal['c'])
                if 'd' in recal.keys():
                    recalibrate_d_vals[i].append(recal['d'])
        if len(recalibrate_a_vals[0]) > 0:
            avals = list()
            for k, v in recalibrate_a_vals.items():
                avals.append(v)
            avals = np.array(avals)
            recalibrate_avg_dict['a'] = np.mean(avals, axis=1)
            recalibrate_stdev_dict['a'] = np.std(avals, axis=1)
        if len(recalibrate_b_vals[0]) > 0:
            bvals = list()
            for k, v in recalibrate_b_vals.items():
                bvals.append(v)
            bvals = np.array(bvals)
            recalibrate_avg_dict['b'] = np.mean(bvals, axis=1)
            recalibrate_stdev_dict['b'] = np.std(bvals, axis=1)
        if len(recalibrate_c_vals[0]) > 0:
            cvals = list()
            for k, v in recalibrate_c_vals.items():
                cvals.append(v)
            cvals = np.array(cvals)
            recalibrate_avg_dict['c'] = np.mean(cvals, axis=1)
            recalibrate_stdev_dict['c'] = np.std(cvals, axis=1)
        if len(recalibrate_d_vals[0]) > 0:
            dvals = list()
            for k, v in recalibrate_d_vals.items():
                dvals.append(v)
            dvals = np.array(dvals)
            recalibrate_avg_dict['d'] = list(np.mean(dvals, axis=1))
            recalibrate_stdev_dict['d'] = list(np.std(dvals, axis=1))

        return recalibrate_avg_dict, recalibrate_stdev_dict

    def _get_recalibration_params(self, savepath, data_type, file_extension):
        if file_extension == '.xlsx':
            recalibrate_dict = pd.read_excel(os.path.join(savepath, 'recalibration_parameters_'+str(data_type)+'.xlsx'), engine='openpyxl').to_dict('records')
        elif file_extension == '.csv':
            recalibrate_dict = pd.read_csv(os.path.join(savepath, 'recalibration_parameters_' + str(data_type) + '.csv')).to_dict('records')
        return recalibrate_dict

    def help(self):
        print('Documentation for', self.splitter)
        pprint(dict(inspect.getmembers(self.splitter))['__doc__'])
        print('\n')
        print('Class methods for,', self.splitter)
        pprint(dict(inspect.getmembers(self.splitter, predicate=inspect.ismethod)))
        print('\n')
        print('Class attributes for,', self.splitter)
        pprint(self.splitter.__dict__)
        return


class SklearnDataSplitter(BaseSplitter):
    """
    Class to wrap any scikit-learn based data splitter, e.g. KFold

    Args:
        splitter (str): string denoting the name of a sklearn.model_selection object, e.g. 'KFold' will draw from sklearn.model_selection.KFold()

        kwargs : key word arguments for the sklearn.model_selection object, e.g. n_splits=5 for KFold()

    Attributes:
        parallel_run: an attribute definining wheteher to run splits with all available computer cores

    Methods:
        get_n_splits: method to calculate the number of splits to perform
            Args:
                None

            Returns:
                (int), number of train/test splits

        split: method to perform split into train indices and test indices
            Args:
                X: (numpy array), array of X features

            Returns:
                (numpy array), array of train and test indices

        _setup_savedir: method to create a savedir based on the provided model, splitter, selector names and datetime
            Args:
                model: (mastml.models.SklearnModel or other estimator object), an estimator, e.g. KernelRidge

                selector: (mastml.feature_selectors or other selector object), a selector, e.g. EnsembleModelFeatureSelector

                savepath: (str), string designating the savepath

            Returns:
                splitdir: (str), string containing the new subdirectory to save results to
    """

    def __init__(self, splitter, **kwargs):

        # Compensate for parallel mode
        self.parallel_run = False
        if 'parallel_run' in kwargs.keys():
            self.parallel_run = kwargs['parallel_run']
            del kwargs['parallel_run']  # Remove key to not break self.splitter

        super(SklearnDataSplitter, self).__init__()
        self.splitter = getattr(sklearn.model_selection, splitter)(**kwargs)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.splitter.get_n_splits(X, y, groups)

    def split(self, X, y=None, groups=None):
        return self.splitter.split(X, y, groups)

    def _setup_savedir(self, model, selector, preprocessor, savepath):
        now = datetime.now()
        try:
            model_name = model.model.__class__.__name__
        except:
            model_name = model.__class__.__name__

        if model_name == 'BaggingRegressor':
            dirname = model_name + '_' + model.base_estimator_ + '_' + self.splitter.__class__.__name__ + '_' +preprocessor.__class__.__name__+'_'+ selector.__class__.__name__
        else:
            dirname = model_name +'_'+self.splitter.__class__.__name__+'_'+preprocessor.__class__.__name__+'_'+selector.__class__.__name__
        dirname = f"{dirname}_{now.year:02d}_{now.month:02d}_{now.day:02d}" \
            f"_{now.hour:02d}_{now.minute:02d}_{now.second:02d}"
        if savepath == None:
            splitdir = os.getcwd()
        else:
            splitdir = os.path.join(savepath, dirname)
        if not os.path.exists(splitdir):
            os.mkdir(splitdir)
        return splitdir


class NoSplit(BaseSplitter):
    """
    Class to just train the model on the training data and test it on that same data. Sometimes referred to as a "Full fit"
    or a "Single fit", equivalent to just plotting y vs. x.

    Args:
        None (only object instance)

    Attributes:
        parallel_run: an attribute definining wheteher to run splits with all available computer cores

    Methods:
        get_n_splits: method to calculate the number of splits to perform
            Args:
                None

            Returns:
                (int), always 1 as only a single split is performed

        split: method to perform split into train indices and test indices
            Args:
                X: (numpy array), array of X features

            Returns:
                (numpy array), array of train and test indices (all data used as train and test for NoSplit)

    """

    def __init__(self, **kwargs):
        self.parallel_run = False  # No splits so no parallel
        super(NoSplit, self).__init__()

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        indices = np.arange(X.shape[0])
        return [[indices, indices]]


class JustEachGroup(BaseSplitter):
    """
    Class to train the model on one group at a time and test it on the rest of the data
    This class wraps scikit-learn's LeavePGroupsOut with P set to n-1. More information is available at:
    http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePGroupsOut.html

    Args:
        None (only object instance)

    Attributes:
        parallel_run: an attribute definining wheteher to run splits with all available computer cores

    Methods:
        get_n_splits: method to calculate the number of splits to perform
            Args:
                groups: (numpy array), array of group labels

            Returns:
                (int), number of unique groups, indicating number of splits to perform

        split: method to perform split into train indices and test indices
            Args:
                X: (numpy array), array of X features

                y: (numpy array), array of y data

                groups: (numpy array), array of group labels

            Returns:
                (numpy array), array of train and test indices

    """

    def __init__(self, **kwargs):
        # Compensate for parallel mode
        self.parallel_run = False
        if 'parallel_run' in kwargs.keys():
            self.parallel_run = kwargs['parallel_run']
            del kwargs['parallel_run']  # Remove key to not break self.splitter
        super(JustEachGroup, self).__init__()

    def get_n_splits(self, X=None, y=None, groups=None):
        return np.unique(groups).shape[0]

    def split(self, X, y, groups):
        n_groups = self.get_n_splits(groups=groups)
        lpgo = ms.LeavePGroupsOut(n_groups=n_groups-1)
        trains_tests = list()
        for train_index, test_index in lpgo.split(X, y, groups):
            trains_tests.append((train_index, test_index))
        return trains_tests


class LeaveCloseCompositionsOut(BaseSplitter):
    """
    Leave-P-out where you exclude materials with compositions close to those the test set

    Computes the distance between the element fraction vectors. For example, the :math:`L_2`
    distance between Al and Cu is :math:`\sqrt{2}` and the :math:`L_1` distance between Al
    and Al0.9Cu0.1 is 0.2.

    Consequently, this splitter requires a list of compositions as the input to `split` rather
    than the features.

    Attributes:
        parallel_run: an attribute definining wheteher to run splits with all available computer cores

    Args:
        composition_df (pd.DataFrame): dataframe containing the vector of material compositions to analyze

        dist_threshold (float): Entries must be farther than this distance to be included in the training set

        nn_kwargs (dict): Keyword arguments for the scikit-learn NearestNeighbor class used to find nearest points

    """

    def __init__(self, composition_df, dist_threshold=0.1, nn_kwargs=None, **kwargs):

        # Compensate for parallel mode
        self.parallel_run = False
        if 'parallel_run' in kwargs.keys():
            self.parallel_run = kwargs['parallel_run']
            del kwargs['parallel_run']  # Remove key to not break self.splitter

        super(LeaveCloseCompositionsOut, self).__init__()
        if nn_kwargs is None:
            nn_kwargs = {}
        self.composition_df = composition_df
        self.dist_threshold = dist_threshold
        self.nn_kwargs = nn_kwargs

    def split(self, X, y=None, groups=None):

        # Generate the composition vectors
        frac_computer = ElementFraction()
        elem_fracs = frac_computer.featurize_many(list(map(Composition, self.composition_df[self.composition_df.columns[0]])), pbar=False)

        # Generate the nearest-neighbor lookup tool
        neigh = NearestNeighbors(**self.nn_kwargs)
        neigh.fit(elem_fracs)

        # Generate a list of all entries
        all_inds = np.arange(0, self.composition_df.shape[0], 1)

        # Loop through each entry in X
        trains_tests = list()
        for i, x in enumerate(elem_fracs):

            # Get all the entries within the threshold distance of the test point
            too_close, = neigh.radius_neighbors([x], self.dist_threshold, return_distance=False)

            # Get the training set as "not these points"
            train_inds = np.setdiff1d(all_inds, too_close)
            test_inds = np.setdiff1d(all_inds, train_inds)

            trains_tests.append((np.asarray(train_inds), np.asarray(test_inds)))
        return trains_tests

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(X)


class LeaveOutPercent(BaseSplitter):
    """
    Class to train the model using a certain percentage of data as training data

    Args:
        percent_leave_out (float): fraction of data to use in training (must be > 0 and < 1)

        n_repeats (int): number of repeated splits to perform (must be >= 1)

    Attributes:
        parallel_run: an attribute definining wheteher to run splits with all available computer cores

    Methods:
        get_n_splits: method to return the number of splits to perform
            Args:
                groups: (numpy array), array of group labels

            Returns:
                (int), number of unique groups, indicating number of splits to perform

        split: method to perform split into train indices and test indices
            Args:
                X: (numpy array), array of X features

                y: (numpy array), array of y data

                groups: (numpy array), array of group labels

            Returns:
                (numpy array), array of train and test indices

    """

    def __init__(self, percent_leave_out=0.2, n_repeats=5, **kwargs):

        # Compensate for parallel mode
        self.parallel_run = False
        if 'parallel_run' in kwargs.keys():
            self.parallel_run = kwargs['parallel_run']
            del kwargs['parallel_run']  # Remove key to not break self.splitter

        super(LeaveOutPercent, self).__init__()
        self.percent_leave_out = percent_leave_out
        self.n_repeats = n_repeats

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_repeats

    def split(self, X, y=None, groups=None):
        indices = range(X.shape[0])
        split = list()
        for i in range(self.n_repeats):
            trains, tests = ms.train_test_split(indices, test_size=self.percent_leave_out, random_state=np.random.randint(1, 1000), shuffle=True)
            split.append((trains, tests))
        return split


class LeaveOutTwinCV(BaseSplitter):
    """
    Class to remove data twins from the test data.

    Args:
        threshold: (int), the threshold at which two data points are considered twins. Default 0.

        ord: (int), The order of the norm of the difference (see scipy.spatial.distance.minkowski). Default 2 (Euclidean Distance).

        auto_threshold: (boolean), true if threshold should be automatically increased until twins corresponding to the ceiling parameter are found. Default False.
        ceiling: (float), fraction of total data to find as twins. Default 0.

    Attributes:
        parallel_run: an attribute definining wheteher to run splits with all available computer cores

    Methods:
        get_n_splits: method to calculate the number of splits to perform across all splitters
            Args:
                X: (numpy array), array of X features

                y: (numpy array), array of y data

                groups: (numpy array), array of group labels

            Returns:
                (int), the number 1 always

        split: method to perform split into train indices and test indices
            Args:
                X: (numpy array), array of X features

                y: (numpy array), array of y data

                groups: (numpy array), array of group labels

            Returns:
                (numpy array), array of train and test indices
    """

    def __init__(self, threshold=0, ord=2, debug=False, auto_threshold=False, ceiling=0, **kwargs):

        # Compensate for parallel mode
        self.parallel_run = False
        if 'parallel_run' in kwargs.keys():
            self.parallel_run = kwargs['parallel_run']
            del kwargs['parallel_run']  # Remove key to not break self.splitter

        super(LeaveOutTwinCV, self).__init__()
        params = locals()
        self.threshold = threshold
        self.splitter = self.__class__.__name__
        self.debug = debug
        self.ord = ord
        self.auto_threshold = auto_threshold
        self.ceiling = ceiling
        self.splitdir = None
        if self.debug:
            for k, v in params.items():
                print(f"{k}\t\t{v}")

    def get_n_splits(self, X=None, y=None, groups=None):
        # self.splitdir (if it exists at this point) must be None else the internal call will create a excel file
        try:
            save_dir = self.splitdir
            self.splitdir = None
        except AttributeError:
            pass
        # internal call
        actual_splits = self.split(X, y, groups)
        try:
            self.splitdir = save_dir
        except AttributeError:
            pass
        return len(actual_splits)

    def split(self, X, y, X_noinput=None, groups=None):
        X = np.array(X)
        y = np.array(y)
        origIdx = set(np.arange(X.shape[0]))
        twinIdx = set()

        l = len(X)
        n = max(int(self.ceiling * l), 2)

        autothreshold_num_twins = []

        threshold = self.threshold

        # compute all twins
        while (len(twinIdx) < n):
            for i, a in enumerate(X):
                for j, b in enumerate(X):
                    if (i != j and j > i):
                        # calculate distance
                        if (minkowski(a, b, self.ord) <= threshold):
                            if i not in twinIdx:
                                twinIdx.add(i)
                            if j not in twinIdx:
                                twinIdx.add(j)
            # update threshold if needed
            if (not self.auto_threshold):
                break
            autothreshold_num_twins.append([threshold, len(twinIdx)])
            if threshold <= 0:
                threshold = 0.1
            threshold *= 1.1
            if self.debug:
                print(threshold)

        if self.debug:
            print("Thresholds / Number of Twins")
            for th, num in autothreshold_num_twins:
                print(f"{th}\t{num}")

        if (self.auto_threshold):
            print("AutoThreshold was enabled for LeaveOutTwinCV.")
        #     print("Thresholds / Number of Twins")
        #     for th, num in autothreshold_num_twins:
        #         print(f"{th}\t{num}")

        if self.splitdir != None:
            autothreshold_num_twins = pd.DataFrame(data=autothreshold_num_twins, columns=["Threshold", "n_twins"])
            filename = "autothreshold_num_twins"
            autothreshold_num_twins.to_excel(os.path.join(self.splitdir, filename)+'.xlsx', index=False)

        # remove twins from original indices
        twinIdx = list(twinIdx)
        for t in twinIdx:
            if t in origIdx:
                origIdx.remove(t)
        origIdx = list(origIdx)

        if self.debug:
            print("Non-Twins / Twins")
            print(origIdx)
            print(twinIdx)

        if not origIdx:
            print("Warning: All data was marked as twins. Consider reducing threshold or enabling autothreshold. If you are using autothreshold make sure ceiling < 1.")
        if not twinIdx:
            print("Warning: No data twins were found, returning train/test split as empty. Consider increasing threshold or setting autothreshold to True.")

        splits = []
        splits.append([origIdx, twinIdx])
        splits.append([twinIdx, origIdx])
        return splits


class LeaveOutClusterCV(BaseSplitter):
    """
    Class to generate train/test split using clustering.

    Args:
        cluster: clustering method from sklearn.cluster used to generate train/test split

        kwargs: takes in any other key argument for optional cluster parameters

    Attributes:
        parallel_run: an attribute definining wheteher to run splits with all available computer cores

    Methods:
        get_n_splits: method to calculate the number of splits to perform across all splitters
            Args:
                X: (numpy array), array of X features

                y: (numpy array), array of y data

                groups: (numpy array), array of group labels

            Returns:
                (int), number of splits

        split: method to perform split into train indices and test indices
            Args:
                X: (numpy array), array of X features

                y: (numpy array), array of y data

                groups: (numpy array), array of group labels

            Returns:
                (numpy array), array of train and test indices

        labels: method that returns cluster labels of X features
            Args:
                    X: (numpy array), array of X features

                    y: (numpy array), array of y data

                    groups: (numpy array), array of group labels

                Returns:
                    (numpy array), array of cluster labels
    """

    def __init__(self, cluster, **kwargs):

        # Compensate for parallel mode
        self.parallel_run = False
        if 'parallel_run' in kwargs.keys():
            self.parallel_run = kwargs['parallel_run']
            del kwargs['parallel_run']  # Remove key to not break self.splitter

        super(LeaveOutClusterCV, self).__init__()

        # generate cluster object of given input
        try:
            self.cluster = getattr(sklearn.cluster, cluster)(**kwargs)
        except AttributeError:
            self.cluster = getattr(sklearn_extra.cluster, cluster)(**kwargs)

    # gets number of splits or clusters
    def get_n_splits(self, X, y=None, groups=None):

        return len(np.unique(self.labels(X)))

    # splits data into train and test based on clusters
    def split(self, X, y=None, groups=None):

        # trains cluster object
        fit_cluster = self.cluster.fit(X)

        # checks if cluster object has either labels_ or row_labels_
        if hasattr(fit_cluster, 'labels_'):
            labels = fit_cluster.labels_

        elif hasattr(fit_cluster, 'row_labels_'):
            labels = fit_cluster.row_labels_

        # set up split list to return
        trains_tests = list()

        # iterate over unique labels
        for i in np.unique(labels):
            # add X indices not in cluster to train
            train = np.where(labels != i)

            # add X indices in cluster to test
            test = np.where(labels == i)

            # append set of train_test to split
            trains_tests.append([train, test])

        return trains_tests

    # returns cluster labels
    def labels(self, X, y=None, groups=None):

        # trains cluster
        fit_cluster = self.cluster.fit(X)

        # checks if cluster object has either labels_ or row_labels_
        if hasattr(fit_cluster, 'labels_'):
            labels = fit_cluster.labels_

        elif hasattr(fit_cluster, 'row_labels_'):
            labels = fit_cluster.row_labels_

        # return labels
        return labels


class LeaveMultiGroupOut(BaseSplitter):
    """
    Class to train the model on multiple groups at a time and test it on the rest of the data

    Args:
        multigroup_size: (int), size of the groups to be made for leave out.

    Attributes:
        parallel_run: an attribute definining wheteher to run splits with all available computer cores

    Methods:
        get_n_splits: method to calculate the number of splits to perform
            Args:
                groups: (numpy array), array of group labels

            Returns:
                (int), number of unique groups, indicating number of splits to perform

        split: method to perform split into train indices and test indices
            Args:
                X: (numpy array), array of X features

                y: (numpy array), array of y data

                groups: (numpy array), array of group labels

            Returns:
                (numpy array), array of train and test indices

    """

    def __init__(self, multigroup_size=2, **kwargs):
        super(LeaveMultiGroupOut, self).__init__()
        self.multigroup_size = multigroup_size
        # Compensate for parallel mode
        self.parallel_run = False
        if 'parallel_run' in kwargs.keys():
            self.parallel_run = kwargs['parallel_run']
            del kwargs['parallel_run']  # Remove key to not break self.splitter

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(list(itertools.combinations(groups, self.multigroup_size)))

    def split(self, X, y, groups):
        unique_groups = np.unique(groups)
        super_groups = list(itertools.combinations(unique_groups, self.multigroup_size))
        inds = np.arange(0, X.shape[0])

        # Need to get the indices of the data corresponding to each regular group
        group_ind_dict = dict()
        for group in unique_groups:
            group_inds = np.where(groups == group)[0]
            group_ind_dict[group] = group_inds

        # Build the train/test index sets by looping over the super groups and finding indices of each normal group
        trains_tests = list()
        for super_group in super_groups:
            group_inds = list()
            for k, v in group_ind_dict.items():
                if k in super_group:
                    group_inds.append(v)
            group_inds = np.concatenate(group_inds)
            tests = group_inds
            trains = [i for i in inds if i not in tests]
            trains = np.array(trains)
            trains_tests.append((trains, tests))

        return trains_tests


class Bootstrap(BaseSplitter):
    """
    # Note: Bootstrap taken directly from sklearn Github (https://github.com/scikit-learn/scikit-learn/blob/0.11.X/sklearn/cross_validation.py)
    # which was necessary as it was later removed from more recent sklearn releases
    Random sampling with replacement cross-validation iterator
    Provides train/test indices to split data in train test sets
    while resampling the input n_bootstraps times: each time a new
    random split of the data is performed and then samples are drawn
    (with replacement) on each side of the split to build the training
    and test sets.
    Note: contrary to other cross-validation strategies, bootstrapping
    will allow some samples to occur several times in each splits. However
    a sample that occurs in the train split will never occur in the test
    split and vice-versa.
    If you want each sample to occur at most once you should probably
    use ShuffleSplit cross validation instead.

    Args:
        n: (int), total number of elements in the dataset

        n_bootstraps: (int), (default is 3) Number of bootstrapping iterations

        train_size: (int or float), (default is 0.5) If int, number of samples to include in the training split
            (should be smaller than the total number of samples passed in the dataset).
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split.

        test_size: (int or float or None), (default is None)
            If int, number of samples to include in the training set
            (should be smaller than the total number of samples passed
            in the dataset).
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the test split.
            If None, n_test is set as the complement of n_train.

        random_state: (int or RandomState), Pseudo number generator state used for random sampling.

    Attributes:
        parallel_run: an attribute definining wheteher to run splits with all available computer cores

    """

    # Static marker to be able to introspect the CV type
    indices = True

    def __init__(self, n, n_bootstraps=3, train_size=.5, test_size=None,
                 n_train=None, n_test=None, random_state=0, **kwargs):

        # Compensate for parallel mode
        self.parallel_run = False
        if 'parallel_run' in kwargs.keys():
            self.parallel_run = kwargs['parallel_run']
            del kwargs['parallel_run']  # Remove key to not break self.splitter

        super(Bootstrap, self).__init__()
        self.n = n
        self.n_bootstraps = n_bootstraps
        if n_train is not None:
            train_size = n_train
            warnings.warn(
                "n_train is deprecated in 0.11 and scheduled for "
                "removal in 0.12, use train_size instead",
                DeprecationWarning, stacklevel=2)
        if n_test is not None:
            test_size = n_test
            warnings.warn(
                "n_test is deprecated in 0.11 and scheduled for "
                "removal in 0.12, use test_size instead",
                DeprecationWarning, stacklevel=2)
        if (isinstance(train_size, float) and train_size >= 0.0 and train_size <= 1.0):
            self.train_size = ceil(train_size * n)
        elif isinstance(train_size, int):
            self.train_size = train_size
        else:
            raise ValueError("Invalid value for train_size: %r" %
                             train_size)
        if self.train_size > n:
            raise ValueError("train_size=%d should not be larger than n=%d" %
                             (self.train_size, n))

        if (isinstance(test_size, float) and test_size >= 0.0 and test_size <= 1.0):
            self.test_size = ceil(test_size * n)
        elif isinstance(test_size, int):
            self.test_size = test_size
        elif test_size is None:
            self.test_size = self.n - self.train_size
        else:
            raise ValueError("Invalid value for test_size: %r" % test_size)
        if self.test_size > n:
            raise ValueError("test_size=%d should not be larger than n=%d" %
                             (self.test_size, n))

        self.random_state = random_state

    def __iter__(self):
        rng = check_random_state(self.random_state)
        for i in range(self.n_bootstraps):
            # random partition
            permutation = rng.permutation(self.n)
            ind_train = permutation[:self.train_size]
            ind_test = permutation[self.train_size:self.train_size
                                   + self.test_size]

            # bootstrap in each split individually
            train = rng.randint(0, self.train_size,
                                size=(self.train_size,))
            test = rng.randint(0, self.test_size,
                               size=(self.test_size,))
            yield ind_train[train], ind_test[test]

    def __repr__(self):
        return ('%s(%d, n_bootstraps=%d, train_size=%d, test_size=%d, '
                'random_state=%d)' % (
                    self.__class__.__name__,
                    self.n,
                    self.n_bootstraps,
                    self.train_size,
                    self.test_size,
                    self.random_state,
                ))

    def __len__(self):
        return self.n_bootstraps

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.__len__()

    def split(self, X, y=None, groups=None):
        indices = range(X.shape[0])
        split = list()
        for trains, tests in self:
            split.append((trains.tolist(), tests.tolist()))
        return split


def flatten_split_summary(split_summary):
    # Takes a split_summary dict and flattens it along certain keys to produce a cohesive summary over all splits
    split_summary_flat = dict()
    keys = ['train_ind', 'test_ind', 'domain_gpr', 'domain_elemental', 'trues', 'preds', 'model_errors_test',
            'leaveout_ind', 'trues_leaveout', 'preds_leaveout', 'model_errors_leaveout', 'model_errors_leaveout_cal']

    for k in keys:
        lst = list()
        for split_number, split in split_summary.items():
            if k in list(split.keys()):
                lst.append(np.array(split[k]).ravel())
        if len(lst) > 0:
            split_summary_flat[k] = np.concatenate(lst)

    return split_summary_flat


