"""
Main MAST-ML module responsible for executing the workflow of a MAST-ML run
"""

import argparse
import inspect
import os
import shutil
import logging
import warnings
import re
import json
from datetime import datetime
from collections import OrderedDict
from os.path import join # We use join tons
from functools import reduce
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import make_scorer

from mastml import conf_parser, data_loader, html_helper, plot_helper, utils, learning_curve, data_cleaner, metrics
from mastml.legos import (data_splitters, feature_generators, feature_normalizers,
                    feature_selectors, model_finder, util_legos, randomizers, hyper_opt)
from mastml.legos import clusterers as legos_clusterers
from mastml.legos import model_hosting

log = logging.getLogger('mastml')

def main(conf_path, data_path, outdir=join(os.getcwd(), 'results_mastml_run'), verbosity=0):
    """
    This method is responsible for setting up the initial stage of the MAST-ML run, such as parsing input directories to
    designate where data will be imported and results saved to, as well as creation of the MAST-ML run log.

    Args:

        conf_path: (str), the path supplied by the user which contains the input configuration file

        data_path: (str), the path supplied by the user which contains the input data file (as CSV or XLSX)

        outdir: (str), the path supplied by the user which determines where the output results are saved to

        verbosity: (int), the verbosity level of the MAST-ML log, which determines the amount of information written to the log.

    Returns:

        outdir: (str), the path supplied by the user which determines where the output results are saved to (needed by other calls in MAST-ML)

    """

    conf_path, data_path, outdir = check_paths(conf_path, data_path, outdir)

    utils.activate_logging(outdir, (str(conf_path), str(data_path), outdir), verbosity=verbosity)

    if verbosity >= 1:
        warnings.simplefilter('error') # turn warnings into errors
    elif verbosity <= -1:
        warnings.simplefilter('ignore') # ignore warnings

    try:
        mastml_run(conf_path, data_path, outdir)
    except utils.MastError as e:
        # catch user errors, log and print, but don't raise and show them that nasty stack
        log.error(str(e))
    except Exception as e:
        # catch the error, save it to file, then raise it back up
        log.error('A runtime exception has occured, please go to '
                      'https://github.com/uw-cmg/MAST-ML/issues and post your issue.')
        log.exception(e)
        raise e
    return outdir # so a calling program can know where we actually saved it

def mastml_run(conf_path, data_path, outdir):
    """
    This method is responsible for conducting the main MAST-ML run workflow

    Args:

        conf_path: (str), the path supplied by the user which contains the input configuration file

        data_path: (str), the path supplied by the user which contains the input data file (as CSV or XLSX)

        outdir: (str), the path supplied by the user which determines where the output results are saved to

    Returns:

        None

    """


    " Runs operations specifed in conf_path on data_path and puts results in outdir "

    # Copy the original input files to the output directory for easy reference
    log.info("Copying input files to output directory...")
    if type(conf_path) is str:
        shutil.copy2(conf_path, outdir)
    elif type(conf_path) is dict:
        with open(join(outdir, 'conf_file.conf'), 'w') as f:
            json.dump(conf_path, f)

    if type(data_path) is str:
        shutil.copy2(data_path, outdir)
    elif type(data_path) is type(pd.DataFrame()):
        data_path.to_excel(join(outdir, 'input_data.xlsx'), index=False)
        data_path = join(outdir, 'input_data.xlsx')

    # Load in and parse the configuration and data files:
    if type(conf_path) is str:
        conf = conf_parser.parse_conf_file(conf_path, from_dict=False)
    elif type(conf_path) is dict:
        conf = conf_parser.parse_conf_file(conf_path, from_dict=True)
    else:
        raise TypeError('Your conf_path must either be a path string to a .conf file or a dict directly containing the config info')

    MiscSettings = conf['MiscSettings']
    is_classification = conf['is_classification']
    # The df is used by feature generators, clusterers, and grouping_column to 
    # create more features for x.
    # X is model input, y is target feature for model
    df, X, X_noinput, X_grouped, y = data_loader.load_data(data_path,
                                         conf['GeneralSetup']['input_features'],
                                         conf['GeneralSetup']['input_target'],
                                         conf['GeneralSetup']['input_grouping'],
                                         conf['GeneralSetup']['input_other'])
    if not conf['GeneralSetup']['input_grouping']:
        X_grouped = pd.DataFrame()

    # Perform data cleaning here
    dc = conf['DataCleaning']
    if 'cleaning_method' not in dc.keys():
        log.warning("You have chosen not to specify a method of data_cleaning in the input file. By default, any feature entries "
                    "containing NaN will result in removal of the feature and any target data entries containing NaN will "
                    "result in removal of that target data point.")
        dc['cleaning_method'] = 'remove'

    if X.shape[1] == 0:
        # There are no X feature vectors specified, so can't clean data
        log.warning("There are no X feature vectors imported from the data file. Therefore, data cleaning cannot be performed.")
    else:
        # Always scan the input data and flag potential outliers
        data_cleaner.flag_outliers(df=df, conf_not_input_features=conf['GeneralSetup']['input_other'],
                                   savepath=outdir,
                                   n_stdevs=3)
        if dc['cleaning_method'] == 'remove':
            df, nan_indices = data_cleaner.remove(df, axis=1)
            X, nan_indices = data_cleaner.remove(X, axis=1)
            X_noinput, nan_indices = data_cleaner.remove(X_noinput, axis=1)
            X_grouped, nan_indices = data_cleaner.remove(X_grouped, axis=1)
        elif dc['cleaning_method'] == 'imputation':
            log.warning("You have selected data cleaning with Imputation. Note that imputation will not resolve missing target data. "
                        "It is recommended to remove missing target data")
            if 'imputation_strategy' not in dc.keys():
                log.warning("You have chosen to perform data imputation but have not selected an imputation strategy. By default, "
                            "the mean will be used as the imputation strategy")
                dc['imputation_strategy'] = 'mean'
            df = data_cleaner.imputation(df, dc['imputation_strategy'], X_noinput.columns)
            X = data_cleaner.imputation(X, dc['imputation_strategy'])
        elif dc['cleaning_method'] == 'ppca':
            log.warning("You have selected data cleaning with PPCA. Note that PPCA will not work to estimate missing target values, "
                        "at least a 2D matrix is needed. It is recommended you remove missing target data")
            df = data_cleaner.ppca(df, X_noinput.columns)
            X = data_cleaner.ppca(X)
        else:
            log.error("You have specified an invalid data cleaning method. Choose from: remove, imputation, or ppca")
            exit()

        # Check if any y target data values are missing or NaN
        shape_before = y.shape
        y, nan_indices = data_cleaner.remove(y, axis=0)
        shape_after = y.shape
        if shape_after != shape_before:
            log.info(
                'Warning: some y target data rows were automatically removed because they were either empty or contained '
                '"NaN" entries. MAST-ML will continue your run with this modified data set.')
            # Need to modify df, X, etc. to remove same rows as were removed from y target data
            df = df.drop(labels=nan_indices, axis=0)
            X = X.drop(labels=nan_indices, axis=0)
            X_noinput = X_noinput.drop(labels=nan_indices, axis=0)
            if X_grouped.shape[0] > 0:
                X_grouped = X_grouped.drop(labels=nan_indices, axis=0)

    # randomly shuffles y values if randomizer is on
    if conf['GeneralSetup']['randomizer'] is True:
        log.warning("Randomizer is enabled, so target feature will be shuffled,"
                 " and results should be null for a given model")
        y = randomizers.Randomizer().fit().transform(df=y)

    """
        # get parameters out for 'validation_column'
    is_validation = 'validation_columns' in conf['GeneralSetup']
    if is_validation:
        if type(conf['GeneralSetup']['validation_columns']) is list:
            validation_column_names = list(conf['GeneralSetup']['validation_columns'])
        elif type(conf['GeneralSetup']['validation_columns']) is str:
            validation_column_names = list()
            validation_column_names.append(conf['GeneralSetup']['validation_columns'][:])
        validation_columns = dict()
        for validation_column_name in validation_column_names:
            validation_columns[validation_column_name] = df[validation_column_name]
        validation_columns = pd.DataFrame(validation_columns)
        validation_X = list()
        validation_y = list()

        # TODO make this block its own function
        for validation_column_name in validation_column_names:
            # X_, y_ = _exclude_validation(X, validation_columns[validation_column_name]), _exclude_validation(y, validation_columns[validation_column_name])
            validation_X.append(pd.DataFrame(_exclude_validation(X, validation_columns[validation_column_name])))
            validation_y.append(pd.DataFrame(_exclude_validation(y, validation_columns[validation_column_name])))
        idxy_list = list()
        for i, _ in enumerate(validation_y):
            idxy_list.append(validation_y[i].index)
        # Get intersection of indices between all prediction columns
        intersection = reduce(np.intersect1d, (i for i in idxy_list))
        X_novalidation = X.iloc[intersection]
        y_novalidation = y.iloc[intersection]
        X_grouped_novalidation = X_grouped.iloc[intersection]
    else:
        X_novalidation = X
        y_novalidation = y
        X_grouped_novalidation = X_grouped
    """


    if conf['MiscSettings']['plot_target_histogram']:
        # First, save input data stats to csv
        y.describe().to_csv(join(outdir, 'input_data_statistics.csv'))
        plot_helper.plot_target_histogram(y, join(outdir, 'target_histogram.png'), label=y.name)

    # Get the appropriate collection of metrics:
    metrics_dict = conf['GeneralSetup']['metrics']

    # Extract columns that some splitter need to do grouped splitting using 'grouping_column'
    # special argument
    splitter_to_group_names = _extract_grouping_column_names(conf['DataSplits'])
    log.debug('splitter_to_group_names:\n' + str(splitter_to_group_names))

    # Instantiate models first so we can snatch them and pass them into feature selectors
    models = _instantiate(conf['Models'],
                          model_finder.name_to_constructor,
                          'model')

    models = _snatch_models(models, conf['FeatureSelection'])

    # Instantiate all the sections of the conf file:
    generators  = _instantiate(conf['FeatureGeneration'],
                               feature_generators.name_to_constructor,
                               'featuregenerator')
    clusterers  = _instantiate(conf['Clustering'],
                               legos_clusterers.name_to_constructor,
                               'clusterer')
    normalizers = _instantiate(conf['FeatureNormalization'],
                               feature_normalizers.name_to_constructor,
                               'featurenormalizer')
    splitters   = _instantiate(conf['DataSplits'],
                               data_splitters.name_to_constructor,
                               'datasplit')

    def snatch_model_cv_and_scoring_for_learning_curve(models):
        models = OrderedDict(models)
        if conf['LearningCurve']:
            # Get model
            name = conf['LearningCurve']['estimator']
            conf['LearningCurve']['estimator'] = models[name]
            del models[name]
            # Get cv
            name = conf['LearningCurve']['cv']
            splitter_count = 0
            for splitter in splitters:
                if name in splitter:
                    conf['LearningCurve']['cv'] = splitter[1]
                    break
                else:
                    splitter_count += 1
            del splitters[splitter_count]
        return models

    models = snatch_model_cv_and_scoring_for_learning_curve(models=models)

    models = _snatch_keras_model(models, conf['Models'])
    original_models = models

    # Need to specially snatch the GPR model if it is in models list because it contains special kernel object
    models = _snatch_gpr_model(models, conf['Models'])
    original_models = models

    # Need to snatch models and CV objects for Hyperparam Opt
    hyperopt_params = _snatch_models_cv_for_hyperopt(conf, models, splitters)

    hyperopts = _instantiate(hyperopt_params,
                             hyper_opt.name_to_constructor,
                             'hyperopt')

    hyperopts = OrderedDict(hyperopts)
    hyperopts = list(hyperopts.items())

    # Snatch splitter for use in feature selection, particularly RFECV
    splitters = OrderedDict(splitters)  # for easier modification
    _snatch_splitters(splitters, conf['FeatureSelection'])
    splitters = list(splitters.items())

    selectors   = _instantiate(conf['FeatureSelection'],
                               feature_selectors.name_to_constructor,
                               'featureselector', X_grouped=np.array(X_grouped).reshape(-1, ), X_indices=np.array(X.index.tolist()).reshape(-1, 1))

    log.debug(f'generators: \n{generators}')
    log.debug(f'clusterers: \n{clusterers}')
    log.debug(f'normalizers: \n{normalizers}')
    log.debug(f'hyperopts: \n{hyperopts}')
    log.debug(f'selectors: \n{selectors}')
    log.debug(f'splitters: \n{splitters}')

    # TODO make this block its own function, and change naming from is_validation to is_test here and throughout. Just symantic annoyance.
    # get parameters out for 'validation_column'
    is_validation = 'input_testdata' in conf['GeneralSetup']
    if is_validation:
        if type(conf['GeneralSetup']['input_testdata']) is list:
            validation_column_names = list(conf['GeneralSetup']['input_testdata'])
        elif type(conf['GeneralSetup']['input_testdata']) is str:
            validation_column_names = list()
            validation_column_names.append(conf['GeneralSetup']['input_testdata'][:])
        validation_columns = dict()
        for validation_column_name in validation_column_names:
            validation_columns[validation_column_name] = df[validation_column_name]
        validation_columns = pd.DataFrame(validation_columns)
        validation_X = list()
        validation_y = list()

    def do_all_combos(X, y, df):
        log.info(f"There are {len(normalizers)} feature normalizers, {len(hyperopts)} hyperparameter optimizers, "
                 f"{len(selectors)} feature selectors, {len(models)} models, and {len(splitters)} splitters.")

        def generate_features():
            log.info("Doing feature generation...")
            dataframes = [instance.fit_transform(df, y) for _, instance in generators]
            dataframe = pd.concat(dataframes, 1)
            log.info("Saving generated data to csv...")
            log.debug(f'generated cols: {dataframe.columns}')
            filename = join(outdir, "generated_features.csv")
            pd.concat([dataframe, X_noinput, y], 1).to_csv(filename, index=False)
            return dataframe
        generated_df = generate_features()

        def remove_constants():
            dataframe = _remove_constant_features(generated_df)
            log.info("Saving generated data without constant columns to csv...")
            filename = join(outdir, "generated_features_no_constant_columns.csv")
            pd.concat([dataframe, X_noinput, y], 1).to_csv(filename, index=False)
            return dataframe
        generated_df = remove_constants()

        # add in generated features
        X = pd.concat([X, generated_df], axis=1)
        # add in generated features to full dataframe
        df = pd.concat([df, generated_df], axis=1)
        # Check size of X; if there are no feature columns then throw error
        if X.shape[1] == 0:
            raise utils.InvalidValue('No feature vectors were found in the dataframe. Please either use feature generation methods'
                               'or specify input_features in the input file.')

        # remove repeat columns (keep the first one)
        def remove_repeats(X):
            repeated_columns = X.loc[:, X.columns.duplicated()].columns
            if not repeated_columns.empty:
                log.warning(f"Throwing away {len(repeated_columns)} because they are repeats.")
                log.debug(f"Throwing away columns because they are repeats: {repeated_columns}")
                X = X.loc[:,~X.columns.duplicated()]
            return X
        X = remove_repeats(X)

        # TODO make this block its own function
        # get parameters out for 'validation_column'
        if is_validation:
            for validation_column_name in validation_column_names:
                # X_, y_ = _exclude_validation(X, validation_columns[validation_column_name]), _exclude_validation(y, validation_columns[validation_column_name])
                validation_X.append(pd.DataFrame(_exclude_validation(X, validation_columns[validation_column_name])))
                validation_y.append(pd.DataFrame(_exclude_validation(y, validation_columns[validation_column_name])))
            idxy_list = list()
            for i, _ in enumerate(validation_y):
                idxy_list.append(validation_y[i].index)
            # Get intersection of indices between all prediction columns
            intersection = reduce(np.intersect1d, (i for i in idxy_list))
            X_novalidation = X.iloc[intersection]
            y_novalidation = y.iloc[intersection]
            if conf['GeneralSetup']['input_grouping']:
                X_grouped_novalidation = X_grouped.iloc[intersection]
            else:
                X_grouped_novalidation = pd.DataFrame()
        else:
            X_novalidation = X
            y_novalidation = y
            if conf['GeneralSetup']['input_grouping']:
                X_grouped_novalidation = X_grouped
            else:
                X_grouped_novalidation = pd.DataFrame()

        def make_clustered_df():
            log.info("Doing clustering...")
            clustered_df = pd.DataFrame()
            for name, instance in clusterers:
                clustered_df[name] = instance.fit_predict(X, y)
            return clustered_df
        clustered_df = make_clustered_df() # Each column is a clustering algorithm

        def make_feature_vs_target_plots():
            if clustered_df.empty:
                for column in X: # plot y against each x column
                    filename = f'{column}_vs_target_scatter.png'
                    plot_helper.plot_scatter(X[column], y, join(outdir, filename),
                                             xlabel=column, groups=None, label=y.name)
            else:
                for name in clustered_df.columns: # for each cluster, plot y against each x column
                    for column in X:
                        filename = f'{column}_vs_target_by_{name}_scatter.png'
                        plot_helper.plot_scatter(X[column], y, join(outdir, filename),
                                                clustered_df[name], xlabel=column, label=y.name)
        if MiscSettings['plot_each_feature_vs_target']:
            make_feature_vs_target_plots()

        log.info("Saving clustered data to csv...")
        # Add new cluster info to X df
        if not clustered_df.empty:
            X = pd.concat([X, clustered_df], axis=1)
        pd.concat([X, y], 1).to_csv(join(outdir, "clusters.csv"), index=False)

        def make_normalizer_selector_dataframe_triples(models):
            triples = []
            nonlocal y, y_novalidation
            for normalizer_name, normalizer_instance in normalizers:

                # Run feature normalization
                log.info(f"Running normalizer {normalizer_name} ...")
                normalizer_instance_y = normalizer_instance
                normalizer_instance_y_novalidation = normalizer_instance

                # HERE- try to address issue with normalizing non-validation part of dataset
                normalizer = normalizer_instance.fit(X_novalidation, y)
                X_normalized = normalizer.transform(X)
                X_novalidation_normalized = normalizer.transform(X_novalidation)

                if conf['MiscSettings']['normalize_target_feature'] is True:
                    yreshape = pd.DataFrame(np.array(y).reshape(-1, 1))
                    y_novalidation_new = pd.DataFrame(np.array(y_novalidation).reshape(-1, 1))
                    y_normalized = normalizer_instance_y.fit_transform(yreshape, yreshape)
                    y_novalidation_normalized = normalizer_instance_y_novalidation.fit_transform(y_novalidation_new, y_novalidation_new)
                    y_normalized.columns = [conf['GeneralSetup']['input_target']]
                    y_novalidation_normalized.columns = [conf['GeneralSetup']['input_target']]
                    y = pd.Series(np.squeeze(y_normalized), name=conf['GeneralSetup']['input_target'])
                    y_novalidation = pd.Series(np.squeeze(y_novalidation_normalized), name=conf['GeneralSetup']['input_target'])
                else:
                    normalizer_instance_y = None
                log.info("Saving normalized data to csv...")
                dirname = join(outdir, normalizer_name)
                os.mkdir(dirname)
                pd.concat([X_normalized, X_noinput, y], 1).to_csv(join(dirname, "normalized.csv"), index=False)

                # Save off the normalizer as .pkl for future import
                joblib.dump(normalizer, join(dirname, str(normalizer.__class__.__name__) + ".pkl"))

                # HERE- find data twins


                # Put learning curve here??
                if conf['LearningCurve']:
                    learning_curve_estimator = conf['LearningCurve']['estimator']
                    learning_curve_scoring = conf['LearningCurve']['scoring']
                    n_features_to_select = int(conf['LearningCurve']['n_features_to_select'])
                    learning_curve_cv = conf['LearningCurve']['cv']
                    try:
                        selector_name = conf['LearningCurve']['selector_name']
                    except KeyError:
                        selector_name = None

                    # Get score name from scoring object
                    scoring_name = learning_curve_scoring._score_func.__name__
                    scoring_name_nice = ''
                    for s in scoring_name.split('_'):
                        scoring_name_nice += s + ' '
                    # Do sample learning curve
                    train_sizes, train_mean, test_mean, train_stdev, test_stdev = learning_curve.sample_learning_curve(X=X_novalidation_normalized, y=y_novalidation,
                                                            estimator=learning_curve_estimator, cv=learning_curve_cv,
                                                            scoring=learning_curve_scoring,
                                                            Xgroups=X_grouped_novalidation)
                    plot_helper.plot_learning_curve(train_sizes, train_mean, test_mean, train_stdev, test_stdev,
                                                    scoring_name_nice, 'sample_learning_curve',
                                                    join(dirname, f'data_learning_curve'))
                    # Do feature learning curve
                    train_sizes, train_mean, test_mean, train_stdev, test_stdev = learning_curve.feature_learning_curve(X=X_novalidation_normalized, y=y_novalidation,
                                                            estimator=learning_curve_estimator, cv=learning_curve_cv,
                                                            scoring=learning_curve_scoring, selector_name=selector_name,
                                                            savepath=dirname,
                                                            n_features_to_select=n_features_to_select,
                                                            Xgroups=X_grouped_novalidation)
                    plot_helper.plot_learning_curve(train_sizes, train_mean, test_mean, train_stdev, test_stdev,
                                                    scoring_name_nice, 'feature_learning_curve',
                                                    join(dirname, f'feature_learning_curve'))



                log.info("Running selectors...")

                # Run feature selection
                for selector_name, selector_instance in selectors:
                    log.info(f"    Running selector {selector_name} ...")
                    dirname = join(outdir, normalizer_name, selector_name)
                    os.mkdir(dirname)
                    # NOTE: Changed from .fit_transform to .fit.transform
                    # because PCA.fit_transform doesn't call PCA.transform
                    if selector_instance.__class__.__name__ == 'MASTMLFeatureSelector':
                        dirname = join(outdir, normalizer_name)
                        X_selected = selector_instance.fit(X_novalidation_normalized, y_novalidation, dirname, X_grouped_novalidation).transform(X_novalidation_normalized)
                    elif selector_instance.__class__.__name__ == 'SequentialFeatureSelector':
                        X_selected = selector_instance.fit(X_novalidation_normalized, y_novalidation).transform(X_novalidation_normalized)
                        # Need to reset indices in case have test data, otherwise df.equals won't properly find column names
                        X_novalidation_normalized_reset = X_novalidation_normalized.reset_index()
                        # SFS renames the columns. Need to replace the column names with correct feature names.
                        feature_name_dict = dict()
                        for feature in X_selected.columns.tolist():
                            for realfeature in X_novalidation_normalized.columns.tolist():
                                if X_novalidation_normalized_reset[realfeature].equals(X_selected[feature]):
                                    feature_name_dict[feature] = realfeature
                        X_selected.rename(columns= feature_name_dict, inplace=True)
                    elif selector_instance.__class__.__name__ == 'PearsonSelector':
                        X_selected = selector_instance.fit(X=X_novalidation_normalized, savepath=dirname, y=y_novalidation).transform(X_novalidation_normalized)
                    else:
                        X_selected = selector_instance.fit(X_novalidation_normalized, y_novalidation).transform(X_novalidation_normalized)
                    features_selected = X_selected.columns.tolist()
                    # Need to do this instead of taking X_selected directly because otherwise won't concatenate correctly with test data values, which are
                    # left out of the feature selection process.
                    X_selected = X_normalized[features_selected]
                    log.info("    Saving selected features to csv...")

                    pd.concat([X_selected, X_noinput, y], 1).to_csv(join(dirname, "selected.csv"), index=False)
                    #TODO: fix this naming convention
                    triples.append((normalizer_name, normalizer_instance_y, selector_name, X_selected))

                    # Run Hyperparam optimization, update model list with optimized model(s)
                    for hyperopt_name, hyperopt_instance in hyperopts:
                        try:
                            log.info(f"    Running hyperopt {hyperopt_name} ...")
                            log.info(f"    Saving optimized hyperparams and data to csv...")
                            dirname = join(outdir, normalizer_name, selector_name, hyperopt_name)
                            os.mkdir(dirname)
                            estimator_name = hyperopt_instance._estimator_name
                            best_estimator = hyperopt_instance.fit(X_selected, y, savepath=os.path.join(dirname, str(estimator_name)+'.csv'))

                            new_name = estimator_name + '_' + str(normalizer_name) + '_' + str(selector_name) + '_' + str(hyperopt_name)
                            new_model = best_estimator
                            models[new_name] = new_model

                            # Update models list with new hyperparams
                            #for model in models:
                            #    # model[0] is name, model[1] is instance
                            #    # Check that this particular model long_name had its hyperparams optimized
                            #    for name in hyperopt_params.keys():
                            #        if model[0] in name[:]:
                            #            model[0] = model[0]+'_nonoptimized'
                            #            #model[1] = best_estimator
                            #            # Need to update model as new model name and instance to handle multiple
                            #            # selector/optimization/fitting path
                            #            new_name = model[0]+'_'+str(normalizer_name)+'_'+str(selector_name)+'_'+str(hyperopt_name)
                            #            new_model = best_estimator
                            #            models[new_name] = new_model

                        except:
                            raise utils.InvalidValue

            return triples
        normalizer_selector_dataframe_triples = make_normalizer_selector_dataframe_triples(models=models)

        ## DataSplits (cross-product)
        ## Collect grouping columns, splitter_to_groupmes is a dict of splitter name to grouping col
        log.debug("Finding splitter-required columns in data...")
        def make_splittername_splitlist_pairs():
            # exclude the testing_only rows from use in splits
            if is_validation:
                validation_X = list()
                validation_y = list()
                for validation_column_name in validation_column_names:
                    #X_, y_ = _exclude_validation(X, validation_columns[validation_column_name]), _exclude_validation(y, validation_columns[validation_column_name])
                    validation_X.append(pd.DataFrame(_exclude_validation(X, validation_columns[validation_column_name])))
                    validation_y.append(pd.DataFrame(_exclude_validation(y, validation_columns[validation_column_name])))
                idxy_list = list()
                for i, _ in enumerate(validation_y):
                    idxy_list.append(validation_y[i].index)
                # Get intersection of indices between all prediction columns
                intersection = reduce(np.intersect1d, (i for i in idxy_list))
                X_ = X.iloc[intersection]
                y_ = y.iloc[intersection]
            else:
                X_, y_ = X, y

            pairs = []

            def fix_index(array):
                return X_.index.values[array] 

            def proper_index(splits):
                """ For example, if X's indexs are [1,4,6] and you split 
                [ [[0],[1,2]], [[1],[0,2]] ] then we would get 
                [ [[1],[4,6]], [[4],[1,6]] ] 
                Needed only for valdation row stuff.
                """
                return tuple(tuple(fix_index(part) for part in split) for split in splits)


            # Collect all the grouping columns, `None` if not needed
            splitter_to_group_column = dict()
            splitter_to_group_column_no_validation = dict()
            for name, instance in splitters:
                # if this splitter depends on grouping
                if name in splitter_to_group_names: 
                    col = splitter_to_group_names[name]
                    log.debug(f"    Finding {col} for {name}...")
                    # Locate the grouping column among all dataframes
                    for df_ in [clustered_df, df, X_]:
                        if col in df_.columns:
                            # FOund it!
                            # Get groups for plotting first
                            splitter_to_group_column[name] = df_[col].values
                            if is_validation:
                                _df_list = list()
                                if df_ is not clustered_df:
                                    # exclude for df_ so that rows match up in splitter
                                    for validation_column_name in validation_column_names:
                                        df_ = _exclude_validation(df_, validation_columns[validation_column_name])
                                        _df_list.append(df_)
                                elif df_ is clustered_df:
                                    # merge the cluster data df_ to full df
                                    df[col] = df_
                                    for validation_column_name in validation_column_names:
                                        df_ = _exclude_validation(df, validation_columns[validation_column_name])
                                        _df_list.append(df_)

                                # Get df_ based on index intersection between all df's in _df_list
                                idxy_list = list()
                                for i, _ in enumerate(_df_list):
                                    idxy_list.append(_df_list[i].index)
                                # Get intersection of indices between all prediction columns
                                intersection = reduce(np.intersect1d, (i for i in idxy_list))
                                df_ = df.iloc[intersection]

                            # and use the no-validation one for the split
                            grouping_data = df_[col].values
                            split = proper_index(instance.split(X_, y_, grouping_data))
                            pairs.append((name, split))
                            break
                    # If we didn't find that column anywhere, raise
                    else:
                        raise utils.MissingColumnError(f'DataSplit {name} needs column {col}, which '
                                                       f'was neither generated nor given by input')

                # If we don't need grouping column
                else: 
                    splitter_to_group_column[name] = None
                    split = proper_index(instance.split(X_, y_))
                    pairs.append((name, split))

            return pairs, splitter_to_group_column
        splittername_splitlist_pairs, splitter_to_group_column = make_splittername_splitlist_pairs()

        log.info("Fitting models to splits...")

        def do_models_splits(models, original_models):
            models = list(models.items())
            original_models = list(original_models.items())
            original_model_names = [model[0] for model in original_models]
            all_results = []
            for normalizer_name, normalizer_instance, selector_name, X in normalizer_selector_dataframe_triples:
                subdir = join(outdir, normalizer_name, selector_name)

                if MiscSettings['plot_each_feature_vs_target']:
                    #if selector_name == 'DoNothing': continue
                    # for each selector/normalizer, plot y against each x column
                    for column in X:
                        filename = f'{column}_vs_target.png'
                        plot_helper.plot_scatter(X[column], y, join(subdir, filename),
                                                 xlabel=column, label=y.name)
                for model_name, model_instance in models:
                    #Here, add logic to only run original models and respective models from hyperparam opt
                    do_split = False
                    if model_name in original_model_names:
                        do_split = True
                    elif (normalizer_name in model_name) and (selector_name in model_name):
                        do_split = True
                    if do_split == True:
                        for splitter_name, trains_tests in splittername_splitlist_pairs:
                            grouping_data = splitter_to_group_column[splitter_name]
                            subdir = join(normalizer_name, selector_name, model_name, splitter_name)
                            log.info(f"    Running splits for {subdir}")
                            subsubdir = join(outdir, subdir)
                            os.makedirs(subsubdir)
                            # NOTE: do_one_splitter is a big old function, does lots
                            runs = do_one_splitter(X, y, model_instance, subsubdir, trains_tests, grouping_data, normalizer_instance)
                            all_results.extend(runs)
            return all_results

        return do_models_splits(models, original_models)

    def do_one_splitter(X, y, model, main_path, trains_tests, grouping_data, normalizer_instance):

        def one_fit(split_num, train_indices, test_indices, normalizer_instance):

            log.info(f"        Doing split number {split_num}")
            train_X, train_y = X.loc[train_indices], y.loc[train_indices]
            test_X,  test_y  = X.loc[test_indices],  y.loc[test_indices]

            # split up groups into train and test as well
            if grouping_data is not None:
                train_groups, test_groups = grouping_data[train_indices], grouping_data[test_indices]
            else:
                train_groups, test_groups = None, None

            path = join(main_path, f"split_{split_num}")
            os.mkdir(path)

            log.info("             Fitting model and making predictions...")
            # Catch the ValueError associated with not being able to convert string to float
            #try:
                #print(train_X, train_y)


                # For Keras model, save model summary to main_path and plot training/validation vals vs. epochs
            if 'KerasRegressor' in str(model.__class__.__name__):
                with open(join(main_path, 'keras_model_summary.txt'), 'w') as f:
                    with redirect_stdout(f):
                        model.summary()
                history = model.fit(train_X, train_y)
                plot_helper.plot_keras_history(model_history=history,
                                                   savepath=join(path,'keras_model_accuracy.png'),
                                                   plot_type='accuracy')
                plot_helper.plot_keras_history(model_history=history,
                                                   savepath=join(path, 'keras_model_loss.png'),
                                                   plot_type='loss')
                pd.DataFrame().from_dict(data=history.history).to_excel(join(path,'keras_model_data.xlsx'))
            else:
                model.fit(train_X, train_y)

            #except ValueError:
            #    raise utils.InvalidValue('MAST-ML has detected an error with one of your feature vectors which has caused an error'
            #                       ' in model fitting.')
            # Save off the trained model as .pkl for future import

            # TODO: note that saving keras models has broken with updated keras version
            if 'KerasRegressor' not in model.__class__.__name__:
                joblib.dump(model, os.path.abspath(join(path, str(model.__class__.__name__)+"_split_"+str(split_num)+".pkl")))

            if is_classification:
                # For classification, need probabilty of prediction to make accurate ROC curve (and other predictions??).
                #TODO:Consider using only predict_proba and not predict() method for classif problems. Have exit escape if probability set to False here.
                # See stackoverflow post:
                #https: // stats.stackexchange.com / questions / 329857 / what - is -the - difference - between - decision
                # - function - predict - proba - and -predict - fun

                #params = model.get_params()
                #if params['probability'] == True:
                try:
                    train_pred_proba = model.predict_proba(train_X)
                    test_pred_proba = model.predict_proba(test_X)
                except:
                    log.error('You need to perform classification with model param probability=True enabled for accurate'
                                ' predictions, if your model has the probability param (e.g. RandomForestClassifier does not. '
                              'Please reset this parameter as applicable and re-run MASTML')
                    exit()
                train_pred = model.predict(train_X)
                test_pred = model.predict(test_X)
            else:
                train_pred = model.predict(train_X)
                test_pred  = model.predict(test_X)
                if train_pred.ndim > 1:
                    train_pred = np.squeeze(train_pred)
                if test_pred.ndim > 1:
                    test_pred = np.squeeze(test_pred)
                if conf['MiscSettings']['normalize_target_feature'] is True:
                    train_pred = normalizer_instance.inverse_transform(train_pred)
                    test_pred = normalizer_instance.inverse_transform(test_pred)
                    train_y = pd.Series(normalizer_instance.inverse_transform(train_y))
                    test_y = pd.Series(normalizer_instance.inverse_transform(test_y))

                # Here- for Random Forest, Extra Trees, and Gradient Boosters output feature importances
                if model.__class__.__name__ in ['RandomForestRegressor', 'ExtraTreesRegressor', 'GradientBoostingRegressor']:
                    pd.concat([pd.DataFrame(X.columns), pd.DataFrame(model.feature_importances_)],  1).to_excel(join(path, str(model.__class__.__name__)+'_featureimportances.xlsx'), index=False)

            # here is where we need to collect validation stats
            if is_validation:
                validation_predictions_list = list()
                validation_y_forpred_list = list()
                for validation_column_name in validation_column_names:
                    validation_X_forpred = _only_validation(X, validation_columns[validation_column_name])
                    validation_y_forpred = _only_validation(y, validation_columns[validation_column_name])
                    log.info("             Making predictions on prediction_only data...")
                    validation_predictions = model.predict(validation_X_forpred)
                    validation_predictions_list.append(validation_predictions)
                    validation_y_forpred_list.append(validation_y_forpred)

                    # save them as 'predicitons.csv'
                    validation_predictions = np.squeeze(validation_predictions)
                    validation_predictions_series = pd.Series(validation_predictions, name='clean_predictions', index=validation_X_forpred.index)
                    #validation_noinput_series = pd.Series(X_noinput.index, index=validation_X.index)
                    pd.concat([validation_X_forpred,  validation_y_forpred,  validation_predictions_series],  1)\
                            .to_csv(join(path, 'predictions_'+str(validation_column_name)+'.csv'), index=False)
            else:
                validation_y = None
            

            # Save train and test data and results to csv:
            log.info("             Saving train/test data and predictions to csv...")
            train_pred_series = pd.DataFrame(train_pred, columns=['train_pred'], index=train_indices)
            train_noinput_series = pd.DataFrame(X_noinput, index=train_indices)
            pd.concat([train_X, train_y, train_pred_series, train_noinput_series], 1)\
                    .to_csv(join(path, 'train.csv'), index=False)
            test_pred_series = pd.DataFrame(test_pred,   columns=['test_pred'],  index=test_indices)
            test_noinput_series = pd.DataFrame(X_noinput, index=test_indices)
            pd.concat([test_X,  test_y,  test_pred_series, test_noinput_series],  1)\
                    .to_csv(join(path, 'test.csv'),  index=False)


            log.info("             Calculating score metrics...")
            split_path = main_path.split(os.sep)

            # collect metrics inside a warning catching block for some things we know we should ignore
            with warnings.catch_warnings():
                # NOTE I tried making this more specific use warnings's regex filter but it would never
                # catch it for some indeterminiable reason.
                # This warning is raised when you ask for Recall on something from y_true that never
                # occors in y_pred. sklearn assumes 0.0, and we want it to do so (silently).
                warnings.simplefilter('ignore', UndefinedMetricWarning)
                train_metrics = OrderedDict((name, function(train_y, train_pred))
                                            for name, (_, function) in metrics_dict.items())
                test_metrics = OrderedDict((name, function(test_y, test_pred))
                                           for name, (_, function) in metrics_dict.items())
                # Need to pass y_train data to get rmse/sigma for test rmse and sigma of train y
                if 'rmse_over_stdev' in metrics_dict.keys():
                    test_metrics['rmse_over_stdev'] = metrics_dict['rmse_over_stdev'][1](test_y, test_pred, train_y)
                if 'R2_adjusted' in metrics_dict.keys():
                    test_metrics['R2_adjusted'] = metrics_dict['R2_adjusted'][1](test_y, test_pred, test_X.shape[1])
                    train_metrics['R2_adjusted'] = metrics_dict['R2_adjusted'][1](train_y, train_pred, train_X.shape[1])

                split_result = OrderedDict(
                    normalizer=split_path[-4],
                    selector=split_path[-3],
                    model=split_path[-2],
                    splitter=split_path[-1],
                    split_num=split_num,
                    y_train_true=train_y.values,
                    y_train_pred=train_pred,
                    y_test_true=test_y.values,
                    y_test_pred=test_pred,
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    train_groups=train_groups,
                    test_groups=test_groups,
                )

                if is_validation:
                    prediction_metrics_list = list()
                    for validation_column_name, validation_y, validation_predictions in zip(validation_column_names, validation_y_forpred_list, validation_predictions_list):
                        prediction_metrics = OrderedDict((name, function(validation_y, validation_predictions))
                                           for name, (_, function) in metrics_dict.items())
                        if 'rmse_over_stdev' in prediction_metrics.keys():
                            # Correct series passed?
                            prediction_metrics['rmse_over_stdev'] = metrics_dict['rmse_over_stdev'][1](validation_y, validation_predictions, train_y)
                        prediction_metrics_list.append(prediction_metrics)
                        split_result['y_validation_true'+'_'+str(validation_column_name)] = validation_y.values
                        split_result['y_validation_pred'+'_'+str(validation_column_name)] = validation_predictions
                    split_result['prediction_metrics'] = prediction_metrics_list
                else:
                    split_result['prediction_metrics'] = None

            if is_classification:
                split_result['y_train_pred_proba'] = train_pred_proba
                split_result['y_test_pred_proba'] = test_pred_proba

            log.info("             Making plots...")
            if MiscSettings['plot_train_test_plots']:
                plot_helper.make_train_test_plots(
                        split_result, path, is_classification, 
                        label=y.name, model=model, train_X=train_X, test_X=test_X, groups=grouping_data)

            if MiscSettings['plot_error_plots']:
                if is_validation:
                    plot_helper.make_error_plots(split_result, path, is_classification,
                                                 label=y.name, model=model, train_X=train_X, test_X=test_X,
                                                 rf_error_method=MiscSettings['rf_error_method'],
                                                 rf_error_percentile=MiscSettings['rf_error_percentile'],
                                                 is_validation = is_validation,
                                                 validation_column_name = validation_column_name,
                                                 validation_X = validation_X_forpred,
                                                 groups=grouping_data)
                else:
                    plot_helper.make_error_plots(split_result, path, is_classification,
                                                 label=y.name, model=model, train_X=train_X, test_X=test_X,
                                                 rf_error_method=MiscSettings['rf_error_method'],
                                                 rf_error_percentile=MiscSettings['rf_error_percentile'],
                                                 is_validation = is_validation,
                                                 validation_column_name = None,
                                                 validation_X= None,
                                                 groups=grouping_data)
            # Write stats in each split path, not main path
            if is_validation:
                _write_stats_tocsv(split_result['train_metrics'],
                         split_result['test_metrics'],
                         path,
                         split_result['prediction_metrics'],
                         validation_column_names)
                _write_stats(split_result['train_metrics'],
                         split_result['test_metrics'],
                         path,
                         split_result['prediction_metrics'],
                         validation_column_names)
            else:
                _write_stats_tocsv(split_result['train_metrics'],
                             split_result['test_metrics'],
                             path)
                _write_stats(split_result['train_metrics'],
                             split_result['test_metrics'],
                             path)

            return split_result

        split_results = []
        for split_num, (train_indices, test_indices) in enumerate(trains_tests):
            split_results.append(one_fit(split_num, train_indices, test_indices, normalizer_instance))

        log.info("    Calculating mean and stdev of scores...")
        def make_train_test_average_and_std_stats():
            train_stats = OrderedDict([('Average Train', None)])
            test_stats  = OrderedDict([('Average Test', None)])
            if is_validation:
                prediction_stats = list()
                num_predictions = len(split_results[0]['prediction_metrics'])
                for i in range(num_predictions):
                    prediction_stats.append(OrderedDict([('Average Prediction', None)]))
            for name in metrics_dict:
                train_values = [split_result['train_metrics'][name] for split_result in split_results]
                test_values  = [split_result['test_metrics'][name]  for split_result in split_results]
                train_stats[name] = (np.mean(train_values), np.std(train_values))
                test_stats[name]  = (np.mean(test_values), np.std(test_values))
                if is_validation:
                    for i in range(num_predictions):
                        prediction_values = [split_result['prediction_metrics'][i][name] for split_result in split_results]
                        prediction_stats[i][name] = (np.mean(prediction_values), np.std(prediction_values))
                test_stats_single = dict()
                test_stats_single[name] = (np.mean(test_values), np.std(test_values))
                if grouping_data is not None:
                    groups = np.array(split_results[0]['test_groups'].tolist()+split_results[0]['train_groups'].tolist())
                    unique_groups = np.union1d(split_results[0]['test_groups'], split_results[0]['train_groups'])
                    plot_helper.plot_metric_vs_group(metric=name, groups=unique_groups, stats=test_values,
                                                     avg_stats = test_stats_single, savepath=join(main_path, str(name)+'_vs_group.png'))
                    plot_helper.plot_metric_vs_group_size(metric=name, groups=groups, stats=test_values,
                                                     avg_stats = test_stats_single, savepath=join(main_path, str(name)+'_vs_group_size.png'))
            del train_stats['Average Train']
            del test_stats['Average Test']
            if is_validation:
                for i in range(num_predictions):
                    del prediction_stats[i]['Average Prediction']
                return train_stats, test_stats, prediction_stats
            else:
                return train_stats, test_stats

        if is_validation:
            avg_train_stats, avg_test_stats, avg_prediction_stats = make_train_test_average_and_std_stats()
            # Here- write average stats to main folder of splitter
            _write_stats_tocsv(avg_train_stats,
                         avg_test_stats,
                         main_path, prediction_metrics=avg_prediction_stats, prediction_names=validation_column_names)
            _write_stats(avg_train_stats,
                         avg_test_stats,
                         main_path, prediction_metrics=avg_prediction_stats, prediction_names=validation_column_names)
        else:
            avg_train_stats, avg_test_stats = make_train_test_average_and_std_stats()
            # Here- write average stats to main folder of splitter
            _write_stats_tocsv(avg_train_stats,
                         avg_test_stats,
                         main_path)
            _write_stats(avg_train_stats,
                         avg_test_stats,
                         main_path)

        def make_average_error_plots(main_path):
            has_model_errors = False
            has_model_errors_validation = False
            dfs_cumulative_errors = list()
            dfs_cumulative_errors_validation = list()
            for split_folder, _, __ in os.walk(main_path):
                if "split" in split_folder:
                    path = join(main_path, split_folder)
                    try:
                        dfs_cumulative_errors.append(pd.read_csv(join(path,'test_cumulative_normalized_error.csv')))
                        if is_validation:
                            dfs_cumulative_errors_validation.append(pd.read_csv(join(path, 'validation_cumulative_normalized_error.csv')))
                    except:
                        pass

            # Concatenate all dfs in list to one big df
            df_cumulative_errors = pd.concat(dfs_cumulative_errors)
            if is_validation:
                df_cumulative_errors_validation = pd.concat(dfs_cumulative_errors_validation)
            # Need to get average values of df columns by averagin over groups of Y True values (since each Y True should
            # only appear once)
            # TODO: change this to get values explicitly from each split and then average, as some Y True values may have same value and appear multiple times
            df_normalized_errors_avgvalues = df_cumulative_errors.groupby('Y True').mean().reset_index()
            y_true = np.array(df_normalized_errors_avgvalues['Y True'])
            y_pred = np.array(df_normalized_errors_avgvalues['Y Pred'])
            if is_validation:
                df_normalized_errors_avgvalues_validation = df_cumulative_errors_validation.groupby('Y True').mean().reset_index()
                y_true_validation = np.array(df_normalized_errors_avgvalues_validation['Y True'])
                y_pred_validation = np.array(df_normalized_errors_avgvalues_validation['Y Pred'])
            try:
                average_error_values = np.array(df_normalized_errors_avgvalues['error_bars_down'])
                has_model_errors = True
            except:
                average_error_values = None
                has_model_errors = False

            if is_validation:
                try:
                    average_error_values_validation = np.array(df_normalized_errors_avgvalues_validation['error_bars_down'])
                    has_model_errors_validation = True
                except:
                    average_error_values_validation = None
                    has_model_errors_validation = False

            plot_helper.plot_average_cumulative_normalized_error(y_true=y_true, y_pred=y_pred,
                                                                 savepath=join(main_path,'test_cumulative_normalized_error_average_allsplits.png'),
                                                                 has_model_errors=has_model_errors,
                                                                 err_avg=average_error_values)
            if is_validation:
                plot_helper.plot_average_cumulative_normalized_error(y_true=y_true_validation, y_pred=y_pred_validation,
                                                                     savepath=join(main_path,
                                                                                   'validation_cumulative_normalized_error_average_allsplits.png'),
                                                                     has_model_errors=has_model_errors_validation,
                                                                     err_avg=average_error_values_validation)
            plot_helper.plot_average_normalized_error(y_true=y_true, y_pred=y_pred,
                                                      savepath=join(main_path,'test_normalized_error_average_allsplits.png'),
                                                                 has_model_errors=has_model_errors,
                                                                 err_avg=average_error_values)
            return

        # Call to make average error plots
        if conf['MiscSettings']['plot_error_plots']:
            log.info("    Making average error plots over all splits")
            make_average_error_plots(main_path=main_path)

        log.info("    Making best/worst plots...")
        def get_best_worst_median_runs():
            # sort splits by the test score of first metric:
            greater_is_better, _ = next(iter(metrics_dict.values())) # get first value pair
            scalar = 1 if greater_is_better else -1
            s = sorted(split_results, key=lambda run: scalar*next(iter(run['test_metrics'])))
            return s[0], s[len(split_results)//2], s[-1]
        worst, median, best = get_best_worst_median_runs()

        def make_pred_vs_true_plots(model, y):
            if conf['MiscSettings']['normalize_target_feature'] == True:
                y = pd.Series(normalizer_instance.inverse_transform(y), name=conf['GeneralSetup']['input_target'])

            if MiscSettings['plot_predicted_vs_true']:
                plot_helper.plot_best_worst_split(y.values, best, worst,
                                                  join(main_path, 'best_worst_split'), label=conf['GeneralSetup']['input_target'])
            predictions = [[] for _ in range(X.shape[0])]
            for split_num, (train_indices, test_indices) in enumerate(trains_tests):
                for i, pred in zip(test_indices, split_results[split_num]['y_test_pred']):
                    predictions[i].append(pred)
            if MiscSettings['plot_predicted_vs_true_average']:
                plot_helper.plot_predicted_vs_true_bars(
                        y.values, predictions, avg_test_stats,
                        join(main_path, 'predicted_vs_true_average'), label=conf['GeneralSetup']['input_target'])
                if grouping_data is not None:
                    plot_helper.plot_predicted_vs_true_bars(
                        y.values, predictions, avg_test_stats,
                        join(main_path, 'predicted_vs_true_average_groupslabeled'),
                        label=conf['GeneralSetup']['input_target'],
                        groups=grouping_data)
            if MiscSettings['plot_best_worst_per_point']:
                plot_helper.plot_best_worst_per_point(y.values, predictions,
                                                      join(main_path, 'best_worst_per_point'),
                                                      metrics_dict, avg_test_stats, label=conf['GeneralSetup']['input_target'])

        if not is_classification:
            make_pred_vs_true_plots(model=model, y=y)

        return split_results

    runs = do_all_combos(X, y, df) # calls do_one_splitter internally

    log.info("Making image html file...")
    html_helper.make_html(outdir)

    log.info("Making html file of all runs stats...")
    _save_all_runs(runs, outdir)

    # Here- do DLHub model hosting if have section
    if bool(conf['ModelHosting']) != False: # dict is empty
        model_hosting.host_model(model_path=conf['ModelHosting']['model_path'],
                                 preprocessor_path=conf['ModelHosting']['preprocessor_path'],
                                 training_data_path=conf['ModelHosting']['training_data_path'],
                                 model_title=conf['ModelHosting']['model_title'],
                                 model_name=conf['ModelHosting']['model_name'],
                                 model_type="scikit-learn")
        log.info('Finished uploading model to DLHub...')

    log.info('Your MAST-ML run has finished successfully!')
    return

def _instantiate(kwargs_dict, name_to_constructor, category, X_grouped=None, X_indices=None):
    """
    Uses name_to_constructor to instantiate every item in kwargs_dict and return
    the list of instantiations
    """
    instantiations = []
    for long_name, (name, kwargs) in kwargs_dict.items():
        log.debug(f'instantiation: {long_name}, {name}({kwargs})')
        try:
            #skip instantiate step for keras model because need to pass dict to build model and not all values directly
            if 'KerasRegressor' in long_name:
                pass

            # Need to construct cv object when have special case of RFECV and LeaveOneGroupOut cross-validation!
            elif name == 'RFECV':
                if 'cv' in kwargs.keys():
                    if X_grouped is not None:
                        if kwargs['cv'].__class__.__name__ == 'LeaveOneGroupOut':
                            trains = list()
                            tests = list()
                            for train_idx, test_idx in LeaveOneGroupOut().split(X=X_indices, y=None, groups=X_grouped):
                                trains.append(train_idx)
                                tests.append(test_idx)
                            custom_cv = zip(trains, tests)
                            kwargs['cv'] = custom_cv
                instantiations.append([long_name, name_to_constructor[name](**kwargs)])
            else:
                instantiations.append([long_name, name_to_constructor[name](**kwargs)])

        except TypeError:
            log.info(f"ARGUMENTS FOR '{name}': {inspect.signature(name_to_constructor[name])}")
            raise utils.InvalidConfParameters(
                f"The {category} '{name}' has invalid parameters: {kwargs}\n"
                f"Signature for '{name}': {inspect.signature(name_to_constructor[name])}")
        except KeyError:
            raise utils.InvalidConfSubSection(
                f"There is no {category} called '{name}'."
                f"All valid {category}: {list(name_to_constructor.keys())}")

    return instantiations

def _grouping_column_to_group_number(X_grouped):
    group_list = X_grouped.values.reshape((1, -1))
    unique_groups = np.unique(group_list).tolist()
    group_dict = dict()
    group_list_asnumber = list()
    for i, group in enumerate(unique_groups):
        group_dict[group] = i+1
    for i, group in enumerate(group_list.tolist()[0]):
        group_list_asnumber.append(group_dict[group])
    X_grouped_asnumber = np.asarray(group_list_asnumber)
    return X_grouped_asnumber

def _snatch_models(models, conf_feature_selection):
    models = OrderedDict(models)
    log.debug(f'models, pre-snatching: \n{models}')
    for selector_name, [_, args_dict] in conf_feature_selection.items():
        if 'estimator' in args_dict:
            model_name = args_dict['estimator']
            try:
                args_dict['estimator'] = models[model_name]
                del models[model_name]
            except KeyError:
                raise utils.MastError(f"The selector {selector_name} specified model {model_name},"
                                      f"which was not found in the [Models] section")
    log.debug(f'models, post-snatching: \n{models}')
    return models

def _snatch_keras_model(models, conf_models):
    for model in conf_models.keys():
        if 'KerasRegressor' in model:
            keras_model = model_finder.KerasRegressor(conf_models[model][1])
            models[model] = keras_model
    return models

def _snatch_gpr_model(models, conf_models):
    for model in models.keys():
        if 'GaussianProcessRegressor' in model:
            import sklearn.gaussian_process
            from sklearn.gaussian_process import GaussianProcessRegressor
            kernel_list = ['WhiteKernel', 'RBF', 'ConstantKernel', 'Matern', 'RationalQuadratic', 'ExpSineSquared', 'DotProduct']
            kernel_operators = ['+', '*', '-']
            params = conf_models[model]
            kernel_string = params[1]['kernel']
            # Need to delete old kernel (as str) from params so can use other specified params in new GPR model
            del params[1]['kernel']
            # Parse kernel_string to identify kernel types and any kernel operations to combine kernels
            kernel_types_asstr = list()
            kernel_types_ascls = list()
            kernel_operators_used = list()

            for s in kernel_string[:]:
                if s in kernel_operators:
                    kernel_operators_used.append(s)

            # Do case for single kernel, no operators
            if len(kernel_operators_used) == 0:
                kernel_types_asstr.append(kernel_string)
            else:
                # New method, using re
                unique_operators = np.unique(kernel_operators_used).tolist()
                unique_operators_asstr = '['
                for i in unique_operators:
                    unique_operators_asstr += str(i)
                unique_operators_asstr += ']'
                kernel_types_asstr = re.split(unique_operators_asstr, kernel_string)

            for kernel in kernel_types_asstr:
                kernel_ = getattr(sklearn.gaussian_process.kernels, kernel)
                kernel_types_ascls.append(kernel_())

            # Case for single kernel
            if len(kernel_types_ascls) == 1:
                kernel = kernel_types_ascls[0]

            kernel_count = 0
            for i, operator in enumerate(kernel_operators_used):
                if i+1 <= len(kernel_operators_used):
                    if operator == "+":
                        if kernel_count == 0:
                            kernel = kernel_types_ascls[kernel_count] + kernel_types_ascls[kernel_count+1]
                        else:
                            kernel += kernel_types_ascls[kernel_count+1]
                    elif operator == "*":
                        if kernel_count == 0:
                            kernel = kernel_types_ascls[kernel_count] * kernel_types_ascls[kernel_count+1]
                        else:
                            kernel *= kernel_types_ascls[kernel_count+1]
                    else:
                        logging.warning('You have chosen an invalid operator to construct a composite kernel. Please choose'
                                      ' either "+" or "*".')
                    kernel_count += 1

            gpr = GaussianProcessRegressor(kernel=kernel, **params[1])
            # Need to delete old GPR from model list and replace with new GPR with correct kernel and other params.
            del models[model]
            models[model] = gpr
            break
    return models

def _snatch_models_cv_for_hyperopt(conf, models, splitters):
    models = list(models.items())
    if conf['HyperOpt']:
        for searchtype, searchparams in conf['HyperOpt'].items():
            for paramtype, paramvalue in searchparams[1].items():
                if paramtype == 'estimator':
                    # Need to grab model and params from Model section of conf file
                    found_model = False
                    for model in models:
                        if model[0] == paramvalue:
                            conf['HyperOpt'][searchtype][1]['estimator'] = model[1]
                            found_model = True
                            break
                    if found_model == False:
                        raise utils.MastError(f"The estimator {paramvalue} could not be found in the input file!")
                if paramtype == 'cv':
                    # Need to grab cv and params from DataSplits section of conf file
                    found_cv = False
                    for splitter in splitters:
                        if splitter[0] == paramvalue:
                            conf['HyperOpt'][searchtype][1]['cv'] = splitter[1]
                            found_cv = True
                            break
                    if found_cv == False:
                        raise utils.MastError(f"The cv object {paramvalue} could not be found in the input file!")
                if paramtype == 'scoring':
                    # Need to grab correct scoring object
                    found_scorer = False
                    metrics_dict = metrics.regression_metrics
                    if paramvalue in metrics_dict.keys():
                        conf['HyperOpt'][searchtype][1]['scoring'] = make_scorer(metrics_dict[paramvalue][1],
                                                                                 greater_is_better=metrics_dict[paramvalue][0])
                        found_scorer = True
                        break
                    if found_scorer == False:
                        raise utils.MastError(
                            f"The scoring object {paramvalue} could not be found in the input file!")

    return conf['HyperOpt']

def _snatch_splitters(splitters, conf_feature_selection):
    log.debug(f'cv, pre-snatching: \n{splitters}')
    for selector_name, (_, args_dict) in conf_feature_selection.items():
        # Here: add snatch to cv object for feature selection with RFECV
        if 'cv' in args_dict:
            cv_name = args_dict['cv']
            try:
                args_dict['cv'] = splitters[cv_name]
                del splitters[cv_name]
            except KeyError:
                raise utils.MastError(f"The selector {selector_name} specified cv splitter {cv_name},"
                                      f"which was not found in the [DataSplits] section")
    log.debug(f'cv, post-snatching: \n{splitters}')

def _extract_grouping_column_names(splitter_to_kwargs):
    splitter_to_group_names = dict()
    for splitter_name, name_and_kwargs in splitter_to_kwargs.items():
        _, kwargs = name_and_kwargs
        if 'grouping_column' in kwargs:
            column_name = kwargs['grouping_column']
            del kwargs['grouping_column'] # because the splitter doesn't actually take this
            splitter_to_group_names[splitter_name] = column_name
    return splitter_to_group_names

def _remove_constant_features(df):
    log.info("Removing constant features, regardless of feature selectors.")
    before = set(df.columns)
    df = df.loc[:, (df != df.iloc[0]).any()]
    removed = list(before - set(df.columns))
    if removed != []:
        log.warning(f'Removed {len(removed)}/{len(before)} constant columns.')
        log.debug("Removed the following constant columns: " + str(removed))
    return df

def _save_all_runs(runs, outdir):
    """
    Produces a giant html table of all stats for all runs
    """
    table = []
    for run in runs:
        od = OrderedDict()
        for name, value in run.items():
            if name == 'train_metrics':
                for k, v in run['train_metrics'].items():
                    od['train_'+k] = v
            elif name == 'test_metrics':
                for k, v in run['test_metrics'].items():
                    od['test_'+k] = v
            else:
                od[name] = value
        table.append(od)
    pd.DataFrame(table).to_html(join(outdir, 'all_runs_table.html'))

def _write_stats(train_metrics, test_metrics, outdir, prediction_metrics=None, prediction_names=None):
    with open(join(outdir, 'stats_summary.txt'), 'w') as f:
        f.write("TRAIN:\n")
        for name,score in train_metrics.items():
            if type(score) == tuple:
                f.write(f"{name}: {'%.3f'%float(score[0])} +/- {'%.3f'%float(score[1])}\n")
            else:
                f.write(f"{name}: {'%.3f'%float(score)}\n")
        f.write("TEST:\n")
        for name,score in test_metrics.items():
            if type(score) == tuple:
                f.write(f"{name}: {'%.3f'%float(score[0])} +/- {'%.3f'%float(score[1])}\n")
            else:
                f.write(f"{name}: {'%.3f'%float(score)}\n")
        if prediction_metrics:
            #prediction metrics now list of dicts for predicting multiple values
            for prediction_metric, prediction_name in zip(prediction_metrics, prediction_names):
                f.write("PREDICTION for "+str(prediction_name)+":\n")
                for name, score in prediction_metric.items():
                    if type(score) == tuple:
                        f.write(f"{name}: {'%.3f'%float(score[0])} +/- {'%.3f'%float(score[1])}\n")
                    else:
                        f.write(f"{name}: {'%.3f'%float(score)}\n")

def _write_stats_tocsv(train_metrics, test_metrics, outdir, prediction_metrics=None, prediction_names=None):
    datadict = dict()
    for name, score in train_metrics.items():
        if type(score) == tuple:
            datadict[name+' train score'] = float(score[0])
            datadict[name+' train stdev'] = float(score[1])
        else:
            datadict[name+' train score'] = float(score)
    for name, score in test_metrics.items():
        if type(score) == tuple:
            datadict[name+' validation score'] = float(score[0])
            datadict[name+' validation stdev'] = float(score[1])
        else:
            datadict[name+' validation score'] = float(score)
    if prediction_names:
        for prediction_metric, prediction_name in zip(prediction_metrics, prediction_names):
            for name, score in prediction_metric.items():
                if type(score) == tuple:
                    datadict[prediction_name+' '+name+' score'] = float(score[0])
                    datadict[prediction_name+' '+name+' stdev'] = float(score[1])
                else:
                    datadict[prediction_name+' '+name+' score'] = float(score)
    pd.DataFrame().from_dict(data=datadict, orient='index').to_csv(join(outdir, 'stats_summary.csv'))
    return

def _exclude_validation(df, validation_column):
    return df.loc[validation_column != 1]

def _only_validation(df, validation_column):
    return df.loc[validation_column == 1]

def check_paths(conf_path, data_path, outdir):
    """
    This method is responsible for error handling of the user-specified paths for the configuration file, data file,
    and output directory.

    Args:

        conf_path: (str), the path supplied by the user which contains the input configuration file

        data_path: (str), the path supplied by the user which contains the input data file (as CSV or XLSX)

        outdir: (str), the path supplied by the user which determines where the output results are saved to

    Returns:

        conf_path: (str), the path supplied by the user which contains the input configuration file

        data_path: (str), the path supplied by the user which contains the input data file (as CSV or XLSX)

        outdir: (str), the path supplied by the user which determines where the output results are saved to

    """


    # Check conf path:
    if type(conf_path) is str:
        if os.path.splitext(conf_path)[1] != '.conf':
            raise utils.FiletypeError(f"Conf file does not end in .conf: '{conf_path}'")
        if not os.path.isfile(conf_path):
            raise utils.FileNotFoundError(f"No such file: {conf_path}")
    elif type(conf_path) is dict:
        pass
    else:
        raise TypeError('Your conf_path must be either a string to .conf file path or a dict')

    # Check data path:
    if type(data_path) is str:
        if os.path.splitext(data_path)[1] not in ['.csv', '.xlsx']:
            raise utils.FiletypeError(f"Data file does not end in .csv or .xlsx: '{data_path}'")
        if not os.path.isfile(data_path):
            raise utils.FileNotFoundError(f"No such file: {data_path}")
    elif type(data_path) is type(pd.DataFrame()):
        pass
    else:
        raise TypeError('Your data_path must be either a string to .csv or .xlsx data file or a pd.DataFrame object')

    # Check output directory:

    if os.path.exists(outdir):
        try:
            os.rmdir(outdir) # succeeds if empty
        except OSError: # directory not empty
            log.warning(f"{outdir} not empty. Renaming...")
            now = datetime.now()
            outdir = outdir.rstrip(os.sep) # remove trailing slash
            outdir = f"{outdir}_{now.month:02d}_{now.day:02d}" \
                     f"_{now.hour:02d}_{now.minute:02d}_{now.second:02d}"
    os.makedirs(outdir)
    log.info(f"Saving to directory '{outdir}'")

    return conf_path, data_path, outdir

def get_commandline_args():
    """
    This method is responsible for parsing and checking the command-line execution of MAST-ML inputted by the user.

    Args:

        None

    Returns:

        (str), the path supplied by the user which contains the input configuration file

        (str), the path supplied by the user which contains the input data file (as CSV or XLSX)

        (str), the path supplied by the user which determines where the output results are saved to

        verbosity: (int), the verbosity level of the MAST-ML log, which determines the amount of information writtent to the log.

    """

    parser = argparse.ArgumentParser(description='MAterials Science Toolkit - Machine Learning')
    parser.add_argument('conf_path', type=str, help='path to mastml .conf file')
    parser.add_argument('data_path', type=str, help='path to csv or xlsx file')
    parser.add_argument('-o', action="store", dest='outdir', default='results',
                        help='Folder path to save output files to. Defaults to results/')
    # from https://stackoverflow.com/a/14763540
    # we only use them to set a bool but it would be nice to have multiple levels in the future
    parser.add_argument('-v', '--verbosity', action="count",
                        help="include this flag for more verbose output")
    parser.add_argument('-q', '--quietness', action="count",
                       help="include this flag to hide [DEBUG] printouts, or twice to hide [INFO]")

    args = parser.parse_args()
    verbosity = (args.verbosity if args.verbosity else 0)\
            - (args.quietness if args.quietness else 0)
    # verbosity -= 1 ## uncomment this for distribution
    return (os.path.abspath(args.conf_path),
            os.path.abspath(args.data_path),
            os.path.abspath(args.outdir),
            verbosity)

if __name__ == '__main__':
    conf_path, data_path, outdir, verbosity = get_commandline_args()
    main(conf_path, data_path, outdir, verbosity)
