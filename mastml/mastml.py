"""
Module for getting a mastml system call and calling all the appropriate subroutines
"""

import argparse
import inspect
import os
import shutil
import logging
import warnings
from datetime import datetime
from collections import OrderedDict
from os.path import join # We use join tons
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import LeaveOneGroupOut

from . import conf_parser, data_loader, html_helper, plot_helper, utils, learning_curve, data_cleaner
from .legos import (data_splitters, feature_generators, feature_normalizers,
                    feature_selectors, model_finder, util_legos)
from .legos import clusterers as legos_clusterers

log = logging.getLogger('mastml')

def main(conf_path, data_path, outdir, verbosity=0):
    " Sets up logger and error catching, then starts the run "
    conf_path, data_path, outdir = check_paths(conf_path, data_path, outdir)

    utils.activate_logging(outdir, (conf_path, data_path, outdir), verbosity=verbosity)

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
    " Runs operations specifed in conf_path on data_path and puts results in outdir "

    # Copy the original input files to the output directory for easy reference
    log.info("Copying input files to output directory...")
    shutil.copy2(conf_path, outdir)
    shutil.copy2(data_path, outdir)

    # Load in and parse the configuration and data files:
    conf = conf_parser.parse_conf_file(conf_path)
    PlotSettings = conf['PlotSettings']
    is_classification = conf['is_classification']
    # The df is used by feature generators, clusterers, and grouping_column to 
    # create more features for x.
    # X is model input, y is target feature for model
    df, X, X_noinput, X_grouped, y = data_loader.load_data(data_path,
                                     conf['GeneralSetup']['input_features'],
                                     conf['GeneralSetup']['target_feature'],
                                     conf['GeneralSetup']['grouping_feature'],
                                     conf['GeneralSetup']['not_input_features'])

    # Perform data cleaning here
    try:
        dc = conf['GeneralSetup']['data_cleaning']
    except KeyError:
        log.warning("You have chosen not to specify a method of data_cleaning in the input file. By default, any feature entries "
                    "containing NaN will result in removal of the feature and any target data entries containing NaN will "
                    "result in removal of that target data point.")
        dc = 'remove'
    if dc == 'remove':
        df = data_cleaner.remove(df, axis=1)
        X = data_cleaner.remove(X, axis=1)
        X_noinput = data_cleaner.remove(X_noinput, axis=1)
        X_grouped = data_cleaner.remove(X_grouped, axis=1)
        y = data_cleaner.remove(y, axis=0)

    # randomly shuffly y values if randomizer is on
    if conf['GeneralSetup']['randomizer'] is True:
        log.warning("Randomizer is enabled, so target feature will be shuffled,"
                 " and results should be null for a given model")
        y = y.sample(frac=1).reset_index(drop=True)

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

    if conf['PlotSettings']['target_histogram']:
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
    models = OrderedDict(models) # for easier modification

    _snatch_models(models, conf['FeatureSelection'])

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

    def snatch_model_cv_and_scoring_for_learning_curve():
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

    snatch_model_cv_and_scoring_for_learning_curve()

    models = list(models.items())

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
    log.debug(f'selectors: \n{selectors}')
    log.debug(f'splitters: \n{splitters}')

    def do_all_combos(X, y, df):
        log.info(f"There are {len(normalizers)} feature normalizers, {len(selectors)} feature"
                 f"selectors {len(models)} models, and {len(splitters)} splitters.")

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

        # remove repeat columns (keep the first one)
        def remove_repeats(X):
            repeated_columns = X.loc[:, X.columns.duplicated()].columns
            if not repeated_columns.empty:
                log.warning(f"Throwing away {len(repeated_columns)} because they are repeats.")
                log.debug(f"Throwing away columns because they are repeats: {repeated_columns}")
                X = X.loc[:,~X.columns.duplicated()]
            return X
        X = remove_repeats(X)

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
                                             xlabel=column, groups=None, ylabel='target_feature', label=y.name)
            else:
                for name in clustered_df.columns: # for each cluster, plot y against each x column
                    for column in X:
                        filename = f'{column}_vs_target_by_{name}_scatter.png'
                        plot_helper.plot_scatter(X[column], y, join(outdir, filename),
                                                clustered_df[name], xlabel=column,
                                                ylabel='target_feature', label=y.name)
        if PlotSettings['feature_vs_target']:
            make_feature_vs_target_plots()

        log.info("Saving clustered data to csv...")
        # Add new cluster info to X df
        if not clustered_df.empty:
            X = pd.concat([X, clustered_df], axis=1)
        pd.concat([X, y], 1).to_csv(join(outdir, "clusters.csv"), index=False)

        def make_normalizer_selector_dataframe_triples():
            triples = []
            for normalizer_name, normalizer_instance in normalizers:
                log.info(f"Running normalizer {normalizer_name} ...")
                X_normalized = normalizer_instance.fit_transform(X, y)
                log.info("Saving normalized data to csv...")
                dirname = join(outdir, normalizer_name)
                os.mkdir(dirname)
                pd.concat([X_normalized, X_noinput, y], 1).to_csv(join(dirname, "normalized.csv"), index=False)

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
                    train_sizes, train_mean, test_mean, train_stdev, test_stdev = learning_curve.sample_learning_curve(X=X_novalidation, y=y_novalidation,
                                                            estimator=learning_curve_estimator, cv=learning_curve_cv,
                                                            scoring=learning_curve_scoring, Xgroups=X_grouped_novalidation)
                    plot_helper.plot_learning_curve(train_sizes, train_mean, test_mean, train_stdev, test_stdev,
                                                    scoring_name_nice, 'sample_learning_curve',
                                                    join(dirname, f'data_learning_curve.png'))
                    # Do feature learning curve
                    train_sizes, train_mean, test_mean, train_stdev, test_stdev = learning_curve.feature_learning_curve(X=X_novalidation, y=y_novalidation,
                                                            estimator=learning_curve_estimator, cv=learning_curve_cv,
                                                            scoring=learning_curve_scoring, selector_name=selector_name,
                                                            n_features_to_select=n_features_to_select,
                                                            Xgroups=X_grouped_novalidation)
                    plot_helper.plot_learning_curve(train_sizes, train_mean, test_mean, train_stdev, test_stdev,
                                                    scoring_name_nice, 'feature_learning_curve',
                                                    join(dirname, f'feature_learning_curve.png'))



                log.info("Running selectors...")
                for selector_name, selector_instance in selectors:
                    log.info(f"    Running selector {selector_name} ...")
                    # NOTE: Changed from .fit_transform to .fit.transform
                    # because PCA.fit_transform doesn't call PCA.transform
                    if selector_instance.__class__.__name__ == 'MASTMLFeatureSelector':
                        X_selected = selector_instance.fit(X_normalized, y, X_grouped).transform(X_normalized)
                    else:
                        X_selected = selector_instance.fit(X_normalized, y).transform(X_normalized)
                    log.info("    Saving selected features to csv...")
                    dirname = join(outdir, normalizer_name, selector_name)
                    os.mkdir(dirname)
                    pd.concat([X_selected, X_noinput, y], 1).to_csv(join(dirname, "selected.csv"), index=False)
                    triples.append((normalizer_name, selector_name, X_selected))
            return triples
        normalizer_selector_dataframe_triples = make_normalizer_selector_dataframe_triples()

        ## DataSplits (cross-product)
        ## Collect grouping columns, splitter_to_group_names is a dict of splitter name to grouping col
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

        def do_models_splits():
            all_results = []
            for normalizer_name, selector_name, X in normalizer_selector_dataframe_triples:
                subdir = join(outdir, normalizer_name, selector_name)

                if PlotSettings['feature_vs_target']:
                    #if selector_name == 'DoNothing': continue
                    # for each selector/normalizer, plot y against each x column
                    for column in X:
                        filename = f'{column}_vs_target.png'
                        plot_helper.plot_scatter(X[column], y, join(subdir, filename),
                                                 xlabel=column, ylabel='target_feature', label=y.name)
                for model_name, model_instance in models:
                    for splitter_name, trains_tests in splittername_splitlist_pairs:
                        grouping_data = splitter_to_group_column[splitter_name]
                        subdir = join(normalizer_name, selector_name, model_name, splitter_name)
                        log.info(f"    Running splits for {subdir}")
                        subsubdir = join(outdir, subdir)
                        os.makedirs(subsubdir)
                        # NOTE: do_one_splitter is a big old function, does lots
                        runs = do_one_splitter(X, y, model_instance, subsubdir, trains_tests, grouping_data)
                        all_results.extend(runs)
            return all_results

        return do_models_splits()

    def do_one_splitter(X, y, model, main_path, trains_tests, grouping_data):

        def one_fit(split_num, train_indices, test_indices):

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
            model.fit(train_X, train_y)
            #joblib.dump(model, join(path, "trained_model.pkl"))
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
            if PlotSettings['train_test_plots']:
                plot_helper.make_train_test_plots(
                        split_result, path, is_classification, 
                        label=y.name, groups=grouping_data)

            if is_validation:
                _write_stats(split_result['train_metrics'],
                         split_result['test_metrics'],
                         main_path,
                         split_result['prediction_metrics'],
                         validation_column_names,)
            else:
                _write_stats(split_result['train_metrics'],
                             split_result['test_metrics'],
                             main_path)

            return split_result

        split_results = []
        for split_num, (train_indices, test_indices) in enumerate(trains_tests):
            split_results.append(one_fit(split_num, train_indices, test_indices))

        log.info("    Calculating mean and stdev of scores...")
        def make_train_test_average_and_std_stats():
            train_stats = OrderedDict([('Average Train', None)])
            test_stats  = OrderedDict([('Average Test', None)])
            for name in metrics_dict:
                train_values = [split_result['train_metrics'][name] for split_result in split_results]
                test_values  = [split_result['test_metrics'][name]  for split_result in split_results]
                train_stats[name] = (np.mean(train_values), np.std(train_values))
                test_stats[name]  = (np.mean(test_values), np.std(test_values))
                test_stats_single = dict()
                test_stats_single[name] = (np.mean(test_values), np.std(test_values))
                if grouping_data is not None:
                    unique_groups = np.union1d(split_results[0]['test_groups'], split_results[0]['train_groups'])
                    plot_helper.plot_metric_vs_group(metric=name, groups=unique_groups, stats=test_values,
                                                     avg_stats = test_stats_single, savepath=join(main_path, str(name)+'_vs_group.png'))
            return train_stats, test_stats
        avg_train_stats, avg_test_stats = make_train_test_average_and_std_stats()
        log.info("    Making best/worst plots...")
        def get_best_worst_median_runs():
            # sort splits by the test score of first metric:
            greater_is_better, _ = next(iter(metrics_dict.values())) # get first value pair
            scalar = 1 if greater_is_better else -1
            s = sorted(split_results, key=lambda run: scalar*next(iter(run['test_metrics'])))
            return s[0], s[len(split_results)//2], s[-1]
        worst, median, best = get_best_worst_median_runs()

        def make_pred_vs_true_plots():
            if PlotSettings['predicted_vs_true']:
                plot_helper.plot_best_worst_split(y.values, best, worst,
                                                  join(main_path, 'best_worst_split.png'), label=y.name)
            predictions = [[] for _ in range(X.shape[0])]
            for split_num, (train_indices, test_indices) in enumerate(trains_tests):
                for i, pred in zip(test_indices, split_results[split_num]['y_test_pred']):
                    predictions[i].append(pred)
            if PlotSettings['predicted_vs_true_bars']:
                plot_helper.plot_predicted_vs_true_bars(
                        y.values, predictions, avg_test_stats,
                        join(main_path, 'average_points_with_bars.png'), label=y.name)
            if PlotSettings['best_worst_per_point']:
                plot_helper.plot_best_worst_per_point(y.values, predictions,
                                                      join(main_path, 'best_worst_per_point.png'),
                                                      metrics_dict, avg_test_stats, label=y.name)
            if PlotSettings['average_normalized_errors']:
                plot_helper.plot_normalized_error(y.values, predictions,
                                                  join(main_path, 'average_normalized_errors.png'),
                                                  avg_test_stats)
            if PlotSettings['average_cumulative_normalized_errors']:
                plot_helper.plot_cumulative_normalized_error(y.values, predictions,
                                                  join(main_path, 'average_cumulative_normalized_errors.png'),
                                                  avg_test_stats)

        if not is_classification:
            make_pred_vs_true_plots()

        return split_results

    runs = do_all_combos(X, y, df) # calls do_one_splitter internally

    log.info("Making image html file...")
    html_helper.make_html(outdir)

    log.info("Making html file of all runs stats...")
    _save_all_runs(runs, outdir)

def _instantiate(kwargs_dict, name_to_constructor, category, X_grouped=None, X_indices=None):
    """
    Uses name_to_constructor to instantiate every item in kwargs_dict and return
    the list of instantiations
    """
    instantiations = []
    for long_name, (name, kwargs) in kwargs_dict.items():
        log.debug(f'instantiation: {long_name}, {name}({kwargs})')
        try:
            # Need to construct cv object when have special case of RFECV and LeaveOneGroupOut cross-validation!
            if name == 'RFECV':
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
            instantiations.append((long_name, name_to_constructor[name](**kwargs)))

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
    log.debug(f'models, pre-snatching: \n{models}')
    for selector_name, (_, args_dict) in conf_feature_selection.items():
        if 'estimator' in args_dict:
            model_name = args_dict['estimator']
            try:
                args_dict['estimator'] = models[model_name]
                del models[model_name]
            except KeyError:
                raise utils.MastError(f"The selector {selector_name} specified model {model_name},"
                                      f"which was not found in the [Models] section")
    log.debug(f'models, post-snatching: \n{models}')

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
    with open(join(outdir, 'stats.txt'), 'w') as f:
        f.write("TRAIN:\n")
        for name,score in train_metrics.items():
            f.write(f"{name}: {'%.3f'%float(score)}\n")
        f.write("TEST:\n")
        for name,score in test_metrics.items():
                f.write(f"{name}: {'%.3f'%float(score)}\n")
        if prediction_metrics:
            #prediction metrics now list of dicts for predicting multiple values
            for prediction_metric, prediction_name in zip(prediction_metrics, prediction_names):
                f.write("PREDICTION for "+str(prediction_name)+":\n")
                for name, score in prediction_metric.items():
                    f.write(f"{name}: {'%.3f'%float(score)}\n")

def _exclude_validation(df, validation_column):
    return df.loc[validation_column != 1]

def _only_validation(df, validation_column):
    return df.loc[validation_column == 1]

def check_paths(conf_path, data_path, outdir):
    # Check conf path:
    if os.path.splitext(conf_path)[1] != '.conf':
        raise utils.FiletypeError(f"Conf file does not end in .conf: '{conf_path}'")
    if not os.path.isfile(conf_path):
        raise utils.FileNotFoundError(f"No such file: {conf_path}")

    # Check data path:
    if os.path.splitext(data_path)[1] not in ['.csv', '.xlsx']:
        raise utils.FiletypeError(f"Data file does not end in .csv or .xlsx: '{data_path}'")
    if not os.path.isfile(data_path):
        raise utils.FileNotFoundError(f"No such file: {data_path}")

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
