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

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.exceptions import UndefinedMetricWarning

from . import conf_parser, data_loader, html_helper, plot_helper, utils
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
    df, X, y = data_loader.load_data(data_path,
                                     conf['GeneralSetup']['input_features'],
                                     conf['GeneralSetup']['target_feature'])

    if conf['PlotSettings']['target_histogram']:
        plot_helper.plot_target_histogram(y, join(outdir, 'target_histogram.png'))

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

    def snatch_model_for_learning_curve():
        PS = conf['PlotSettings']
        GS = conf['GeneralSetup']
        if PS['data_learning_curve'] or PS['feature_learning_curve']:
            name = GS['learning_curve_model']
            GS['learning_curve_model'] = models[name]
            del models[name]
    snatch_model_for_learning_curve()

    models = list(models.items())

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
    selectors   = _instantiate(conf['FeatureSelection'],
                               feature_selectors.name_to_constructor,
                               'featureselector')
    splitters   = _instantiate(conf['DataSplits'],
                               data_splitters.name_to_constructor,
                               'datasplit')

    log.debug(f'generators: \n{generators}')
    log.debug(f'clusterers: \n{clusterers}')
    log.debug(f'normalizers: \n{normalizers}')
    log.debug(f'selectors: \n{selectors}')
    log.debug(f'splitters: \n{splitters}')

    def do_combos(X):
        log.info(f"There are {len(normalizers)} feature normalizers, {len(selectors)} feature"
                 f"selectors {len(models)} models, and {len(splitters)} splitters.")

        def generate_features():
            log.info("Doing feature generation...")
            dataframes = [instance.fit_transform(df, y) for _, instance in generators]
            dataframe = pd.concat(dataframes, 1)

            log.info("Saving generated data to csv...")
            log.debug(f'generated cols: {dataframe.columns}')
            filename = join(outdir, "generated_features.csv")
            pd.concat([dataframe, y], 1).to_csv(filename, index=False)
            return dataframe
        generated_df = generate_features()

        def remove_constants():
            dataframe = _remove_constant_features(generated_df)
            log.info("Saving generated data without constant columns to csv...")
            filename = join(outdir, "generated_features_no_constant_columns.csv")
            pd.concat([dataframe, y], 1).to_csv(filename, index=False)
            return dataframe
        generated_df = remove_constants()

        # add in generated features
        # TODO this adds in some of the grouping features, can the model cheat on these?
        X = pd.concat([X, generated_df], axis=1)

        # remove repeat columns (keep the first one)
        def remove_repeats(X):
            repeated_columns = X.loc[:, X.columns.duplicated()].columns
            if not repeated_columns.empty:
                log.warning(f"Throwing away {len(repeated_columns)} because they are repeats.")
                log.debug(f"Throwing away columns because they are repeats: {repeated_columns}")
                X = X.loc[:,~X.columns.duplicated()]
            return X
        X = remove_repeats(X)

        def make_clusters():
            log.info("Doing clustering...")
            clustered_df = pd.DataFrame()
            for name, instance in clusterers:
                clustered_df[name] = instance.fit_predict(X, y)
            return clustered_df
        clustered_df = make_clusters()

        def plot_scatters(): # put it in a function for namespace preserving
            plots_made = 0
            if clustered_df.empty:
                # plot y against each x column
                for column in X:
                    filename = f'{column}_vs_target_scatter.png'
                    plots_made += 1
                    if plots_made > 10: break
                    plot_helper.plot_scatter(X[column], y, join(outdir, filename),
                                             xlabel=column, ylabel='target_feature')
            else:
                # for each cluster, plot y against each x column
                for name in clustered_df.columns:
                    for column in X:
                        filename = f'{column}_vs_target_by_{name}_scatter.png'
                        plots_made += 1
                        if plots_made > 10: break
                        plot_helper.plot_scatter(X[column], y, join(outdir, filename),
                                                clustered_df[name], xlabel=column,
                                                ylabel='target_feature')
        if PlotSettings['feature_vs_target']:
            plot_scatters()

        log.info("Saving clustered data to csv...")
        pd.concat([clustered_df, y], 1).to_csv(join(outdir, "clusters.csv"), index=False)

        def normalize_and_select():
            triples = []
            for normalizer_name, normalizer_instance in normalizers:

                log.info(f"Running normalizer {normalizer_name} ...")
                X_normalized = normalizer_instance.fit_transform(X, y)

                log.info("Saving normalized data to csv...")
                dirname = join(outdir, normalizer_name)
                os.mkdir(dirname)
                pd.concat([X_normalized, y], 1).to_csv(join(dirname, "normalized.csv"), index=False)

                # FeatureSelection (cross-product)
                log.info("Running selectors...")
                for selector_name, selector_instance in selectors:

                    log.info(f"    Running selector {selector_name} ...")
                    # NOTE: Changed from fit_transform because PCA's fit_transform
                    #       doesn't call transform (does transformation itself).
                    X_selected = selector_instance.fit(X_normalized, y).transform(X_normalized)

                    log.info("    Saving selected features to csv...")
                    dirname = join(outdir, normalizer_name, selector_name)
                    os.mkdir(dirname)
                    pd.concat([X_selected, y], 1).to_csv(join(dirname, "selected.csv"), index=False)

                    triples.append((normalizer_name, selector_name, X_selected))
            return triples
        post_selection = normalize_and_select()

        ## DataSplits (cross-product)
        ## Collect grouping columns, splitter_to_group_names is a dict of splitter name to grouping col
        log.debug("Finding splitter-required columns in data...")
        def make_splits():
            splits = []
            for name, instance in splitters:
                if name in splitter_to_group_names:
                    col = splitter_to_group_names[name]
                    log.debug(f"    Finding {col} for {name}...")
                    for df_ in [clustered_df, X, df]: # TODO: Are these the right objects to check?
                        if col in df_.columns:
                            splits.append((name, tuple(instance.split(X, y, df_[col].values))))
                            break
                    else:
                        raise utils.MissingColumnError(f'DataSplit {name} needs column {col}, which'
                                                       f'was neither generated nor given by input')
                else:
                    splits.append((name, tuple(instance.split(X, y))))
            return splits
        splits = make_splits()

        log.info("Fitting models to splits...")

        def do_models_splits():
            all_results = []
            for normalizer_name, selector_name, X in post_selection:
                subdir = join(outdir, normalizer_name, selector_name)
                if conf['PlotSettings']['data_learning_curve']:
                    learning_curve_model = conf['GeneralSetup']['learning_curve_model']
                    learning_curve_score = conf['GeneralSetup']['learning_curve_score']
                    plot_helper.plot_sample_learning_curve(
                            learning_curve_model, X, y, learning_curve_score,
                            2, join(subdir, f'learning_curve.png'))

                if conf['PlotSettings']['feature_learning_curve']:
                    learning_curve_model = conf['GeneralSetup']['learning_curve_model']
                    learning_curve_score = conf['GeneralSetup']['learning_curve_score']
                    plot_helper.plot_feature_learning_curve(
                            learning_curve_model, X, y, learning_curve_score,
                            join(subdir, f'learning_curve.png'))

                if PlotSettings['feature_vs_target']:
                    if selector_name == 'DoNothing': continue
                    # for each selector/normalizer, plot y against each x column
                    for column in X:
                        filename = f'{column}_vs_target.png'
                        plot_helper.plot_scatter(X[column], y, join(subdir, filename),
                                                 xlabel=column, ylabel='target_feature')
                for model_name, model_instance in models:
                    for splitter_name, trains_tests in splits:
                        subdir = join(normalizer_name, selector_name, model_name, splitter_name)
                        log.info(f"    Running splits for {subdir}")
                        subsubdir = join(outdir, subdir)
                        os.makedirs(subsubdir)
                        runs = do_splitter(X, model_instance, subsubdir, trains_tests,)
                        all_results.extend(runs)
            return all_results

        return do_models_splits()

    def do_splitter(X, model, main_path, trains_tests):

        def one_fit(split_num, train_indices, test_indices):
            log.info(f"        Doing split number {split_num}")
            train_X, train_y = X.loc[train_indices], y.loc[train_indices]
            test_X,  test_y  = X.loc[test_indices],  y.loc[test_indices]

            path = join(main_path, f"split_{split_num}")
            os.mkdir(path)

            log.info("             Fitting model and making predictions...")
            model.fit(train_X, train_y)
            #joblib.dump(model, join(path, "trained_model.pkl"))
            train_pred = model.predict(train_X)
            test_pred  = model.predict(test_X)

            # Save train and test data and results to csv:
            log.info("             Saving train/test data and predictions to csv...")
            train_pred_series = pd.DataFrame(train_pred, columns=['train_pred'], index=train_indices)
            pd.concat([train_X, train_y, train_pred_series], 1)\
                    .to_csv(join(path, 'train.csv'), index=False)
            test_pred_series = pd.DataFrame(test_pred,   columns=['test_pred'],  index=test_indices)
            pd.concat([test_X,  test_y,  test_pred_series],  1)\
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
                                            for name, function in metrics_dict.items())
                test_metrics = OrderedDict((name, function(test_y, test_pred))
                                           for name, function in metrics_dict.items())

            split_result = OrderedDict(
                normalizer = split_path[-4],
                selector = split_path[-3],
                model = split_path[-2],
                splitter = split_path[-1],
                split_num = split_num,
                y_train_true = train_y.values,
                y_train_pred = train_pred,
                y_test_true  = test_y.values,
                y_test_pred  = test_pred,
                train_metrics = train_metrics,
                test_metrics  = test_metrics
            )

            log.info("             Making plots...")
            if PlotSettings['main_plots']:
                plot_helper.make_main_plots(split_result, path, is_classification)
            _write_stats(split_result['train_metrics'],
                         split_result['test_metrics'],
                         main_path)

            return split_result

        split_results = []
        for split_num, (train_indices, test_indices) in enumerate(trains_tests):
            split_results.append(one_fit(split_num, train_indices, test_indices))

        log.info("    Calculating mean and stdev of scores...")
        def make_stats():
            train_stats = OrderedDict([('Average Train', None)])
            test_stats  = OrderedDict([('Average Test', None)])
            for name in metrics_dict:
                train_values = [split_result['train_metrics'][name] for split_result in split_results]
                test_values  = [split_result['test_metrics'][name]  for split_result in split_results]
                train_stats[name] = (np.mean(train_values),
                                     np.std(train_values) / np.sqrt(len(train_values)))
                test_stats[name]  = (np.mean(test_values),
                                     np.std(test_values) / np.sqrt(len(test_values)))
            return train_stats, test_stats
        avg_train_stats, avg_test_stats = make_stats()

        log.info("    Making best/worst plots...")
        # sort splits by the test score of first metric:
        split_results.sort(key=lambda run: list(run['test_metrics'].items())[0][1])
        worst, median, best = split_results[0], split_results[len(split_results)//2], split_results[-1]

        # collect all predictions in a combo for each point in the dataset
        def do_plots():
            if PlotSettings['predicted_vs_true']:
                plot_helper.plot_best_worst_split(best, worst,
                                                  join(main_path, 'best_worst_split.png'))
            predictions = [[] for _ in range(X.shape[0])]
            for split_num, (train_indices, test_indices) in enumerate(trains_tests):
                for i, pred in zip(test_indices, split_results[split_num]['y_test_pred']):
                    predictions[i].append(pred)
            if PlotSettings['predicted_vs_true_bars']:
                plot_helper.plot_predicted_vs_true_bars(y.values, predictions, avg_test_stats,
                                                        join(main_path, 'best_worst_average.png'))
            if PlotSettings['best_worst_per_point']:
                plot_helper.plot_best_worst_per_point(y.values, predictions,
                                                      join(main_path, 'best_worst_per_point.png'),
                                                      metrics_dict, avg_test_stats)
        if not is_classification:
            do_plots()

        return split_results


    runs = do_combos(X)

    log.info("Making image html file...")
    html_helper.make_html(outdir)

    log.info("Making html file of all runs stats...")
    _save_all_runs(runs, outdir)

def _instantiate(kwargs_dict, name_to_constructor, category):
    """
    Uses name_to_constructor to instantiate every item in kwargs_dict and return
    the list of instantiations
    """

    instantiations = []
    for long_name, (name, kwargs) in kwargs_dict.items():
        log.debug(f'instantiation: {long_name}, {name}({kwargs})')
        try:
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

def _write_stats(train_metrics, test_metrics, outdir):
    with open(join(outdir, 'stats.txt'), 'w') as f:
        f.write("TRAIN:\n")
        for name,score in train_metrics.items():
            f.write(f"{name}: {score}\n")
        f.write("TEST:\n")
        for name,score in test_metrics.items():
                f.write(f"{name}: {score}\n")

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
