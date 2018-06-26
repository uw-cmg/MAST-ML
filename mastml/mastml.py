"""
Module for getting a mastml system call and calling all the appropriate subroutines
"""

import argparse
import inspect
import itertools
import os
import shutil
import warnings
import logging
import time

from collections import OrderedDict
from os.path import join # We use join tons

import numpy as np
import pandas as pd
from sklearn.externals import joblib

from . import conf_parser, data_loader, html_helper, plot_helper, metrics, utils
from .legos import data_splitters, feature_generators, feature_normalizers, feature_selectors, model_finder, util_legos

log = logging.getLogger('mastml')

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
        log.warning("Output dir already exists. Deleting...")
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    log.info(f"Saving to directory 'outdir'")

def mastml_run(conf_path, data_path, outdir):
    " Runs operations specifed in conf_path on data_path and puts results in outdir "

    # Load in and parse the configuration and data files:
    conf = conf_parser.parse_conf_file(conf_path)
    df, input_features, target_feature = data_loader.load_data(data_path, **conf['GeneralSetup'])

    # Get the appropriate collection of metrics:
    big_metrics_dict = metrics.classification_metrics if conf['is_classification'] else metrics.regression_metrics
    metrics_dict = {name: big_metrics_dict[name] for name in conf['metrics']}

    # Instantiate all the sections of the conf file:
    generators  = _instantiate(conf['FeatureGeneration'],    feature_generators.name_to_constructor,  'feature generator')
    normalizers = _instantiate(conf['FeatureNormalization'], feature_normalizers.name_to_constructor, 'feature normalizer')
    selectors   = _instantiate(conf['FeatureSelection'],     feature_selectors.name_to_constructor,   'feature selector')
    models      = _instantiate(conf['Models'],               model_finder.name_to_constructor,        'model')
    splitters   = _instantiate(conf['DataSplits'],           data_splitters.name_to_constructor,      'data split')

    X, y = df[input_features], df[target_feature]
    runs = _do_combos(X, y, generators, normalizers, selectors, models, splitters,
                      metrics_dict, outdir, conf['is_classification'])

    log.info("Making image html file...")
    html_helper.make_html(outdir)

    log.info("Making html file of all runs stats...")
    _save_all_runs(runs, outdir)

    # Copy the original input files to the output directory for easy reference
    log.info("Copying input files to output directory...")
    shutil.copy2(conf_path, outdir)
    shutil.copy2(data_path, outdir)


def _do_combos(X, y, generators, normalizers, selectors, models, splitters,
               metrics_dict, outdir, is_classification):
    """
    Uses cross product to generate, normalize, and select the input data, then saves it.
    Calls _do_splits for actual model fits.
    """

    log.info(f"There are {len(normalizers)} feature normalizers, {len(selectors)} feature selectors,"
          f" {len(models)} models, and {len(splitters)} splitters.")

    log.info("Doing feature generation...")
    generators_union = util_legos.DataFrameFeatureUnion([instance for name,instance in generators])
    X_generated = generators_union.fit_transform(X, y)

    log.info("Saving generated data to csv...")
    pd.concat([X_generated, y], 1).to_csv(join(outdir, "data_generated.csv"), index=False)

    post_selection = []
    for normalizer_name, normalizer_instance in normalizers:

        log.info("Running normalizer {normalizer_name} ...")
        X_normalized = normalizer_instance.fit_transform(X_generated, y)

        log.info("Saving normalized data to csv...")
        dirname = join(outdir, normalizer_name)
        os.mkdir(dirname)
        pd.concat([X_normalized, y], 1).to_csv(join(dirname, "normalized.csv"), index=False)

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

            post_selection.append((normalizer_name, selector_name, X_selected))

    splits = [(name, tuple(instance.split(X, y))) for name, instance in splitters]

    log.info("Fitting models to splits...")
    all_results = []

    for normalizer_name, selector_name, df in post_selection:
        for model_name, model_instance in models:
            for splitter_name, trains_tests in splits:
                subdir = join(normalizer_name, selector_name, model_name, splitter_name)
                log.info(f"    Running splits for {subdir}")
                path = join(outdir, subdir)
                os.makedirs(path)
                runs = _do_splits(df, y, model_instance, path, metrics_dict, trains_tests, is_classification)
                all_results.extend(runs)

    return all_results


def _do_splits(X, y, model, main_path, metrics_dict, trains_tests, is_classification):
    """
    For a fixed normalizer,selector,model,splitter,
    train and test the model on each split that the splitter makes
    """
    split_results = []
    for split_num, (train_indices, test_indices) in enumerate(trains_tests):

        #for parameters in model_parameters_list:
        #    pass

        log.info(f"        Doing split number {split_num}")
        train_X, train_y = X.loc[train_indices], y.loc[train_indices]
        test_X,  test_y  = X.loc[test_indices],  y.loc[test_indices]

        path = join(main_path, f"split_{split_num}")
        os.mkdir(path)

        log.info("             Fitting model and making predictions...")
        model.fit(train_X, train_y)
        joblib.dump(model, join(path, "trained_model.pkl"))
        train_pred = model.predict(train_X)
        test_pred  = model.predict(test_X)

        # Save train and test data and results to csv:
        log.info("             Saving train/test data and predictions to csv...")
        train_pred_series = pd.DataFrame(train_pred, columns=['train_pred'], index=train_indices)
        pd.concat([train_X, train_y, train_pred_series], 1).to_csv(join(path, 'train.csv'), index=False)
        test_pred_series = pd.DataFrame(test_pred,   columns=['test_pred'],  index=test_indices)
        pd.concat([test_X,  test_y,  test_pred_series],  1).to_csv(join(path, 'test.csv'),  index=False)

        log.info("             Calculating score metrics...")
        split_path = main_path.split(os.sep)
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
            train_metrics = OrderedDict((name, function(train_y, train_pred))
                                        for name,function in metrics_dict.items()),
            test_metrics  = OrderedDict((name, function(test_y, test_pred))
                                        for name,function in metrics_dict.items()),
        )


        log.info("             Making plots...")
        plot_helper.make_plots(split_result, path, is_classification)

        split_results.append(split_result)

    log.info("    Calculating mean and stdev of scores...")
    train_stats = OrderedDict()
    test_stats  = OrderedDict()
    for name in metrics_dict:
        train_values = [split_result['train_metrics'][name] for split_result in split_results]
        test_values  = [split_result['test_metrics'][name]  for split_result in split_results]
        train_stats[name] = (np.mean(train_values), np.std(train_values))
        test_stats[name]  = (np.mean(test_values),  np.std(test_values))

    log.info("    Making best/worst plots...")
    split_results.sort(key=lambda run: list(run['test_metrics'].items())[0][1]) # sort splits by the test score of first metric
    worst, median, best = split_results[0], split_results[len(split_results)//2], split_results[-1]
    if not is_classification:
        plot_helper.plot_best_worst(best['y_test_true'], best['y_test_pred'], worst['y_test_true'],
                                    worst['y_test_pred'], os.path.join(main_path, 'best_worst_overlay.png'), test_stats)

    return split_results


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


def _instantiate(kwargs_dict, name_to_constructor, category):
    """
    Uses name_to_constructor to instantiate every item in kwargs_dict and return
    the list of instantiations
    """
    instantiations = []
    for long_name, (name, kwargs) in kwargs_dict.items():
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAterials Science Toolkit - Machine Learning')

    parser.add_argument('conf_path', type=str, help='path to mastml .conf file')
    parser.add_argument('data_path', type=str, help='path to csv or xlsx file')
    parser.add_argument('-o', action="store", dest='outdir', default='results',
                        help='Folder path to save output files to. Defaults to results/')

    args = parser.parse_args()

    check_paths(os.path.abspath(args.conf_path),
                os.path.abspath(args.data_path),
                os.path.abspath(args.outdir))

    utils.activate_logging(args.outdir, args)


    try:
        mastml_run(os.path.abspath(args.conf_path),
                   os.path.abspath(args.data_path),
                   os.path.abspath(args.outdir))
    except utils.MastError as e:
        # catch user errors, log and print, but don't raise and show them that nasty stack
        log.error(str(e))
    except Exception as e:
        # catch the error, save it to file, then raise it back up
        log.error('A runtime exception has occured, please go to '
                      'https://github.com/uw-cmg/MAST-ML/issues and post your issue.')
        log.exception(e)
        raise e
