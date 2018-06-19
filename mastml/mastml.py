" Does one big mastml run "

import sys
import argparse
import collections
import inspect
import itertools
import os.path
import shutil
import collections
import warnings

import pandas as pd
import sklearn.pipeline
import sklearn.model_selection
from sklearn.externals import joblib

from . import conf_parser, data_loader, html_helper, plot_helper, metrics, utils
from .legos import data_splitters, feature_generators, feature_normalizers, feature_selectors, model_finder, util_legos

utils.Tee('log.txt', 'w')
utils.TeeErr('errors.txt', 'w')

def mastml_run(conf_path, data_path, outdir):
    " Runs operations specifed in conf_path on data_path and puts results in outdir "
    
    # Check conf path:
    if os.path.splitext(conf_path)[1] != '.conf':
        raise Exception(f"Conf file does not end in .conf: '{conf_path}'")
    if not os.path.isfile(conf_path):
        raise FileNotFoundError(f"No such file: {conf_path}")

    # Check data path:
    if os.path.splitext(data_path)[1] not in ['.csv','.xlsx']:
        raise Exception(f"Data file does not end in .csv or .xlsx: '{data_path}'")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"No such file: {data_path}")

    # Check output directory:
    if os.path.exists(outdir):
        print("Output dir already exists. Press enter to delete it or ctrl-c to cancel.")
        input()
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    print(f"Saving to directory 'outdir'")

    # Load in and parse the configuration and data files:
    conf = conf_parser.parse_conf_file(conf_path)
    df, input_features, target_feature = data_loader.load_data(data_path, **conf['GeneralSetup'])

    # Get the appropriate collection of metrics:
    metrics_dict = metrics.classification_metrics if conf['is_classification'] else metrics.regression_metrics

    # Instantiate all the sections of the conf file:
    generators = _instantiate(conf['FeatureGeneration'], feature_generators.name_to_constructor, 'feature generator')
    normalizers = _instantiate(conf['FeatureNormalization'], feature_normalizers.name_to_constructor, 'feature normalizer')
    selectors = _instantiate(conf['FeatureSelection'], feature_selectors.name_to_constructor, 'feature selector')
    models = _instantiate(conf['Models'], model_finder.name_to_constructor, 'model')
    splitters = _instantiate(conf['DataSplits'], data_splitters.name_to_constructor, 'data split')

    X, y = df[input_features], df[target_feature]
    runs = _do_fits(X, y, generators, normalizers, selectors, models, splitters, conf['metrics'], metrics_dict, outdir, conf['is_classification'])

    #print("Saving images...")
    #image_paths = plot_helper.make_plots(runs, conf['is_classification'], outdir)

    print("Making image html file...")
    #html_helper.make_html(outdir, image_paths, data_path, ['computed csv.notcsv'], conf_path,
    #        os.path.join(outdir, 'results.html'), 'errors.txt', 'debug.txt', best=None, median=None, worst=None)

    print("Making data html file...")
    # Save a table of all the runs to an html file
    # We have to flatten the subdicts to make it all fit in one table
    _save_all_runs(runs, outdir)

    # Copy the original input files to the output directory for easy reference
    print("Copying input files to output directory...")
    shutil.copy2(conf_path, outdir)
    shutil.copy2(data_path, outdir)

    # TODO: clustering legos
    # TODO: implement is_trainable column in csv
    # TODONE: model caching and csv saving after data generation
    # TODONE: address the great dataframe vs array debate for pipelines
    # TODONE: save cached models
    # TODONE: Get best, worst, average runs
    # TODONE: put feature generation once at the beginning only
    # TODONE: create intermediate "save to csv" lego blocks

def _do_fits(X, y, generators, normalizers, selectors, models, splitters, metrics, metrics_dict, outdir, is_classification):
    ospj = os.path.join

    generators_union = util_legos.DataFrameFeatureUnion(generators)
    splits = [pair for splitter in splitters for pair in splitter.split(X,y)]

    print("Doing feature generation & normalization & selection...")
    X_generated = generators_union.fit_transform(X, y)
    X_generated.to_csv(ospj(outdir, "data_generated.csv"))
    Xs_selected = []
    for normalizer in normalizers:
        dirname = ospj(outdir, normalizer.__class__.__name__)
        os.mkdir(dirname)
        X_normalized = normalizer.fit_transform(X_generated, y)
        X_normalized.to_csv(ospj(dirname, "normalized.csv"))

        for selector in selectors:
            dirname = ospj(outdir, normalizer.__class__.__name__, selector.__class__.__name__)
            os.mkdir(dirname)
            X_selected = selector.fit_transform(X_normalized, y)
            X_selected.to_csv(ospj(dirname, "selected.csv"))
            Xs_selected.append((normalizer, selector, X_selected))

    print("Fitting models to datas...")
    num_fits = len(normalizers) * len(selectors) * len(models) * len(splits)
    runs = [None] * num_fits
    for fit_num, ((normalizer, selector, X_selected), model, (split_num, (train_indices, test_indices))) \
            in enumerate(itertools.product(Xs_selected, models, enumerate(splits))):
        path = ospj(outdir, normalizer.__class__.__name__, selector.__class__.__name__,
                    model.__class__.__name__, 'split_'+str(split_num))
        os.makedirs(path)
        #import pdb; pdb.set_trace()
        train_X, train_y = X_selected.loc[train_indices], y.loc[train_indices]
        test_X,  test_y  = X_selected.loc[test_indices], y.loc[test_indices]
        print(f"Fit number {fit_num}/{num_fits}")
        model.fit(train_X, train_y)
        joblib.dump(model, ospj(path, "trained_model.pkl"))
        train_pred = model.predict(train_X)
        test_pred  = model.predict(test_X)
        pd.concat([train_X, train_y, pd.DataFrame(train_pred,columns=['train_pred'])], axis=1).to_csv(ospj(path, 'train.csv'))
        pd.concat([test_X, test_y, pd.DataFrame(test_pred,columns=['test_pred'])], axis=1).to_csv(ospj(path, 'test.csv'))
        run = collections.OrderedDict( # The ordered dict ensures column order in saved html
            split=str(split_num),
            normalizer=normalizer.__class__.__name__,
            selector=selector.__class__.__name__,
            model=model.__class__.__name__,
            train_true=train_y.values,
            train_pred=train_pred,
            test_true=test_y.values,
            test_pred=test_pred,
            train_metrics=[], # a list of (metric_name, value) tuples
            test_metrics=[],
        )
        for name in metrics:
            metric = metrics_dict[name]
            run['train_metrics'].append( (name, metric(train_y, train_pred)) )
            run['test_metrics'].append(  (name, metric(test_y,  test_pred))  )

        # TODO: add train data plotting
        if is_classification:
            plot_helper.plot_confusion_matrix(test_y.values, test_pred, ospj(path, 'test_confusion_matrix.png'), run['test_metrics'])
            plot_helper.plot_confusion_matrix(train_y.values, train_pred, ospj(path, 'train_confusion_matrix.png'), run['train_metrics'])
        else: # is_regression
            plot_helper.plot_predicted_vs_true(test_y.values, test_pred, ospj(path, 'test_predicted_vs_true.png'), run['test_metrics'])
            plot_helper.plot_residuals_histogram(test_y.values, test_pred, ospj(path, 'test_residuals_histogram.png'), run['test_metrics'])
            plot_helper.plot_predicted_vs_true(train_y.values, train_pred, ospj(path, 'train_predicted_vs_true.png'), run['train_metrics'])
            plot_helper.plot_residuals_histogram(train_y.values, train_pred, ospj(path, 'train_residuals_histogram.png'), run['train_metrics'])
        with open(ospj(path, 'stats.txt'), 'w') as f:
            f.write("TRAIN:\n")
            for name,score in run['train_metrics']:
                f.write(f"{name}: {score}\n")
            f.write("TEST:\n")
            for name,score in run['test_metrics']:
                f.write(f"{name}: {score}\n")

        runs[fit_num] = run
    
    return runs

def _save_all_runs(runs, outdir):
    for_table = []
    for run in runs:
        od = collections.OrderedDict()
        for name, value in run.items():
            if name == 'train_metrics':
                for k,v in run['train_metrics']:
                    od['train_'+k] = v
            elif name == 'test_metrics':
                for k,v in run['test_metrics']:
                    od['test_'+k] = v
            else:
                od[name] = value
        for_table.append(od)
    pd.DataFrame(for_table).to_html(outdir + 'results.html')

def _instantiate(kwargs_dict, name_to_constructor, category):
    "Uses name_to_constructor to instantiate ever item in kwargs_dict and return the list of instantiations"
    instantiations = []
    for name, kwargs in kwargs_dict.items():
        try:
            instantiations.append(name_to_constructor[name](**kwargs))
        except TypeError:
            print(f"ARGUMENTS FOR '{name}':", inspect.signature(name_to_constructor[name]))
            raise TypeError(f"The {category} '{name}' has invalid parameters: {kwargs}\n"
                            f"The arguments for '{name}' are printed above the call stack.")
        except KeyError:
            raise KeyError(f"There is no {category} called '{name}'.")
    return instantiations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAterials Science Toolkit - Machine Learning')

    parser.add_argument('conf_path', type=str, help='path to mastml .conf file')
    parser.add_argument('data_path', type=str, help='path to csv or xlsx file')
    parser.add_argument('-o', action="store", dest='outdir', default='results',
            help='Folder path to save output files to. Defaults to results/')

    args = parser.parse_args()

    mastml_run(os.path.abspath(args.conf_path),
               os.path.abspath(args.data_path),
               os.path.abspath(args.outdir))

