" Does one big mastml run "

import argparse
import itertools
import os.path
import shutil

import pandas as pd
import sklearn.pipeline
import sklearn.model_selection

from . import conf_parser, data_loader, plot_helper
from .legos import data_splitters, feature_generators, feature_normalizers, feature_selectors, model_finder, utils

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
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print(f"Saving to directory 'outdir'")

    # Load in and parse the configuration and data files:
    conf = conf_parser.parse_conf_file(conf_path)
    df, input_features, target_feature = data_loader.load_data(data_path, **conf['GeneralSetup'])

    # Instantiate all the sections of the conf file:
    generators = _instantiate(conf['FeatureGeneration'], feature_generators.name_to_constructor, 'feature generator')
    normalizers = _instantiate(conf['FeatureNormalization'], feature_normalizers.name_to_constructor, 'feature normalizer')
    selectors = _instantiate(conf['FeatureSelection'], feature_selectors.name_to_constructor, 'feature selector')
    models = _instantiate(conf['Models'], model_finder.name_to_constructor, 'model')
    splitters = _instantiate(conf['DataSplits'], data_splitters.name_to_constructor, 'data split')

    # Build necessary units for main
    generators_union = utils.DataFrameFeatureUnion(generators)
    X, y = df[input_features], df[target_feature]
    splits = [pair for splitter in splitters for pair in splitter.split(X,y)]

    # NOTE: This is done using itertools.product instead of sklearn.model_selection.GridSearchCV
    #  because we need to heavily inspect & use & modify data within each iteration of the loop.
    for normalizer, selector, model in itertools.product(normalizers, selectors, models):
        pipe = sklearn.pipeline.make_pipeline(generators_union, normalizer, selector, model)
        for split_num, (train_indices, test_indices) in enumerate(splits):
            train_X, train_y = df.loc[train_indices, input_features], df.loc[train_indices, target_feature]
            test_X,  test_y  = df.loc[test_indices,  input_features], df.loc[test_indices,  target_feature]
            pipe.fit(train_X, train_y)
            train_predictions = pipe.predict(train_X)
            test_predictions  = pipe.predict(test_X)
            filename = f"split_{split_num}" \
                       f"_normalizer_{normalizer.__class__.__name__}" \
                       f"_selector_{selector.__class__.__name__}" \
                       f"_model_{model.__class__.__name__}.png"
            plot_helper.plot_confusion_matrix(test_y.values, test_predictions, filename=os.path.join(outdir, filename))

    # Copy the original input files to the output directory for easy reference
    shutil.copy2(conf_path, outdir)
    shutil.copy2(data_path, outdir)

    # TODO: plotting
    # TODO: statistics
    # TODO: save cached models
    # TODO: create intermediate "save to csv" lego blocks
    # TODO: clustering legos
    # TODO: Get best, worst, average runs
    # TODO: model caching and csv saving after data generation
    # TODO: address the great dataframe vs array debate for pipelines
    # maybe TODO: put feature generation once at the beginning only

def _instantiate(kwargs_dict, name_to_constructor, category):
    "Uses name_to_constructor to instantiate ever item in kwargs_dict and return the list of instantiations"
    instantiations = []
    for name, kwargs in kwargs_dict.items():
        try:
            instantiations.append(name_to_constructor[name](**kwargs))
        except TypeError:
            raise TypeError(f"The {category} '{name}' has invalid parameters: {args}")
            # TODO: print class argument and doc string
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

