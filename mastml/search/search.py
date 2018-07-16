" use: python3 -m masmtl.search.search settings.conf data.csv -o results/"

# TODO: use standard logging?

import argparse
import logging
from os.path import join

from sklearn.base import is_classifier
from configobj import ConfigObj
import pandas as pd

from .hill_climbing import climb_hill
from .grid_search import GridSearch
from .genetic_search import GeneticSearch
from .data_handler import DataHandler
from ..legos import model_finder
from .. import mastml, utils, metrics
from ..conf_parser import fix_types

log = logging.getLogger('mastml')

grid_search_user_params = [ # parameters to GridSearch initializer which user has permission to set
    'param_strings', 'model', 'xlabel', 'ylabel', 'fix_random_for_testing',
    'num_cvtests', 'mark_outlying_points', 'num_folds', 'percent_leave_out',
    'processors', 'pop_upper_limit', 'num_bests',
]

genetic_search_user_params = [ # parameters to GeneticSearch initializer which user has permission to set
    'param_strings', 'model', 'num_folds', 'percent_leave_out', 'num_cvtests',
    'mark_outlying_points', 'num_bests', 'fix_random_for_testing', 'processors', 'pop_upper_limit',
    'num_gas', 'ga_pop_size', 'convergence_generations', 'max_generations', 'crossover_prob',
    'mutation_prob', 'shift_prob', 'gen_tol',
]

hill_climbing_user_params = [
    'model', 'score_func', 'num_steps', 'num_restarts'
]

def parse_conf_file(filepath):
    "Accepts the filepath of a conf file and returns its parsed dictionary"
    conf = ConfigObj(filepath)
    if 'GeneralSetup' not in conf:
        raise Exception(f"Conf must contain [GeneralSetup] section")

    section = conf['GeneralSetup']
    for param in ['target_feature', 'istest_feature',]:
        if param not in section or section[param] == 'Auto':
            raise Exception(f"Section [GeneralSetup] cannot omit or Auto '{param}'")
    if 'input_features' not in section or section['input_features'] == 'Auto':
        section['input_features'] = None

    searchers = ['GeneticSearch', 'GridSearch', 'HillClimbing']
    if not any(x in conf for x in searchers):
        raise Exception(f"Conf must contain at least one of {searchers}")

    for searcher in searchers:
        if searcher not in conf: continue
        if 'model' not in conf[searcher]:
            raise Exception(f"{searcher} must specify a model")
        for param, value in conf[searcher].items():
            conf[searcher][param] = fix_types(value)

    def fix_hill_climbing():
        HC = conf['HillClimbing']
        HC['model'] = model_finder.find_model(HC['model']) # NOTE: not instantiated

        if 'score_func' not in HC:
            raise Exception(f"HillCimbing requires score_func parameter")
        is_c = is_classifier(HC['model'])
        metrics_dict =  metrics.classification_metrics if is_c else metrics.regression_metrics
        metrics_dict = {name: (func if greater_is_better else (lambda yt, yp: -func(yt, yp)))
                        for name, (greater_is_better, func) in metrics_dict.items()}
                                             
        task = 'classification' if is_c else 'regression'
        if HC['score_func'] not in metrics_dict:
            raise Exception(f"score_func {HC['score_func']} is not valid for {task}")
        HC['score_func'] = metrics_dict[HC['score_func']]

        param_dict = {}
        for param, value in tuple(HC.items()):
            if param in hill_climbing_user_params: continue
            if not isinstance(value, list):
                raise Exception(
                        f"All model parameters need to be lists in HillClimbing:{param}={value}")
            param_dict[param] = value
            del HC[param]
        HC['param_dict'] = param_dict

    if 'HillClimbing' in conf:
        fix_hill_climbing()

    def fix_grid_search():
        GrS = conf['GridSearch']
        GrS['model'] = model_finder.find_model(GrS['model'])() # NOTE: is instantiated
        for param in GrS:
            if param not in grid_search_user_params:
                raise Exception(f"GridSearch does not take param '{param}'")
    if 'GridSearch' in conf:
        fix_grid_search()

    def fix_genetic_search():
        GeS = conf['GeneticSearch']
        GeS['model'] = model_finder.find_model(GeS['model'])() # NOTE: is instantiated
        for param in GeS:
            if param not in genetic_search_user_params:
                raise Exception(f"GeneticSearch does not take param '{param}'")
    if 'GeneticSearch' in conf:
        fix_genetic_search()

    # TODO: check if the parameters in `param_strings` are valid for `model`

    return conf

def load_data(data_path, input_features, target_feature, istest_feature):
    df = pd.read_csv(data_path)
    if len(df.columns) != len(set(df.columns)):
        raise Exception("Repeated column name in input data")
    if len(df.columns) < 3:
        raise Exception("Input data must have at least three columns")

    if input_features is None: # if user omits it or uses 'Auto'
        input_features = [col for col in df.columns if col!=target_feature and col!=istest_feature]

    return df, input_features, target_feature, istest_feature

def do_run(conf_path, data_path, outdir):
    conf = parse_conf_file(conf_path)
    df, input_features, target_feature, istest_feature = \
            load_data(data_path, **conf['GeneralSetup'])

    def run_grid_search():
        dh = DataHandler(df, input_features=input_features, target_feature=target_feature)
        model = conf['GridSearch'].pop('model')
        param_strings = conf['GridSearch'].pop('param_strings')
        searcher = GridSearch(param_strings, dh, dh, model, outdir, **conf['GridSearch'])
        searcher.run()
    if 'GridSearch' in conf:
        run_grid_search()

    def run_genetic_search():
        dh = DataHandler(df, input_features=input_features, target_feature=target_feature)
        model = conf['GeneticSearch'].pop('model')
        param_strings = conf['GeneticSearch'].pop('param_strings')
        searcher = GeneticSearch(param_strings, dh, dh, model, outdir, **conf['GeneticSearch'])
        searcher.run()
    if 'GeneticSearch' in conf:
        run_genetic_search()

    def run_hill_climbing():
        HC = conf['HillClimbing']
        X = df[input_features]
        y = df[target_feature]
        model      = HC.pop('model')
        score_func = HC.pop('score_func')
        param_dict = HC.pop('param_dict')
        best_score, best_params = climb_hill(model, X, y, param_dict, score_func, **HC)
        string = f"HILL CLIMBING: best_score={best_score}, best_params={best_params}"
        print(string)
        with open(join(outdir, 'hill_climbing.txt'), 'w') as f:
            f.write(string)
    if 'HillClimbing' in conf:
        run_hill_climbing()

    print("Done!")

if __name__ == '__main__':
    conf_path, data_path, outdir, verbosity = mastml.get_commandline_args() # argparse stuff
    conf_path, data_path, outdir = mastml.check_paths(conf_path, data_path, outdir)
    utils.activate_logging(outdir, (conf_path, data_path, outdir), verbosity=verbosity)

    try:
        do_run(conf_path, data_path, outdir)
    except utils.MastError as e:
        # catch user errors, log and print, but don't raise and show them that nasty stack
        log.error(str(e))
    except Exception as e:
        # catch the error, save it to file, then raise it back up
        log.error('A runtime exception has occured, please go to '
                  'https://github.com/uw-cmg/MAST-ML/issues and post your issue.')
        log.exception(e)
        raise e
