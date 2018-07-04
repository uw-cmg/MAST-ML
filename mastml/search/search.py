" use: python3 -m masmtl.search.search settings.conf data.csv -o results/"

# TODO: use standard logging?

import argparse
import logging

from configobj import ConfigObj
import pandas as pd

from .grid_search import GridSearch
from .genetic_search import GeneticSearch
from .data_handler import DataHandler
from ..legos import model_finder
from .. import mastml, utils
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

    if 'GeneticSearch' not in conf and 'GridSearch' not in conf:
        raise Exception(f"Conf must contain at least one of [GeneticSearch] and [GridSearch] sections.")

    for name, valid_params in [['GridSearch',    grid_search_user_params],
                               ['GeneticSearch', genetic_search_user_params]]:
        if name not in conf: continue
        section = conf[name]
        for param in ['param_strings', 'model']:
            if param not in section:
                raise Exception(f"Section [{name}] must specify parameter '{param}'")
        for param in section:
            if param not in valid_params:
                raise Exception(f"Section [{name}] doesn't support parameter '{param}'")
            section[param] = fix_types(section[param]) # turn 'True' to True, '5' to 5, etc

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
    df, input_features, target_feature, istest_feature = load_data(data_path, **conf['GeneralSetup'])
    # something like this:
    dh = DataHandler(df, input_features=input_features, target_feature=target_feature)
    for name, constructor in [['GridSearch', GridSearch] , ['GeneticSearch', GeneticSearch]]:
        if name not in conf: continue
        section = conf[name]
        model = model_finder.name_to_constructor[section['model']]() # TODO: make nice errors
        params = section['param_strings'] # TODO: Any preprocessing necessary?
        del section['param_strings']
        del section['model']
        searcher = constructor(params, dh, dh, model, outdir, **section)
        searcher.run() # TODO: save result somehow?
    print("Done!")

if __name__ == '__main__':
    conf_path, data_path, outdir, verbosity = mastml.get_commandline_args() # argparse stuff
    mastml.check_paths(conf_path, data_path, outdir)
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
