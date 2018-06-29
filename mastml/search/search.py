" use: python3 -m masmtl.search.search settings.conf data.csv -o results/"

import argparse

from .grid_search import GridSearch
from .genetic_search import GeneticSearch
from .data_handler import DataHandler
from ..legos import model_finder
from .. import mastml, utils

genetic_search_user_params = [ # parameters to GridSearch initializer which user has permission to set
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
    # TODO

def do_run(conf_path, data_path, outdir):
    conf = parse_conf_file(conf_path)
    df = pd.read_csv(data_path)
    # something like this:
    #dh = DataHandler(df, input_features=['a','b'], target_feature='y')
    #gs = GeneticSearch(params, dh, dh, model, './results/', num_folds=5, max_generations=1, ga_pop_size=3, processors=1)
    #gs.run()



if __name__ == '__main__':
    conf_path, data_path, outdir = mastml.get_paths()
    mastml.check_paths(conf_path, data_path, outdir)
    utils.activate_logging(outdir, (conf_path, data_path, outdir))

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

