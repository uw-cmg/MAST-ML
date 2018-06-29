" use: python3 -m masmtl.search.search settings.conf data.csv -o results/"

import argparse

from .grid_search import GridSearch
from .genetic_search import GeneticSearch
from .data_handler import DataHandler
from ..legos import model_finder
from .. import mastml, utils

def parse_conf_file(filepath):
    "Accepts the filepath of a conf file and returns its parsed dictionary"
    conf = ConfigObj(filepath)
    # TODO




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

