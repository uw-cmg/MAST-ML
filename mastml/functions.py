from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
from tqdm import tqdm

import sys
import os


def parallel(func, x, *args, **kwargs):
    '''
    Run some function in parallel.

    inputs:
        func = The function to apply.
        x = The list of items to apply function on.

    outputs:
        data = List of items returned by func.
    '''

    pool = Pool(os.cpu_count())
    part_func = partial(func, *args, **kwargs)

    with Pool(os.cpu_count()) as pool:
        data = list(tqdm(
                         pool.imap(part_func, x),
                         total=len(x),
                         file=sys.stdout)
                         )

    return data
