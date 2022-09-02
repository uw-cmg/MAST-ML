from pathos.multiprocessing import ProcessingPool as Pool
from scipy.optimize import minimize
from functools import partial
from tqdm import tqdm

import numpy as np
import sys
import os


def parallel(func, x, *args, **kwargs):
    '''
    Run some function in parallel.
    '''

    cpu_count = os.cpu_count()
    part_func = partial(func, *args, **kwargs)
    with Pool(cpu_count) as pool:
        data = list(tqdm(
                         pool.imap(part_func, x),
                         total=len(x),
                         file=sys.stdout
                         ))

    return data


def llh(std, res, x, func):
    '''
    Compute the log likelihood.
    '''

    total = np.log(2*np.pi)
    total += np.log(func(x, std)**2)
    total += (res**2)/(func(x, std)**2)
    total *= -0.5

    return total


def set_llh(std, y, y_pred, x, func):
    '''
    Compute the log likelihood for a dataset.
    '''

    res = y-y_pred

    # Get negative to use minimization instead of maximization of llh
    opt = minimize(
                   lambda x: -sum(llh(std, res, x, func))/len(res),
                   x,
                   method='nelder-mead',
                   )

    params = opt.x

    return params


def chunck(x, n, mode='points'):
    '''
    Devide x data into n sized bins.
    '''

    if mode == 'points':
        size = len(x)
        n = size//n

    for i in range(0, len(x), n):

        x_new = x[i:i+n]
        if len(x_new) == n:
            yield x_new


# Polynomial given coefficients
def poly(c, std):
    total = 0.0
    for i in range(len(c)):
        total += c[i]*std**i
    return abs(total)
