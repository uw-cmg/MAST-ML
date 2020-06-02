"""
The utils module contains a collection of miscellaneous methods and error handling used throughout MAST-ML
"""
import numpy as np
import sys
import logging
import textwrap
import time
import os
import random
from os.path import join
from collections import defaultdict
from math import log, floor, ceil

class BetweenFilter(object):
    """
    Class to aid in handling logger display levels

    Args:

         min_level: (int), minimum verbosity level

         max_level: (int), maximum verbosity level

    Methods:

        filter: Method to return logging level of logging.logRecord object

            Args:

                logRecord: (python logging.logRecord object)

            Returns:

                (int) logging level of logging.logRecord object, which is between the min and max provided levels
    """
    def __init__(self, min_level, max_level):
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, logRecord):
        return self.min_level <= logRecord.levelno <= self.max_level

def activate_logging(savepath, paths, logger_name='mastml', to_screen=True, to_file=True, verbosity = 0):
    """
    Method to create MAST-ML logger file

    Args:

        savepath: (str), string specifying the save path

        paths: (list), list containing strings of path locations for config file, data file, and results folder

        logger_name: (str), name of logger file

        to_screen: (bool), whether or not to write the log contents to the screen during a run

        to_file: (bool), whether or not to write the log contents to a file in the savepath

        verbosity: (int), controls the amount of output produced in the logger. Accepted values:

                0 shows everything

                -1 hides debug

                -2 hides info (so no stdout except print)

                -3 hides warning

                -4 hides error

                -5 hides all output

    Returns:

        None

    """
    #formatter = logging.Formatter("%(filename)s : %(funcName)s %(message)s")
    time_formatter = logging.Formatter("[%(levelname)s] %(asctime)s : %(message)s")
    level_formatter = logging.Formatter("[%(levelname)s] %(message)s")

    rootLogger = logging.getLogger(logger_name)
    rootLogger.setLevel(logging.DEBUG)

    verbosalize_logger(rootLogger, verbosity)

    if to_file:
        # send everything to log.log
        log_hdlr = logging.StreamHandler(open(join(savepath, 'log.log'), 'a'))
        log_hdlr.setLevel(logging.DEBUG)
        log_hdlr.setFormatter(time_formatter)
        rootLogger.addHandler(log_hdlr)

        # send WARNING and above to errors.log
        errors_hdlr = logging.StreamHandler(open(join(savepath, 'errors.log'), 'a'))
        errors_hdlr.setLevel(logging.WARNING)
        errors_hdlr.setFormatter(time_formatter)
        rootLogger.addHandler(errors_hdlr)

    if to_screen:
        # send INFO and DEBUG (if not suprressed) to stdout
        lower_level = logging.INFO if verbosity >= 0 else logging.DEBUG
        if verbosity >= 0:
            lower_level = logging.DEBUG # DEBUG and INFO
        elif verbosity == -1:
            lower_level = logging.INFO # only INFO
        else:
            lower_level = logging.WARNING # effectively disables stdout

        stdout_hdlr = logging.StreamHandler(sys.stdout)
        stdout_hdlr.setLevel(lower_level)
        stdout_hdlr.addFilter(BetweenFilter(lower_level, logging.INFO))
        stdout_hdlr.setFormatter(level_formatter)
        rootLogger.addHandler(stdout_hdlr)

        # send WARNING and above to stderr,
        # verbosity of -3 sets WARNING, -4 sets  ERROR, and -5 sets CRITICAL
        lower_level = max(-10*verbosity + 10, logging.WARNING)
        stderr_hdlr = logging.StreamHandler(sys.stderr)
        stderr_hdlr.setLevel(lower_level)
        stderr_hdlr.setFormatter(level_formatter)
        rootLogger.addHandler(stderr_hdlr)

    log_header(paths, rootLogger) # only shows up in files

def log_header(paths, log):
    """
    Method to create header for MAST-ML logger

    Args:

        paths: (list), list containing strings of path locations for config file, data file, and results folder

        log: (logging object), a python log

    Returns:

        None

    """
    logo = textwrap.dedent(f"""\
           __  ___     __________    __  _____
          /  |/  /__ _/ __/_  __/___/  |/  / /
         / /|_/ / _ `/\ \  / / /___/ /|_/ / /__
        /_/  /_/\_,_/___/ /_/     /_/  /_/____/
    """)

    date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    header = (f"\n\n{logo}\n\nMAST-ML run on {date_time} using \n"
              f"conf file: {os.path.basename(paths[0])}\n"
              f"csv file:  {os.path.basename(paths[1])}\n"
              f"saving to: {os.path.basename(paths[2])}\n\n")

    # only shows on stdout and log.log
    log.info(header)

## Custom errors for mastml:

class MastError(Exception):
    """
    Base class for MAST-ML specific errors that should be shown to the user
    """
    pass

class ConfError(MastError):
    """
    Class representing error in input configuration file
    """
    pass

class InvalidModel(MastError):
    """
    Class representing error when model does not exist
    """
    pass

class MissingColumnError(MastError):
    """
    Class representing error raised when your csv doesn't have the specified column
    """
    pass

class InvalidConfParameters(MastError):
    """
    Class representing error raised when you have invalid input configuration file parameters
    """
    pass

class InvalidConfSubSection(MastError):
    """
    Class representing error raised when an invalid subsection name is present in the input configuration file
    """
    pass

class InvalidConfSection(MastError):
    """
    Class representing error raised when an invalid section name is present in the input configuration file
    """
    pass

class FiletypeError(MastError):
    """
    Class representing error raised when an improper file extension is used
    """
    pass

class FileNotFoundError(MastError): # sorry for re-using builtin name
    """
    Class representing error raised when a needed file cannot be found
    """
    pass

class InvalidValue(MastError):
    """
    Class representing error raised when an invalid value has been used
    """
    pass


## Math utilities to aid plot_helper to make ranges
#TODO: can likely remove from here as in plot_helper now
def nice_range(lower, upper):
    """
    Method to create a range of values, including the specified start and end points, with nicely spaced intervals

    Args:

        lower: (float or int), lower bound of range to create

        upper: (float or int), upper bound of range to create

    Returns:

        (list), list of numerical values in established range

    """

    flipped = 1 # set to -1 for inverted

    # Case for validation where nan is passed in
    if np.isnan(lower):
        lower = 0
    if np.isnan(upper):
        upper = 0.1

    if upper < lower:
        upper, lower = lower, upper
        flipped = -1
    return [_int_if_int(x) for x in _nice_range_helper(lower, upper)][::flipped]

def _nice_range_helper(lower, upper):
    """
    Method to help make a better range of axis ticks

    Args:

        lower: (float), lower value of axis ticks

        upper: (float), upper value of axis ticks

    Returns:

        upper: (float), modified upper tick value fixed based on set of axis ticks

    """
    steps = 8
    diff = abs(lower - upper)

    # special case where lower and upper are the same
    if diff == 0:
        return [lower,]

    # the exact step needed
    step = diff / steps

    # a rough estimate of best step
    step = _nearest_pow_ten(step) # whole decimal increments

    # tune in one the best step size
    factors = [0.1, 0.2, 0.5, 1, 2, 5, 10]

    # use this to minimize how far we are from ideal step size
    def best_one(steps_factor):
        steps_count, factor = steps_factor
        return abs(steps_count - steps)
    n_steps, best_factor = min([(diff / (step * f), f) for f in factors], key = best_one)

    #print('should see n steps', ceil(n_steps + 2))
    # multiply in the optimal factor for getting as close to ten steps as we can
    step = step * best_factor

    # make the bounds look nice
    lower = _three_sigfigs(lower)
    upper = _three_sigfigs(upper)

    start = _round_up(lower, step)

    # prepare for iteration
    x = start # pointless init
    i = 0

    # itereate until we reach upper
    while x < upper - step:
        x = start + i * step
        yield _three_sigfigs(x) # using sigfigs because of floating point error
        i += 1

    # finish off with ending bound
    yield upper

def _three_sigfigs(x):
    """
    Method invoking special case of _n_sigfigs to return 3 sig figs

    Args:

        x: (float), an axis tick number

    Returns:

        (float), number of sig figs (always 3)

    """
    return _n_sigfigs(x, 3)

def _n_sigfigs(x, n):
    """
    Method to return number of sig figs to use for axis ticks

    Args:

        x: (float), an axis tick number

    Returns:

        (float), number of sig figs

    """
    sign = 1
    if x == 0:
        return 0
    if x < 0: # case for negatives
        x = -x
        sign = -1
    if x < 1:
        base = n - round(log(x, 10))
    else:
        base = (n-1) - round(log(x, 10))
    return sign * round(x, base)

def _nearest_pow_ten(x):
    """
    Method to return the nearest power of ten for an axis tick value

    Args:

        x: (float), an axis tick number

    Returns:

        (float), nearest power of ten of x

    """
    sign = 1
    if x == 0:
        return 0
    if x < 0: # case for negatives
        x = -x
        sign = -1
    return sign*10**ceil(log(x, 10))

def _int_if_int(x):
    """
    Method to return integer mapped value of x

    Args:

        x: (float or int), a number

    Returns:

        x: (float), value of x mapped as integer

    """
    if int(x) == x:
        return int(x)
    return x

def _round_up(x, inc):
    """
    Method to round up the value of x

    Args:

        x: (float or int), a number

        inc: (float), an increment for axis ticks

    Returns:

        (float), value of x rounded up

    """
    sign = 1
    if x < 0: # case for negative
        x = -x
        sign = -1

    return sign * inc * ceil(x / inc)

def verbosalize_logger(log, verbosity):
    if verbosity <= 0:
        return

    if verbosity >= 8:
        while True:
            log.critical('MASTML'*random.randint(3,2**(verbosity-4)))

    #old_log = log._log

    #def new_log(level, msg, *args, **kwargs):
    #    old_log(level, [None, None, to_upper, to_full_width, to_leet, deep_fry, deep_fry_2, emojify][verbosity](msg), *args, **kwargs)

    #log._log = new_log

## Joke functions:
"""
### String formatting funcs for inserting into log._log when in verbose mode
def to_upper(message):
    return str(message).upper()

def to_full_width(message):
    message = str(message)
    ret = []
    return ''.join(chr(ord(c)+0xFEE0) if c.isalnum() else c for c in message)

def to_leet(message):
    conv = {'a': '4', 'b': '8', 'e': '3', 'l': '1', 'o': '0', 's': '5', 't': '7'}
    message = ''.join(conv[c] if c in conv else c for c in str(message))
    return message.upper()

def deep_fry_helper(s):
    for c in s:
        if random.random() < 0.2:
            yield chr(random.randint(1, 10000))
        elif random.random() < 0.5:
            yield c.upper()
        else:
            yield c

def deep_fry(message):
    return ''.join(deep_fry_helper(str(message)))

def deep_fry_2_helper(s):
    for c in s:
        if random.random() < 0.01:
            x = chr(random.randint(1, 10000))
            yield ''.join(x for _ in range(random.randint(1,5)))
        elif random.random() < 0.1:
            yield ''.join(c for _ in range(random.randint(1,9)))
        elif random.random() < 0.01:
            yield chr(ord(c)*10)
        elif random.random() < 0.2:
            yield c.upper()
        else:
            yield c

def deep_fry_2(message):
    return ''.join(deep_fry_2_helper(str(message)))

def emojify(message):
    message = str(message)
    words = {'score':0x1f600, 'splits': chr(0x21A9)+chr(0x21AA), 'split': chr(0x21A9)+chr(0x21AA), 'score':0x2728, 'number':0x0023,
             'train':0x1f682, 'test':0x1f4dd, 'models':0x1F483, 'model':0x1F483, 'plot':0x1f4ca, 'image':0x1f5bc,
             'file': 0x1f5c4, 'files':0x1f5c3, ' to':0x27a1, '1':0x261d, '2':0x270c}
    for word in words:
        if type(words[word]) is int:
            words[word] = chr(words[word])
        words[word] += ' '
    for word, emoji in words.items():
        message = message.replace(' '+word+' ', ' '+emoji+' ')
    for word, emoji in words.items():
        message = message.replace(word, emoji)
    return message.upper()

def verbosalize_logger(log, verbosity):
    if verbosity <= 0:
        return

    if verbosity >= 8:
        while True:
            log.critical('MSATML'*random.randint(3,2**(verbosity-4)))

    old_log = log._log

    def new_log(level, msg, *args, **kwargs):
        old_log(level, [None, None, to_upper, to_full_width, to_leet, deep_fry, deep_fry_2, emojify][verbosity](msg), *args, **kwargs)

    log._log = new_log
"""
