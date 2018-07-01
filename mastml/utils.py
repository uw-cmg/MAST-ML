import sys
import logging
import textwrap
import time
import os
from os.path import join
from collections import defaultdict

class BetweenFilter(object):
    """ inclusive on both sides """
    def __init__(self, min_level, max_level):
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, logRecord):
        return self.min_level <= logRecord.levelno <= self.max_level

def activate_logging(savepath, paths, logger_name='mastml', to_screen=True, to_file=True,
        verbosity = 0):
    """ 
    savepath is a dir for where to save log
    paths list of 3 strings we show so the user knows what is being ran
    verbosity: argument for what is shown on screen (does not affect files):
        0 shows everything
        -1 hides debug
        -2 hides info (so no stdout except print)
        -3 hides warning
        -4 hides error
        -5 hides all output
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

def to_upper(message):
    return str(message).upper()

def to_leet(message):
    conv = {'a': '4', 'b': '8', 'e': '3', 'l': '1', 'o': '0', 's': '5', 't': '7'}
    message = ''.join(conv[c] if c in conv else c for c in str(message))
    return message.upper()

from emoji import emojize
def emojify(message):
    words = {'INFO': ':notebook:', 'WARNING': ':warning:', 'ERROR': ':x:', 'CRITICAL': ':fire:'}
    conv = {'1': 'one', '2':':two:', '3':':three:', '4': ':four:'}
    message = str(message)
    for word, emoji in words.items():
        message = message.replace(word, emoji)
    message = ''.join(conv[c] if c in conv else c for c in str(message))
    message = emojize(message)
    return message.upper()

def verbosalize_logger(log, verbosity):
    if verbosity <= 0:
        return

    old_log = log._log

    def new_log(level, msg, *args, **kwargs):
        old_log(level, [None, to_upper, to_leet, emojify][verbosity](msg), *args, **kwargs)

    log._log = new_log
    return



    # no longer needed but so beautiful it needs to end up in the commit
    levels = ['debug', 'info', 'warning', 'error', 'critical']
    converter = [None, to_upper, to_leet, emojify][level]

    for level in levels:
        def apply_new_printer(old_printer):
            def new_printer(message, *args, **kwargs):
                message = converter(message)
                old_printer(message, *args, **kwargs)
            setattr(log, level, new_printer)

        old_printer = getattr(log, level)
        apply_new_printer(old_printer)

class MastError(Exception):
    """ base class for errors that should be shown to the user """
    pass

class ConfError(MastError):
    """ error in conf file """
    pass

class MissingColumnError(MastError):
    """ raised when your csv doesn't have the column you asked for """
    pass

class InvalidConfParameters(MastError):
    """ invalid conf params """
    pass

class InvalidConfSubSection(MastError):
    """ invalid section name """
    pass
class InvalidConfSection(MastError):
    """ invalid section name """
    pass

class FiletypeError(MastError):
    """ for using the wrong file extentions """
    pass

class FileNotFoundError(MastError): # sorry for re-using builtin name
    pass

class InvalidValue(MastError):
    pass
