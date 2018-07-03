import sys
import logging
import textwrap
import time
import os
import random
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

class MastError(Exception):
    """ base class for errors that should be shown to the user """
    pass

class ConfError(MastError):
    """ error in conf file """
    pass

class InvalidModel(MastError):
    """ Model does not exist """
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


#### DO NOT READ BELOW THIS LINE ####

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

    if verbosity >= 7:
        while True:
            log.critical('MSATML'*random.randint(3,2**(verbosity-5)))

    old_log = log._log

    def new_log(level, msg, *args, **kwargs):
        old_log(level, [None, to_upper, to_full_width, to_leet, deep_fry, deep_fry_2, emojify][verbosity](msg), *args, **kwargs)

    log._log = new_log
