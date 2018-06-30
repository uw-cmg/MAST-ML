import sys
import logging
import textwrap
import time
import os
from os.path import join

class BetweenFilter(object):
    """ inclusive on both sides """
    def __init__(self, min_level, max_level):
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, logRecord):
        return self.min_level <= logRecord.levelno <= self.max_level

def activate_logging(savepath, paths, logger_name='mastml', to_screen=True, to_file=True,
        ignore_debug=False):

    #formatter = logging.Formatter("%(filename)s : %(funcName)s %(message)s")
    time_formatter = logging.Formatter("[%(levelname)s] %(asctime)s : %(message)s")
    level_formatter = logging.Formatter("[%(levelname)s] %(message)s")

    rootLogger = logging.getLogger(logger_name)
    rootLogger.setLevel(logging.DEBUG)


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
        # send INFO and DEBUG (if not suprressed with `hide_debug`) to stdout
        lower_level = logging.INFO if ignore_debug else logging.DEBUG
        stdout_hdlr = logging.StreamHandler(sys.stdout)
        stdout_hdlr.setLevel(lower_level)
        stdout_hdlr.addFilter(BetweenFilter(lower_level, logging.INFO))
        stdout_hdlr.setFormatter(level_formatter)
        rootLogger.addHandler(stdout_hdlr)

        # send WARNING and above to stderr
        stderr_hdlr = logging.StreamHandler(sys.stderr)
        stderr_hdlr.setLevel(logging.WARNING)
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
