import sys
import logging

class BetweenFilter(object):
    """ inclusive on both sides """
    def __init__(self, min_level, max_level):
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, logRecord):
        return self.min_level <= logRecord.levelno <= self.max_level

def activate_logging(logger_name='mastml', to_screen=True, to_file=True):

    #formatter = logging.Formatter("%(filename)s : %(funcName)s %(message)s")
    formatter = logging.Formatter("%(asctime)s : %(message)s")

    rootLogger = logging.getLogger(logger_name)
    rootLogger.setLevel(logging.DEBUG)


    if to_file:
        log_hdlr = logging.StreamHandler(open('log.log', 'a'))
        log_hdlr.setLevel(logging.DEBUG)
        log_hdlr.addFilter(BetweenFilter(logging.DEBUG, logging.INFO))
        log_hdlr.setFormatter(formatter)
        rootLogger.addHandler(log_hdlr)

        errors_hdlr = logging.StreamHandler(open('errors.log', 'a'))
        errors_hdlr.setLevel(logging.WARNING)
        errors_hdlr.setFormatter(formatter)
        rootLogger.addHandler(errors_hdlr)

    log_header() # only shows up in files

    if to_screen:
        stdout_hdlr = logging.StreamHandler(sys.stdout)
        stdout_hdlr.setLevel(logging.DEBUG)
        stdout_hdlr.addFilter(BetweenFilter(logging.DEBUG, logging.INFO))
        rootLogger.addHandler(stdout_hdlr)

        stderr_hdlr = logging.StreamHandler(sys.stderr)
        stderr_hdlr.setLevel(logging.WARNING)
        rootLogger.addHandler(stderr_hdlr)


def log_header():

    logo = textwrap.dedent(f"""\
           __  ___     __________    __  _____ 
          /  |/  /__ _/ __/_  __/___/  |/  / / 
         / /|_/ / _ `/\ \  / / /___/ /|_/ / /__
        /_/  /_/\_,_/___/ /_/     /_/  /_/____/
    """)

    date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    header = (f"\n\n{logo}\n\nMAST-ML run on {date_time} using \n"
              f"conf file: {os.path.basename(args.conf_path)}\n"
              f"csv file:  {os.path.basename(args.data_path)}\n"
              f"saving to: {os.path.basename(args.outdir)}\n\n")

    # using highest logging level so it shows up in ALL log files
    log.critical(header)
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
