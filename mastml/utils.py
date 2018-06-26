import sys
import logging


def activate_logging(logger_name='mastml', to_screen=True, to_file=True):


    #formatter = logging.Formatter("%(filename)s : %(funcName)s %(message)s")
    formatter = logging.Formatter("%(asctime)s : %(message)s")

    rootLogger = logging.getLogger(logger_name)
    rootLogger.setLevel(logging.DEBUG)

    if to_screen:
        stdout_hdlr = logging.StreamHandler(sys.stdout)
        stdout_hdlr.setLevel(logging.DEBUG)
        rootLogger.addHandler(stdout_hdlr)

        stderr_hdlr = logging.StreamHandler(sys.stderr)
        stderr_hdlr.setLevel(logging.WARNING)
        rootLogger.addHandler(stderr_hdlr)

    if to_file:
        log_hdlr = logging.StreamHandler(open('log.log', 'a'))
        log_hdlr.setLevel(logging.DEBUG)
        log_hdlr.setFormatter(formatter)
        rootLogger.addHandler(log_hdlr)

        errors_hdlr = logging.StreamHandler(open('errors.log', 'a'))
        errors_hdlr.setLevel(logging.WARNING)
        errors_hdlr.setFormatter(formatter)
        rootLogger.addHandler(errors_hdlr)


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
