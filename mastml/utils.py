import sys
import logging

def activate_logging(to_screen=True, to_file=True):

    rootLogger = logging.getLogger()
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
        rootLogger.addHandler(log_hdlr)

        errors_hdlr = logging.StreamHandler(open('errors.log', 'a'))
        errors_hdlr.setLevel(logging.WARNING)
        rootLogger.addHandler(errors_hdlr)


class MastError(Exception):
    """ base class for errors that should be shown to the user """
    pass

class ConfError(MastError):
    """ error in conf file """
    pass
