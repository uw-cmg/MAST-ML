import configparser
import importlib
from warnings import warn

__author__ = 'haotian'


def parse(filename='default.conf'):
    warn('"configuration_parser" is now deprecated', DeprecationWarning)
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(filename)
    return config
