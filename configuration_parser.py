import configparser
import importlib

__author__ = 'haotian'


def parse(filename='default.conf'):
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(filename)
    return config
