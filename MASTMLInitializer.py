__author__ = 'Ryan Jacobs'

from configobj import ConfigObj, ConfigObjError
import sys
import os
from sklearn.kernel_ridge import KernelRidge

class ConfigFileParser(object):
    """Class to read in and parse contents of config file
    """
    def __init__(self, configfile):
        self.configfile = configfile

    def get_config_dict(self):
        return self._parse_config_file()

    def _parse_config_dict(self):
        config_dict = self._parse_config_file()
        config_dict_parsed = {}
        for section_name, section_contents in config_dict.items():
            for subsection_name, subsection_contents in section_contents.items():
                config_dict_parsed[subsection_name]=subsection_contents
        return config_dict_parsed

    def _get_config_dict_depth(self, test_dict, level=0):
        if not isinstance(test_dict, dict) or not test_dict:
            return level
        return max(self._get_config_dict_depth(test_dict=test_dict[k], level=level+1) for k in test_dict)

    def _parse_config_file(self):
        cwd = os.getcwd()
        if os.path.exists(cwd+"/"+str(self.configfile)):
            try:
                config_dict = ConfigObj(self.configfile)
                return config_dict
            except(ConfigObjError, IOError):
                print('Could not read in input file %s') % str(self.configfile)
                sys.exit()
        else:
            raise OSError('The input file you specified, %s, does not exist in the path %s' % (str(self.configfile), str(cwd)))

class MASTMLWrapper(object):
    """Class that takes parameters from parsed config file and performs calls to appropriate MASTML methods
    """
    def __init__(self, configdict):
        self.configdict = configdict

    def process_config_keyword(self, keyword):
        keywordsetup = {}
        if not self.configdict[str(keyword)]:
            raise IOError('This dict does not contain the relevant key, %s' % str(keyword))
        for k, v in self.configdict[str(keyword)].items():
            keywordsetup[k] = v
        return keywordsetup

    def get_machinelearning_model(self, keyword):
        if keyword == "gkrr_model":
            model = KernelRidge(alpha=float(self.configdict['Model Parameters']['gkrr_model']['alpha']),
                                coef0=int(self.configdict['Model Parameters']['gkrr_model']['coef0']),
                                degree=int(self.configdict['Model Parameters']['gkrr_model']['degree']),
                                gamma=float(self.configdict['Model Parameters']['gkrr_model']['gamma']),
                                kernel=str(self.configdict['Model Parameters']['gkrr_model']['kernel']),
                                kernel_params=None)
            return model
