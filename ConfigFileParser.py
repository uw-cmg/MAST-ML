__author__ = 'Ryan Jacobs'

import configobj
from pprint import pprint

class ConfigFileParser(object):
    """Class to parse contents of config file
    """
    def __init__(self, configfile):
        self.configfile = configfile

    def get_parsed_config_dict(self):
        #return self._parse_config_dict()
        return self._parse_config_file()

    def _parse_config_dict(self):
        config_dict = self._parse_config_file()
        config_dict_parsed = {}
        for section_name, section_contents in config_dict.items():
            for subsection_name, subsection_contents in section_contents.items():
                config_dict_parsed[subsection_name]=subsection_contents
        return config_dict_parsed

    def _parse_config_file(self):
        config_dict = configobj.ConfigObj(self.configfile)
        return config_dict

class MASTMLWrapper(ConfigFileParser):
    """Class that takes parameters from parsed config file and performs calls to appropriate MASTML methods
    """

    def __init__(self, configfile):
        super(MASTMLWrapper, self).__init__(configfile)

    def process_config(self):
        config_dict = self.get_parsed_config_dict()
        pprint(config_dict)
        # Here, would do calls to MASTML rountines, like GKRR model, based on keywords in config file
        return None





