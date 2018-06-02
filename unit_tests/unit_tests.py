import unittest
import os
import sys
import shutil
import warnings
import logging
logging.basicConfig(filename='unit_tests.log')

# so we can find MASTML package
sys.path.append('../')

import MASTML, MASTMLInitializer
from ConfigFileValidator import ConfigFileValidator
from ConfigTemplate import configtemplate

# pipe program stdout into /dev/null so we can read 
# unittest prinout more easily
#sys.stdout = open(os.devnull, 'w')

class SmokeTest(unittest.TestCase):

    def setUp(self):
        self.folders = list()

    def test_full_run(self):
        configfile = 'full_run.conf'
        warnings.simplefilter('ignore')
        MASTML.MASTMLDriver(configfile=configfile).run_MASTML()
        self.folders.append('results/full_run')

    def test_basic_example(self):
        configfile = 'example_input.conf'
        warnings.simplefilter('ignore')
        MASTML.MASTMLDriver(configfile=configfile).run_MASTML()
        self.folders.append('results/example_results')

    def test_classifiers(self):
        configfile = 'classifiers.conf'
        warnings.simplefilter('ignore')
        MASTML.MASTMLDriver(configfile=configfile).run_MASTML()
        self.folders.append('results/classifiers')

    def test_example_classifier(self):
        configfile = 'example_classifier.conf'
        warnings.simplefilter('ignore')
        MASTML.MASTMLDriver(configfile=configfile).run_MASTML()
        self.folders.append('results/example_classifier')

    def tearDown(self):
        for f in self.folders:
            pass
            # uncomment for no leftover folders:
            #shutil.rmtree(f)

def leaves_of_dict(dictt):
    """ 
    Returns non-dict elements of dict recursively, as a generator
    """
    for key, val in dictt.items():
        if isinstance(val, dict):
            yield from leaves_of_dict(val)
        else:
            yield val

class ParseTemplate(unittest.TestCase):

    legal_types = ['string',
                   'bool',
                   'string_list',
                   'Auto',
                   'integer']

    def setUp(self):
        conf_folder  = '../MASTML_input_file_template/'
        conf_file = 'MASTML_input_file_template.conf'

        self.parser = MASTMLInitializer.ConfigFileParser(conf_file)
        self.config_dict = self.parser.get_config_dict(conf_folder)

        self.type_template = configtemplate
        #self.typed_config_dict


    def test_dict_is_not_empty(self):
        self.assertTrue(0 < len(self.config_dict.keys()))

    def test_config_dict_values_are_lists_or_strings(self):
        for leaf in leaves_of_dict(self.config_dict):
             # leaf is leaf in the dict way, not in the list way.
             # check if leaf is string or every member of leaf is string
             self.assertTrue(isinstance(leaf, str) or
                             all(isinstance(x, str) for x in leaf))
 
    def test_template_has_only_type_names_and_lists_of_strings(self):
        for leaf in leaves_of_dict(self.type_template):
            # should either be a list of strings (that lists off legal options
            # or a string that matching the ones in legal_types
            self.assertTrue(leaf in self.legal_types or
                            all(isinstance(x, str) for x in leaf))

    def _validate_conf_file(self, filename):
        configdict = MASTMLInitializer.ConfigFileParser(filename).get_config_dict(os.getcwd())
        conf_file_dict = ConfigFileValidator(configdict, logging).run_config_validation()
        return conf_file_dict
        
    def test_validate_valid_conf_file(self):
        self._validate_conf_file('example_input.conf')

    def test_validate_bad_sections_conf_file(self):
        self.assertRaises(KeyError, lambda: self._validate_conf_file('config_with_bad_sections.conf'))

    def test_validate_bad_subsections_conf_file(self):
        self.assertRaises(KeyError, lambda: self._validate_conf_file('config_with_bad_subsections.conf'))

    def test_validate_bad_values_conf_file(self):
        self.assertRaises(KeyError, lambda: self._validate_conf_file('config_with_bad_values.conf'))

    def test_validate_invalid_model_conf_file(self):
        self.assertRaises(KeyError, lambda: self._validate_conf_file('config_with_invalid_model.conf'))

    def tearDown(self):
        pass

class TestConfigKeys(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

if __name__=='__main__':
    unittest.main()
