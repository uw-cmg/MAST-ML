import unittest
import os
import sys
import shutil
import warnings
import pdb

# so we can find MASTML package
sys.path.append('../')

import MASTML
import MASTMLInitializer

# pipe program stdout into a log file so we can read 
# unittest prinout more easily
sys.stdout = open(os.devnull, 'w')

class SmokeTest(unittest.TestCase):

    def setUp(self):
        self.folders = list()

    def test_full_run(self):
        configfile = 'conf_files/full_run.conf'
        warnings.simplefilter('ignore')
        MASTML.MASTMLDriver(configfile=configfile).run_MASTML()
        self.folders.append('results/full_run')

    def test_basic_example(self):
        configfile = 'conf_files/example_input.conf'
        warnings.simplefilter('ignore')
        MASTML.MASTMLDriver(configfile=configfile).run_MASTML()
        self.folders.append('results/example_results')

    def test_classifiers(self):
        configfile = 'conf_files/classifiers.conf'
        warnings.simplefilter('ignore')
        MASTML.MASTMLDriver(configfile=configfile).run_MASTML()
        self.folders.append('results/classifiers')

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

        self.config_file_constructor = MASTMLInitializer.ConfigFileConstructor(conf_file)
        self.type_template = self.config_file_constructor.get_config_template()
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
            self.assertTrue(leaf in legal_types or
                            all(isinstance(x, str) for x in leaf))

    def _errors_present_from_conf_file(self, filename):
        file_validator = MASTMLInitializer.ConfigFileValidator(filename)
        _, errors_present = file_validator.run_config_validation()
        return errors_present
        
    def test_validate_valid_conf_file(self):
        self.assertFalse(self._errors_present_from_conf_file('example_input.conf'))

    def test_validate_bad_sections_conf_file(self):
        self.assertFalse(self._errors_present_from_conf_file('config_with_bad_sections.conf'))

    def test_validate_bad_subsections_conf_file(self):
        self.assertFalse(self._errors_present_from_conf_file('config_with_bad_subsections.conf'))

    def test_validate_bad_values_conf_file(self):
        self.assertFalse(self._errors_present_from_conf_file('config_with_bad_values.conf'))

    def test_validate_invalid_model_conf_file(self):
        self.assertFalse(self._errors_present_from_conf_file('config_with_invalid_model.conf'))

    def tearDown(self):
        pass

class TestConfigKeys(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

if __name__=='__main__':
    unittest.main()
