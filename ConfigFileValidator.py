from copy import deepcopy
from distutils.util import strtobool

class ConfigFileValidator:
    """
    Class to validate contents of user-specified MASTML input file and flag any errors
    """
    def __init__(self, configdict, configtemplate, logger):
        #print("ConfigFileValidator configdict:", configdict)
        self.configdict = configdict
        self.configtemplate = configtemplate
        self.logger = logger
        self.errors_present = False

    def run_config_validation(self):

        self._check_config_section_names()
        self.logger.info('Done checking section names')

        self._check_config_subsection_names()
        self.logger.info('Done checking subsection names')

        self._check_config_subsection_values()
        self.logger.info('Done checking type casting values and checking parameter keywords')

        self._check_config_heading_compatibility()
        self.logger.info('Done checking model and test_case names')

        self._check_model_type_consistency()
        self.logger.info('Done checking target feature names and model type consistency')

        if self.errors_present:
            raise Exception('Errors found in your .conf file, check log file for all errors')

    def _check_config_section_names(self):
        # Check if extra sections are in input file that shouldn't be
        for k in self.configdict.keys():
            if k not in self.configtemplate.keys():
                self.logger.error('Conf file missing section: %s' % str(k))
                self.errors_present = True

        # Check if any sections are missing from input file
        for k in self.configtemplate.keys():
            if k not in self.configdict.keys():
                self.logger.error('You are missing the section called %s in your input file. To correct this issue, add this section to your input file.' % str(k))
                self.errors_present = True

    def _check_config_subsection_names(self):
        for section in self.configtemplate.keys():
            depth = _get_dict_depth(self.configtemplate[section])
            if depth == 1:
                # Check that required subsections are present for each section in user's input file.
                for section_key in self.configtemplate[section].keys():
                    if section_key not in self.configdict[section].keys():
                        self.logger.error('Your input file is missing section key %s, which is part of the %s section. Please include this section key in your input file.' % (section_key, section))
                        self.errors_present = True
                # Check that user's input file does not have any extra section keys that would later throw errors
                for section_key in self.configdict[section].keys():
                    if section_key not in self.configtemplate[section].keys():
                        self.logger.error('Your input file contains an extra section key or misspelled section key: %s in the %s section. Please correct this section key in your input file.' % (section_key, section))
                        self.errors_present = True
            if depth == 2:
                # Check that all subsections in input file are appropriately named. Note that not all subsections in template need to be present in input file
                for subsection_key in self.configdict[section].keys():
                    if section == 'Data Setup':
                        if not (type(subsection_key) == str):
                            logging.error('Data Setup needs subsection keys to be strings')
                            self.errors_present = True
                    if section == 'Test Parameters':
                        if '_' in subsection_key:
                            if not (subsection_key.split('_')[0] in self.configtemplate[section]):
                                self.logger.error('Your input file contains an improper subsection key name: %s.' % subsection_key)
                                self.errors_present = True
                        else:
                            if not (subsection_key in self.configtemplate[section]):
                                self.logger.error('Your input file contains an improper subsection key name: %s.' % subsection_key)
                                self.errors_present = True
                    if section == 'Model Parameters':
                        if not (subsection_key in self.configtemplate[section]):
                            self.logger.error('Your input file contains an improper subsection key name: %s.' % subsection_key)
                            self.errors_present = True

                    for subsection_param in self.configdict[section][subsection_key]:
                        if section == 'Data Setup':
                            subsection_key = 'string'
                        if section == 'Test Parameters':
                            if '_' in subsection_key:
                                subsection_key = subsection_key.split('_')[0]
                        try:
                            if not (subsection_param in self.configtemplate[section][subsection_key]):
                                self.logger.error('Your input file contains an improper subsection parameter name: %s.' % subsection_param)
                                self.errors_present = True
                        except KeyError:
                            self.logger.error('Your input file contains an improper subsection key name: %s.' % subsection_key)
                            self.errors_present = True

            elif depth > 2:
                self.logger.error('Too many subsections in conf file')
                self.errors_present = True

    def _check_config_subsection_values(self):
        # First do some manual cleanup for values that can be string or list

        configdict = string_to_list(self.configdict)
        for section in configdict.keys():
            depth = _get_dict_depth(test_dict=configdict[section])
            if depth == 1:
                for section_key in configdict[section].keys():
                    if type(self.configtemplate[section][section_key]) is str:
                        if self.configtemplate[section][section_key] == 'string':
                            self.configdict[section][section_key] = str(self.configdict[section][section_key])
                        elif self.configtemplate[section][section_key] == 'bool':
                            self.configdict[section][section_key] = bool(strtobool(self.configdict[section][section_key]))
                        elif self.configtemplate[section][section_key] == 'integer':
                            self.configdict[section][section_key] = int(self.configdict[section][section_key])
                        elif self.configtemplate[section][section_key] == 'float':
                            self.configdict[section][section_key] = float(self.configdict[section][section_key])
                        else:
                            logging.info('Error: Unrecognized data type encountered in input file template')
                            sys.exit()
                    elif type(self.configtemplate[section][section_key]) is list:
                        if type(self.configdict[section][section_key]) is str:
                            if self.configdict[section][section_key] not in self.configtemplate[section][section_key]:
                                logging.info('Error: Your input file contains an incorrect parameter keyword: %s' % str(self.configdict[section][section_key]))
                                errors_present = bool(True)
                        if type(self.configdict[section][section_key]) is list:
                            for param_value in self.configdict[section][section_key]:
                                if section_key == 'test_cases':
                                    if '_' in param_value:
                                        param_value = param_value.split('_')[0]
                                    if param_value not in self.configtemplate[section][section_key]:
                                        logging.info('Error: Your input file contains an incorrect parameter keyword: %s' % param_value)
                                        errors_present = bool(True)
            if depth == 2:
                for subsection_key in self.configdict[section].keys():
                    for param_name in self.configdict[section][subsection_key].keys():
                        subsection_key_template = subsection_key
                        if section == 'Data Setup':
                            subsection_key_template = 'string'
                        elif section == 'Test Parameters':
                            if '_' in subsection_key:
                                subsection_key_template = subsection_key.split('_')[0]
                        if type(self.configtemplate[section][subsection_key_template][param_name]) is str:
                            if self.configtemplate[section][subsection_key_template][param_name] == 'string':
                                self.configdict[section][subsection_key][param_name] = str(self.configdict[section][subsection_key][param_name])
                            if self.configtemplate[section][subsection_key_template][param_name] == 'bool':
                                self.configdict[section][subsection_key][param_name] = bool(strtobool(self.configdict[section][subsection_key][param_name]))
                            if self.configtemplate[section][subsection_key_template][param_name] == 'integer':
                                self.configdict[section][subsection_key][param_name] = int(self.configdict[section][subsection_key][param_name])
                            if self.configtemplate[section][subsection_key_template][param_name] == 'float':
                                self.configdict[section][subsection_key][param_name] = float(self.configdict[section][subsection_key][param_name])
        
    def _check_config_heading_compatibility(self):
        """ Check that listed test_cases coincide with subsection names in Test_Parameters and
        Model_Parameters, and flag test cases that won't be run """
        cases = ['models', 'test_cases']
        params = ['Model Parameters', 'Test Parameters']
        for param, case in zip(params, cases):
            test_cases = self.configdict['Models and Tests to Run'][case]
            test_parameter_subsections = self.configdict[param].keys()
            tests_being_run = list()
            if type(test_cases) is list:
                for test_case in test_cases:
                    if test_case not in test_parameter_subsections:
                        self.logger.error('You have listed test case/model %s, which does not coincide with the corresponding subsection name in the Test Parameters/Model Parameters section. These two names need to be the same, and the keyword must be correct' % test_case)
                        self.errors_present = True
                    else:
                        tests_being_run.append(test_case)
            elif type(test_cases) is str:
                if test_cases not in test_parameter_subsections:
                    self.logger.error('You have listed test case/model %s, which does not coincide with the corresponding subsection name in the Test Parameters/Model Parameters section. These two names need to be the same, and the keyword must be correct' % test_cases)
                    self.errors_present = True
                else:
                    tests_being_run.append(test_cases)
            for test_case in test_parameter_subsections:
                if test_case not in tests_being_run:
                    self.logger.warn('You have specified the test/model %s, which is not listed in your test_cases/models section. It will be skipped.' % test_case)
                    self.errors_present = True

    def _check_model_type_consistency(self):
        regressors_present = classifiers_present = False
        for model in self.configdict['Models and Tests to Run']['models']:
            print(model)
            if 'regressor' in model: regressors_present = True
            if 'classifier' in model: classifiers_present = True
        if regressors_present and classifiers_present:
            raise Exception("Cannot have both regressors and classifiers in model collection")
        if not regressors_present and not classifiers_present:
            raise Exception("You must have either a classifier or a regressor model")

        #TODO is adding new fields to configdict a bad idea?
        self.configdict['General Setup']['is_classification'] = classifiers_present

def _get_dict_depth(test_dict, level=0):
    if not isinstance(test_dict, dict) or not test_dict:
        return level
    return max(_get_dict_depth(test_dict[k], level+1) for k in test_dict)

def string_to_list(configdict):
    for section_heading in configdict.keys():
        if section_heading == 'General Setup':
            if type(configdict[section_heading]['input_features']) is str:
                templist = list()
                templist.append(configdict[section_heading]['input_features'])
                configdict[section_heading]['input_features'] = templist
            if type(configdict[section_heading]['labeling_features']) is str:
                templist = list()
                templist.append(configdict[section_heading]['labeling_features'])
                configdict[section_heading]['labeling_features'] = templist
        if section_heading == 'Models and Tests to Run':
            if type(configdict[section_heading]['models']) is str:
                templist = list()
                templist.append(configdict[section_heading]['models'])
                configdict[section_heading]['models'] = templist
            if type(configdict[section_heading]['test_cases']) is str:
                templist = list()
                templist.append(configdict[section_heading]['test_cases'])
                configdict[section_heading]['test_cases'] = templist
    return configdict
