__author__ = 'Ryan Jacobs, Tam Mayeshiba'

from configobj import ConfigObj, ConfigObjError
from validate import Validator, VdtTypeError
import sys
import os
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
import importlib
import logging

class ConfigFileParser(object):
    """Class to read in and parse contents of config file
    """
    def __init__(self, configfile):
        self.configfile = configfile

    def get_config_dict(self, path_to_file):
        return self._parse_config_file(path_to_file=path_to_file)

    def _get_config_dict_depth(self, test_dict, level=0):
        if not isinstance(test_dict, dict) or not test_dict:
            return level
        return max(self._get_config_dict_depth(test_dict=test_dict[k], level=level+1) for k in test_dict)

    def _parse_config_file(self, path_to_file):
        if not path_to_file:
            path_to_file = os.getcwd()

        if os.path.exists(path_to_file+"/"+str(self.configfile)):
            try:
                config_dict = ConfigObj(self.configfile)
                return config_dict
            except(ConfigObjError, IOError):
                print('Could not read in input file %s') % str(self.configfile)
                sys.exit()
        else:
            raise OSError('The input file you specified, %s, does not exist in the path %s' % (str(self.configfile), str(path_to_file)))

class ConfigFileValidator(ConfigFileParser):
    """Class to validate contents of user-specified MASTML input file and flag any errors
    """
    def __init__(self, configfile):
        super().__init__(configfile)

    def run_config_validation(self):
        errors_present = False
        validator = self._generate_validator()
        configdict = self.get_config_dict(path_to_file=os.getcwd())
        config_files_path = configdict['General Setup']['config_files_path']
        validationdict_names = ConfigFileParser(configfile='mastmlinputvalidationnames.conf').get_config_dict(path_to_file=config_files_path)
        validationdict_types = ConfigFileParser(configfile='mastmlinputvalidationtypes.conf').get_config_dict(path_to_file=config_files_path)
        #validationdict_names = ConfigFileParser(configfile='mastmlinputvalidationnames.conf').get_config_dict(path_to_file=None)
        #validationdict_types = ConfigFileParser(configfile='mastmlinputvalidationtypes.conf').get_config_dict(path_to_file=None)
        logging.info('MASTML is checking that the section names of your input file are valid...')
        configdict, errors_present = self._check_config_headings(configdict=configdict, validationdict=validationdict_names,
                                                                 validator=validator, errors_present=errors_present)
        self._check_for_errors(errors_present=errors_present)
        section_headings = [k for k in validationdict_names.keys()]

        logging.info('MASTML is converting the datatypes of values in your input file...')
        for section_heading in section_headings:
            configdict, errors_present = self._check_section_datatypes(configdict=configdict, validationdict=validationdict_types,
                                                                       validator=validator, errors_present=errors_present,
                                                                       section_heading=section_heading)
            self._check_for_errors(errors_present=errors_present)

        logging.info('MASTML is checking that the subsection names and values in your input file are valid...')
        for section_heading in section_headings:
            errors_present = self._check_section_names(configdict=configdict, validationdict=validationdict_names,
                                                       errors_present=errors_present, section_heading=section_heading)
            self._check_for_errors(errors_present=errors_present)


        return configdict, errors_present

    def _check_config_headings(self, configdict, validationdict, validator, errors_present):
        for k in validationdict.keys():
            if k not in configdict.keys():
                logging.info('You are missing the %s section in your input file' % str(k))
                errors_present = bool(True)

        return configdict, errors_present

    def _check_section_datatypes(self, configdict, validationdict, validator, errors_present, section_heading):
        # First do some manual cleanup for values that can be string or string_list, because of issue with configobj
        if section_heading == 'General Setup':
            if type(configdict['General Setup']['input_features']) is str:
                templist = []
                templist.append(configdict['General Setup']['input_features'])
                configdict['General Setup']['input_features'] = templist

        if section_heading == 'Models and Tests to Run':
            if type(configdict['Models and Tests to Run']['models']) is str:
                templist = []
                templist.append(configdict['Models and Tests to Run']['models'])
                configdict['Models and Tests to Run']['models'] = templist
            if type(configdict['Models and Tests to Run']['test_cases']) is str:
                templist = []
                templist.append(configdict['Models and Tests to Run']['test_cases'])
                configdict['Models and Tests to Run']['test_cases'] = templist

        # Check the data type of section and subsection headings and values
        configdict_depth = self._get_config_dict_depth(test_dict=configdict[section_heading])
        datatypes = ['string', 'integer', 'float', 'boolean', 'string_list', 'int_list', 'float_list']
        if section_heading in ['General Setup', 'Data Setup', 'Models and Tests to Run', 'Model Parameters']:
            for k in configdict[section_heading].keys():
                if configdict_depth == 1:
                    try:
                        datatype = validationdict[section_heading][k]
                        if datatype in datatypes:
                            configdict[section_heading][k] = validator.check(check=datatype, value=configdict[section_heading][k])
                    except VdtTypeError:
                        logging.info('The parameter %s in your %s section did not successfully convert to %s' % (k, section_heading, datatype))
                        errors_present = bool(True)

                if configdict_depth > 1:
                    for kk in configdict[section_heading][k].keys():
                        try:
                            if k in validationdict[section_heading]:
                                datatype = validationdict[section_heading][k][kk]
                                if datatype in datatypes:
                                    configdict[section_heading][k][kk] = validator.check(check=datatype, value=configdict[section_heading][k][kk])
                        except(VdtTypeError):
                            logging.info('The parameter %s in your %s : %s section did not successfully convert to string' % (section_heading, k, kk))
                            errors_present = bool(True)

        return configdict, errors_present

    def _check_section_names(self, configdict, validationdict, errors_present, section_heading):
        # Check that required section or subsections are present in user's input file.
        configdict_depth = self._get_config_dict_depth(test_dict=configdict[section_heading])
        if section_heading in ['General Setup', 'Data Setup', 'Models and Tests to Run']:
            for k in validationdict[section_heading].keys():
                if k not in configdict[section_heading].keys():
                    logging.info('The %s section of your input file has an input parameter entered incorrectly: %s' % (section_heading, k))
                    errors_present = bool(True)
                if k in ['models', 'test_cases']:
                    for case in configdict[section_heading][k]:
                        if case not in validationdict[section_heading][k]:
                            logging.info('The %s : %s section of your input file has an unknown input parameter %s. Trying base name in front of underscores.' % (section_heading, k, case))
                            case_base = case.split("_")[0] #Allow permuatations of the same test, like SingleFit_myfitA and SingleFit_myfitB
                            if case_base not in validationdict[section_heading][k]:
                                logging.info('The %s : %s section of your input file has an input parameter entered incorrectly: %s' % (section_heading, k, case))
                                errors_present = bool(True)
                if configdict_depth > 1:
                    for kk in validationdict[section_heading][k].keys():
                        if kk not in configdict[section_heading][k].keys():
                            logging.info('The %s section of your input file has an input parameter entered incorrectly: %s : %s' % (section_heading, k, kk))
                            errors_present = bool(True)
        return errors_present

    def _check_for_errors(self, errors_present):
        if errors_present == bool(True):
            logging.info('Errors have been detected in your MASTML setup. Please correct the errors and re-run MASTML')
            sys.exit()

    def _generate_validator(self):
        return Validator()

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

    # This method returns relevant model object based on input file. Fitting the model is performed later
    def get_machinelearning_model(self, model_type):
        if model_type == 'linear_model':
            model = LinearRegression(fit_intercept=bool(self.configdict['Model Parameters']['linear_model']['fit_intercept']))
            return model
        if model_type == 'lkrr_model':
            model = KernelRidge(alpha = float(self.configdict['Model Parameters']['gkrr_model']['alpha']),
                               gamma = float(self.configdict['Model Parameters']['gkrr_model']['gamma']),
                               kernel = str(self.configdict['Model Parameters']['gkrr_model']['kernel']))
            return model
        if model_type == 'gkrr_model':
            model = KernelRidge(alpha=float(self.configdict['Model Parameters']['gkrr_model']['alpha']),
                                coef0=int(self.configdict['Model Parameters']['gkrr_model']['coef0']),
                                degree=int(self.configdict['Model Parameters']['gkrr_model']['degree']),
                                gamma=float(self.configdict['Model Parameters']['gkrr_model']['gamma']),
                                kernel=str(self.configdict['Model Parameters']['gkrr_model']['kernel']),
                                kernel_params=None)
            return model
        if model_type == 'decision_tree_model':
            model = DecisionTreeRegressor(criterion=str(self.configdict['Model Parameters']['decision_tree_model']['criterion']),
                                               splitter=str(self.configdict['Model Parameters']['decision_tree_model']['splitter']),
                                               max_depth=int(self.configdict['Model Parameters']['decision_tree_model']['max_depth']),
                                               min_samples_leaf=int(self.configdict['Model Parameters']['decision_tree_model']['min_samples_leaf']),
                                               min_samples_split=int(self.configdict['Model Parameters']['decision_tree_model']['min_samples_split']))
            return model
        if model_type == 'extra_tree_model':
            model = ExtraTreesRegressor(criterion=str(self.configdict['Model Parameters']['extra_tree_model']['criterion']),
                                               n_estimators=str(self.configdict['Model Parameters']['extra_tree_model']['n_estimators']),
                                               max_depth=int(self.configdict['Model Parameters']['extra_tree_model']['max_depth']),
                                               min_samples_leaf=int(self.configdict['Model Parameters']['extra_tree_model']['min_samples_leaf']),
                                               min_samples_split=int(self.configdict['Model Parameters']['extra_tree_model']['min_samples_split']))
            return model
        if model_type == 'randomforest_model':
            model = RandomForestRegressor(criterion=str(self.configdict['Model Parameters']['randomforest_model']['criterion']),
                                          n_estimators=int(self.configdict['Model Parameters']['randomforest_model']['n_estimators']),
                                          max_depth=int(self.configdict['Model Parameters']['randomforest_model']['max_depth']),
                                          min_samples_split=int(self.configdict['Model Parameters']['randomforest_model']['min_samples_split']),
                                          min_samples_leaf=int(self.configdict['Model Parameters']['randomforest_model']['min_samples_leaf']),
                                          max_leaf_nodes=int(self.configdict['Model Parameters']['randomforest_model']['max_leaf_nodes']),
                                          n_jobs=int(self.configdict['Model Parameters']['randomforest_model']['n_jobs']),
                                          warm_start=bool(self.configdict['Model Parameters']['randomforest_model']['warm_start']))
            return model
        if model_type == 'nn_model':
            model = MLPRegressor(hidden_layer_sizes=int(self.configdict['Model Parameters']['nn_model']['hidden_layer_sizes']),
                                 activation=str(self.configdict['Model Parameters']['nn_model']['activation']),
                                 solver=str(self.configdict['Model Parameters']['nn_model']['solver']),
                                 alpha=float(self.configdict['Model Parameters']['nn_model']['alpha']),
                                 batch_size='auto',
                                 learning_rate='constant',
                                 max_iter=int(self.configdict['Model Parameters']['nn_model']['max_iter']),
                                 tol=float(self.configdict['Model Parameters']['nn_model']['tol']))
            return model
        else:
            raise TypeError('You have specified an invalid model_type name in your input file')

    # This method will call the different classes corresponding to each test type, which are being organized by Tam
    def get_machinelearning_test(self, test_type, model, save_path, *args, **kwargs):
        mod_name = test_type.split("_")[0] #ex. KFoldCV_5fold goes to KFoldCV
        test_module = importlib.import_module('%s' % (mod_name))
        test_class_def = getattr(test_module, mod_name)
        logging.debug("Parameters passed by keyword:")
        logging.debug(kwargs)
        test_class = test_class_def(model=model,
                            save_path = save_path,
                            **kwargs)
        test_class.run()
        return None
