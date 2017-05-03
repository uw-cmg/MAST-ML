__author__ = 'Ryan Jacobs, Tam Mayeshiba'

from configobj import ConfigObj, ConfigObjError
from validate import Validator, VdtTypeError
import sys
import os
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import sklearn.tree as tree
import neurolab as nl
import importlib
import logging

class ConfigFileParser(object):
    """Class to read in and parse contents of config file
    """
    def __init__(self, configfile):
        self.configfile = configfile

    def get_config_dict(self):
        return self._parse_config_file()

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

class ConfigFileValidator(ConfigFileParser):
    """Class to validate contents of user-specified MASTML input file and flag any errors
    """
    def __init__(self, configfile, validationfile):
        super(ConfigFileValidator, self).__init__(self)
        self.configfile = configfile
        self.validationfile = validationfile

    def run_config_validation(self):
        errors_present = False
        validator = self._generate_validator()
        configdict = self.get_config_dict()
        validationdict = ConfigFileParser(configfile=self.validationfile).get_config_dict()
        configdict, errors_present = self._check_config_headings(configdict=configdict, validationdict=validationdict,
                                                                 validator=validator, errors_present=errors_present)
        self._check_for_errors(errors_present=errors_present)
        configdict, errors_present = self._check_general_setup(configdict=configdict, validationdict=validationdict,
                                                               validator=validator, errors_present=errors_present)
        self._check_for_errors(errors_present=errors_present)
        configdict, errors_present = self._check_data_setup(configdict=configdict, validationdict=validationdict,
                                                            validator=validator, errors_present=errors_present)
        self._check_for_errors(errors_present=errors_present)
        configdict, errors_present = self._check_models_and_tests_to_run(configdict=configdict, validationdict=validationdict,
                                                            validator=validator, errors_present=errors_present)

        """
        try:
            configdict['models'] = validator.check(check='string', value=configdict['models'])
        except(VdtTypeError):
            configdict['models'] = validator.check(check='int_list', value=configdict['models'])
        print(configdict['models'])
        print(type(configdict['models'][0]))
        """

        return configdict, errors_present

    def _check_config_headings(self, configdict, validationdict, validator, errors_present):
        try:
            for k in configdict.keys():
                k = validator.check(check='string', value=k)
        except(VdtTypeError):
            logging.info('The section %s in your input file did not successfully import as a string' % str(k))
            errors_present = bool(True)

        for k in validationdict.keys():
            if k not in configdict.keys():
                logging.info('You are missing the %s section in your input file' % str(k))
                errors_present = bool(True)

        return configdict, errors_present

    def _check_general_setup(self, configdict, validationdict, validator, errors_present):
        try:
            for k, v in configdict['General Setup'].items():
                if k == 'save_path' or k == 'target_feature':
                    k = validator.check(check='string', value=k)
                    configdict['General Setup'][k] = validator.check(check='string', value=configdict['General Setup'][k])
                if k == 'input_features':
                    try:
                        k = validator.check(check='string', value=k)
                    except(VdtTypeError):
                        k = validator.check(check='string_list', value=k)
        except(VdtTypeError):
            logging.info('The parameter %s in your General Setup section did not successfully import as a string' % str(k))
            errors_present = bool(True)

        for k in validationdict['General Setup'].keys():
            if k not in configdict['General Setup'].keys():
                logging.info('The General Setup section of your input file is missing the input parameter: %s' % str(k))
                errors_present = bool(True)

        return configdict, errors_present

    def _check_data_setup(self, configdict, validationdict, validator, errors_present):
        try:
            for k, v in configdict['Data Setup'].items():
                k = validator.check(check='string', value=k)
        except(VdtTypeError):
            logging.info('The parameter %s in your Data Setup section did not successfully import as a string' % str(k))
            errors_present = bool(True)

        try:
            for k, v in configdict['Data Setup'].items():
                for kk, vv in v.items():
                    configdict['Data Setup'][k][kk] = validator.check(check='string', value=configdict['Data Setup'][k][kk])
        except(VdtTypeError):
            logging.info('The parameter %s in your Data Setup section did not successfully import as a string' % str(kk))
            errors_present = bool(True)

        for k in validationdict['Data Setup'].keys():
            if k == 'Initial' and k not in configdict['Data Setup'].keys():
                logging.info('The Data Setup section of your input file is missing the input parameter: %s' % str(k))
                errors_present = bool(True)

        return configdict, errors_present

    def _check_models_and_tests_to_run(self, configdict, validationdict, validator, errors_present):
        try:
            for k, v in configdict['Models and Tests to Run'].items():
                k = validator.check(check='string', value=k)
                if type(v) is str:
                    v = validator.check(check='string', value=v)
                elif type(v) is list:
                    v = validator.check(check='string_list', value=v)
                else:
                    logging.log('The parameter %s = %s in your Models and Tests to Run section is an invalid data type.'
                                'Supported types are string (one entry) or list of strings (multiple entries)' % str(k), str(v))
        except(VdtTypeError):
            logging.info('The parameter %s = %s in your Models and Tests to Run section did not successfully import as a string' % str(k), str(v))
            errors_present = bool(True)



        return configdict, errors_present

    def _check_test_parameters(self, configdict, validator):
        pass

    def _check_model_parameters(self, configdict, validator):
        pass

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
            model = tree.DecisionTreeRegressor(criterion=str(self.configdict['Model Parameters']['decision_tree_model']['split_criterion']),
                                               splitter=str(self.configdict['Model Parameters']['extra_tree_model']['splitter']),
                                               max_depth=int(self.configdict['Model Parameters']['decision_tree_model']['max_depth']),
                                               min_samples_leaf=int(self.configdict['Model Parameters']['decision_tree_model']['min_samples_leaf']),
                                               min_samples_split=int(self.configdict['Model Parameters']['decision_tree_model']['min_samples_split']))
        if model_type == 'extra_tree_model':
            model = tree.ExtraTreeRegressor(criterion=str(self.configdict['Model Parameters']['extra_tree_model']['split_criterion']),
                                               splitter=str(self.configdict['Model Parameters']['extra_tree_model']['splitter']),
                                               max_depth=int(self.configdict['Model Parameters']['extra_tree_model']['max_depth']),
                                               min_samples_leaf=int(self.configdict['Model Parameters']['extra_tree_model']['min_samples_leaf']),
                                               min_samples_split=int(self.configdict['Model Parameters']['extra_tree_model']['min_samples_split']))
            return model
        if model_type == 'randomforest_model':
            model = RandomForestRegressor(criterion=str(self.configdict['Model Parameters']['randomforest_model']['split_criterion']),
                                          n_estimators=int(self.configdict['Model Parameters']['randomforest_model']['estimators']),
                                          max_depth=int(self.configdict['Model Parameters']['randomforest_model']['max_depth']),
                                          min_samples_split=int(self.configdict['Model Parameters']['randomforest_model']['min_samples_split']),
                                          min_samples_leaf=int(self.configdict['Model Parameters']['randomforest_model']['min_samples_leaf']),
                                          max_leaf_nodes=int(self.configdict['Model Parameters']['randomforest_model']['max_leaf_nodes']),
                                          n_jobs=int(self.configdict['Model Parameters']['randomforest_model']['jobs']))
            return model
        if model_type == 'nn_model_neurolab':
            model = nl.net.newff(minmax=int(self.configdict['Model Parameters']['nn_model_neurolab']['minmax']),
                                 size=int(self.configdict['Model Parameters']['nn_model_neurolab']['size']),
                                 transf=str(self.configdict['Model Parameters']['nn_model_neurolab']['transfer_function']))
            train = str(self.configdict['Model Parameters']['nn_model_neurolab']['training_method'])
            epochs = int(self.configdict['Model Parameters']['nn_model_neurolab']['epochs'])
            show = bool(self.configdict['Model Parameters']['nn_model_neurolab']['show'])
            goal = float(self.configdict['Model Parameters']['nn_model_neurolab']['goal'])
            return (model, train, epochs, show, goal)
        if model_type == 'nn_model_sklearn':
            pass
        if model_type == 'nn_model_tensorflow':
            pass
        else:
            raise TypeError('You have specified an invalid model_type name in your input file')

        # Add generic file import for non-sklearn models

    # This method will call the different classes corresponding to each test type, which are being organized by Tam
    def get_machinelearning_test(self, test_type, model, data, save_path, *args, **kwargs):
        mod_name = test_type.split("_")[0] #ex. KFoldCV_5fold goes to KFoldCV
        test_module = importlib.import_module('%s' % (mod_name))
        test_execute = getattr(test_module, 'execute')
        test_execute(model=model, data=data, savepath=save_path, **kwargs)
        return None