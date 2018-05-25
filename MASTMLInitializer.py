__author__ = 'Ryan Jacobs, Tam Mayeshiba'
__maintainer__ = 'Ryan Jacobs'
__version__ = '1.0'
__email__ = 'rjacobs3@wisc.edu'
__date__ = 'October 14th, 2017'

import sys
import os
import importlib
import logging
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR
import sklearn.gaussian_process.kernels as skkernel
from sklearn.gaussian_process import GaussianProcessRegressor
from configobj import ConfigObj, ConfigObjError
import distutils.util as du

class ConfigFileParser(object):
    """
    Class to read in contents of MASTML input files

    Args:
        configfile (MASTML configfile object) : a MASTML input file, as a configfile object

    Methods:
        get_config_dict : returns dict representation of configfile

            Args:
                path_to_file (str) : path indicating where config file is stored

            Returns:
                dict : configdict of parsed input file
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
        if not os.path.exists(path_to_file):
            logging.info('You must specify a valid path')
            sys.exit()
        if os.path.exists(path_to_file+"/"+str(self.configfile)):
            original_dir = os.getcwd()
            os.chdir(path_to_file)
            try:
                config_dict = ConfigObj(self.configfile)
                os.chdir(original_dir)
                return config_dict
            except(ConfigObjError, IOError):
                logging.info('Could not read in input file %s') % str(self.configfile)
                sys.exit()
        else:
            raise OSError('The input file you specified, %s, does not exist in the path %s' % (str(self.configfile), str(path_to_file)))

class ConfigFileConstructor(ConfigFileParser):
    """
    Constructor class to build input file template. Used for input file validation and data type casting.

    Args:
        configfile (MASTML configfile object) : a MASTML input file, as a configfile object

    Methods:
        get_config_template : returns template of input file for validation

            Returns:
                dict : configdict of template input file
    """
    def __init__(self, configfile):
        super().__init__(configfile=configfile)
        self.configtemplate = dict()

    def get_config_template(self):

        self.configtemplate['General Setup'] = {
            'save_path': 'string',
            'input_features': ['string', 'string_list', 'Auto'],
            'target_feature': 'string',
            'grouping_feature': 'string',
            'labeling_features': ['string'],
        }

        self.configtemplate['Data Setup'] = {
            'string': {'data_path': 'string'},
        }

        self.configtemplate['Feature Normalization'] = {
            'normalize_x_features': 'bool',
            'normalize_y_feature': 'bool',
            'feature_normalization_type': ['standardize', 'normalize'],
            'feature_scale_min': 'float',
            'feature_scale_max': 'float',
        }

        self.configtemplate['Feature Generation'] = {
            'perform_feature_generation': 'bool',
            'add_magpie_features': 'bool',
            'add_materialsproject_features': 'bool',
            'add_citrine_features': 'bool',
            'materialsproject_apikey': 'string',
            'citrine_apikey': 'string',
        }

        self.configtemplate['Feature Selection'] = {
            'perform_feature_selection': 'bool',
            'remove_constant_features': 'bool',
            'feature_selection_algorithm': [
                'univariate_feature_selection',
                'recursive_feature_elimination',
                'sequential_forward_selection',
                'basic_forward_selection',
                ],
            'use_mutual_information':  'bool',
            'number_of_features_to_keep':  'integer',
            'scoring_metric':  [
                'mean_squared_error',
                'mean_absolute_error',
                'root_mean_squared_error',
                'r2_score',
            ],
            'generate_feature_learning_curve': 'bool',
            'model_to_use_for_learning_curve': [
                'linear_model_regressor',
                'linear_model_lasso_regressor',
                'lkrr_model_regressor',
                'gkrr_model_regressor',
                'support_vector_machine_model_regressor',
                'decision_tree_model_regressor',
                'extra_trees_model_regressor',
                'randomforest_model_regressor',
                'adaboost_model_regressor',
                'nn_model_regressor',
                'gaussianprocess_model_regressor',
            ],
        }

        self.configtemplate['Models and Tests to Run'] = {
            'models': [
                'linear_model_regressor',
                'linear_model_lasso_regressor',
                'lkrr_model_regressor',
                'gkrr_model_regressor',
                'dummy_model',
                'support_vector_machine_model_regressor',
                'decision_tree_model_regressor',
                'extra_trees_model_regressor',
                'randomforest_model_regressor',
                'adaboost_model_regressor',
                'nn_model_regressor',
                'gaussianprocess_model_regressor',
            ],
            'test_cases': [
                'SingleFit',
                'SingleFitPerGroup',
                'SingleFitGrouped',
                'KFoldCV',
                'LeaveOneOutCV',
                'LeaveOutPercentCV',
                'LeaveOutGroupCV',
                'PredictionVsFeature',
                'PlotNoAnalysis',
                'ParamGridSearch',
                'ParamOptGA',
            ],
        }

        self.configtemplate['Test Parameters'] = dict()
        for test_case in self.configtemplate['Models and Tests to Run']['test_cases']:
            self.configtemplate['Test Parameters'][test_case] = dict()
        for key, param in self.configtemplate['Test Parameters'].items():
            if key in ['SingleFit', 'SingleFitPerGroup', 'SingleFitGrouped',
                     'KFoldCV', 'LeaveOneOutCV', 'LeaveOutPercentCV', 'LeaveOutGroupCV',
                     'PlotNoAnalysis', 'PredictionVsFeature']:
                param['training_dataset'] = 'string'
                param['testing_dataset'] = 'string'
                param['xlabel'] = 'string'
                param['ylabel'] = 'string'
            if key in ['PlotNoAnalysis', 'PredictionVsFeature']:
                param['feature_plot_feature'] = 'string'
                param['plot_filter_out'] = 'string'
                param['data_labels'] = 'string'
            if key == 'SingleFit':
                param['plot_filter_out'] = 'string'
            if key == 'SingleFitPerGroup':
                param['plot_filter_out'] = 'string'
            if key == 'SingleFitGrouped':
                param['plot_filter_out'] = 'string'
            if key in ['KFoldCV', 'LeaveOneOutCV', 'LeaveOutPercentCV', 'LeaveOutGroupCV']:
                param['num_cvtests'] = 'integer'
                param['mark_outlying_points'] = 'integer'
            if key == 'KFoldCV':
                param['num_folds'] = 'integer'
            if key == 'LeaveOutPercentCV':
                param['percent_leave_out'] = 'float'
            if key in ['ParamOptGA', 'ParamGridSearch']:
                param['training_dataset'] = 'string'
                param['testing_dataset'] = 'string'
                param['num_folds'] = 'integer'
                param['num_cvtests'] = 'integer'
                param['percent_leave_out'] = 'float'
                param['pop_upper_limit'] = 'integer'
                for i in range(5):
                    param['param_%s' % str(i+1)] = 'string'

        self.configtemplate['Model Parameters'] = dict()
        models = [k for k in self.configtemplate['Models and Tests to Run']['models']]
        for model in models:
            self.configtemplate['Model Parameters'][model] = dict()
        for key, param in self.configtemplate['Model Parameters'].items():
            if key in ['linear_model_regressor', 'linear_model_lasso_regressor']:
                param['fit_intercept'] = 'bool'
            if key == 'linear_model_lasso_regressor':
                param['alpha'] = 'float'
            if key == 'gkrr_model_regressor':
                param['alpha'] = 'float'
                param['gamma'] = 'float'
                param['coef0'] = 'float'
                param['degree'] = 'integer'
                param['kernel'] = ['linear', 'cosine', 'polynomial', 'sigmoid', 'rbf', 'laplacian']
            if key == 'lkrr_model_regressor': #TODO
                param['alpha'] = 'float'
                param['gamma'] = 'float'
                param['coef0'] = 'float'
                param['degree'] = 'integer'
                param['kernel'] = ['linear', 'cosine', 'polynomial', 'sigmoid', 'rbf', 'laplacian']
            if key == 'support_vector_machine_model_regressor':
                param['error_penalty'] = 'float'
                param['gamma'] = 'float'
                param['coef0'] = 'float'
                param['degree'] = 'integer'
                param['kernel'] = ['linear', 'cosine', 'polynomial', 'sigmoid', 'rbf', 'laplacian']
            if key == 'decision_tree_model_regressor':
                param['criterion'] = ['mae', 'mse', 'friedman_mse']
                param['splitter'] = ['random', 'best']
                param['max_depth'] = 'integer'
                param['min_samples_leaf'] = 'integer'
                param['min_samples_split'] = 'integer'
            if key in ['extra_trees_model_regressor', 'randomforest_model_regressor']:
                param['criterion'] = ['mse', 'mae']
                param['n_estimators'] = 'integer'
                param['max_depth'] = 'integer'
                param['min_samples_leaf'] = 'integer'
                param['min_samples_split'] = 'integer'
                param['max_leaf_nodes'] = 'integer'
            if key == 'randomforest_model_regressor':
                param['n_jobs'] = 'integer'
                param['warm_start'] = 'bool'
            if key == 'adaboost_model_regressor':
                param['base_estimator_max_depth'] = 'integer'
                param['n_estimators'] = 'integer'
                param['learning_rate'] = 'float'
                param['loss'] = ['linear' 'square', 'exponential']
            if key == 'nn_model_regressor':
                param['hidden_layer_sizes'] = 'tuple'
                param['activation'] = ['identity', 'logistic', 'tanh', 'relu']
                param['solver'] = ['lbfgs', 'sgd', 'adam']
                param['alpha'] = 'float'
                param['max_iterations'] = 'integer'
                param['tolerance'] = 'float'
            if key == 'gaussianprocess_model_regressor':
                param['kernel'] = ['rbf']
                param['RBF_length_scale'] = 'float'
                param['RBF_length_scale_bounds'] = 'tuple'
                param['alpha'] = 'float'
                param['optimizer'] = ['fmin_l_bfgs_b']
                param['n_restarts_optimizer'] = 'integer'
                param['normalize_y'] = 'bool'

        return self.configtemplate

class ConfigFileValidator(ConfigFileConstructor, ConfigFileParser):
    """
    Class to validate contents of user-specified MASTML input file and flag any errors

    Args:
        configfile (MASTML configfile object) : a MASTML input file, as a configfile object

    Methods:
        run_config_validation : checks configfile object for errors

            Returns:
                dict : configdict of parsed input file
                bool : whether any errors occurred parsing the input file
            Throws:
                MASTML.ConfigFileError : after everything has been checked, if ANY checks fail, raise 
                            exception with message to check log.
    """
    def __init__(self, configfile):
        super().__init__(configfile)
        self.get_config_template()
        self.errors_present = False

    def run_config_validation(self):
        configdict = self.get_config_dict(path_to_file=os.getcwd())

        # Check section names
        logging.info('MASTML is checking that the section names of your input file are valid...')
        configdict = self._check_config_section_names(configdict=configdict)
         
        logging.info('MASTML input file section names are valid')

        # Check subsection names
        logging.info('MASTML is checking that the subsection names and values in your input file are valid...')
        self._check_config_subsection_names(configdict=configdict)
        logging.info('MASTML input file subsection names are valid')

        logging.info('MASTML is checking your subsection parameter values and converting the datatypes of values in your input file...')
        configdict = self._check_config_subsection_values(configdict=configdict)
        logging.info('MASTML subsection parameter values converted to correct datatypes, and parameter keywords are valid')

        logging.info('MASTML is cross-checking model and test_case names in your input file...')
        configdict = self._check_config_heading_compatibility(configdict=configdict)
        logging.info('MASTML model and test_case names are valid')

        logging.info('MASTML is checking that your target feature name is formatted correctly...')
        configdict = self._check_target_feature(configdict=configdict)
        logging.info('MASTML target feature name is valid')

        if self.errors_present:
            raise MASTML.ConfigFileError('Errors found in your .conf file, check log file for all errors')

        return configdict

    def _check_config_section_names(self, configdict):
        # Check if extra sections are in input file that shouldn't be
        for k in configdict.keys():
            if k not in self.configtemplate.keys():
                logging.info('Error: You have an extra section called %s in your input file. To correct this issue, remove this extra section.' % str(k))
                self.errors_present = True

        # Check if any sections are missing from input file
        for k in self.configtemplate.keys():
            if k not in configdict.keys():
                logging.info('Error: You are missing the section called %s in your input file. To correct this issue, add this section to your input file.' % str(k))
                self.errors_present = True

        return configdict

    def _check_config_subsection_names(self, configdict):
        for section in self.configtemplate.keys():
            depth = self._get_config_dict_depth(test_dict=self.configtemplate[section])
            if depth == 1:
                # Check that required subsections are present for each section in user's input file.
                for section_key in self.configtemplate[section].keys():
                    if section_key not in configdict[section].keys():
                        logging.info('Error: Your input file is missing section key %s, which is part of the %s section. Please include this section key in your input file.' % (section_key, section))
                        self.errors_present = True
                # Check that user's input file does not have any extra section keys that would later throw errors
                for section_key in configdict[section].keys():
                    if section_key not in self.configtemplate[section].keys():
                        logging.info('Error: Your input file contains an extra section key or misspelled section key: %s in the %s section. Please correct this section key in your input file.' % (section_key, section))
                        self.errors_present = True
            if depth == 2:
                # Check that all subsections in input file are appropriately named. Note that not all subsections in template need to be present in input file
                for subsection_key in configdict[section].keys():
                    if section == 'Data Setup':
                        if not (type(subsection_key) == str):
                            self.errors_present = True
                    if section == 'Test Parameters':
                        if '_' in subsection_key:
                            if not (subsection_key.split('_')[0] in self.configtemplate[section]):
                                logging.info('Error: Your input file contains an improper subsection key name: %s.' % subsection_key)
                                self.errors_present = True
                        else:
                            if not (subsection_key in self.configtemplate[section]):
                                logging.info('Error: Your input file contains an improper subsection key name: %s.' % subsection_key)
                                self.errors_present = True
                    if section == 'Model Parameters':
                        if not (subsection_key in self.configtemplate[section]):
                            logging.info('Error: Your input file contains an improper subsection key name: %s.' % subsection_key)
                            self.errors_present = True

                    for subsection_param in configdict[section][subsection_key]:
                        if section == 'Data Setup':
                            subsection_key = 'string'
                        if section == 'Test Parameters':
                            if '_' in subsection_key:
                                subsection_key = subsection_key.split('_')[0]
                        try:
                            if not (subsection_param in self.configtemplate[section][subsection_key]):
                                logging.info('Error: Your input file contains an improper subsection parameter name: %s.' % subsection_param)
                                self.errors_present = True
                        except KeyError:
                            logging.info('Error: Your input file contains an improper subsection key name: %s.' % subsection_key)
                            self.errors_present = True

            elif depth > 2:
                logging.info('There is an error in your input file setup: too many subsections.')
                self.errors_present = True

    def _check_config_subsection_values(self, configdict):
        """
        Iterates recursively through `configdict` alongside `self.configtemplate`
        and typecasts each entry of `configdict` to the types contained in
        `self.configtemplate`. 
        """

        # First do some manual cleanup for values that can be string or list
        def string_to_list(configdict):
            for section_heading, subsect in configdict.items():
                if section_heading == 'General Setup':
                    if type(subsect['input_features']) is str:
                        subsect['input_features'] = [subsect['input_features'],]
                    if type(subsect['labeling_features']) is str:
                        subsect['labeling_features'] = [subsect['labeling_features'],]

                if section_heading == 'Models and Tests to Run':
                    if type(subsect['models']) is str:
                        subsect['models'] = [subsect['models'],]
                    if type(subsect['test_cases']) is str:
                        subsect['test_cases'] = [subsect['test_cases'],]

            return configdict

        configdict = string_to_list(configdict=configdict)
        for section in configdict.keys():
            depth = self._get_config_dict_depth(test_dict=configdict[section])
            if depth == 1:
                for section_key in configdict[section].keys():
                    if type(self.configtemplate[section][section_key]) is str:
                        if self.configtemplate[section][section_key] == 'string':
                            configdict[section][section_key] = str(configdict[section][section_key])
                        elif self.configtemplate[section][section_key] == 'bool':
                            configdict[section][section_key] = bool(du.strtobool(configdict[section][section_key]))
                        elif self.configtemplate[section][section_key] == 'integer':
                            configdict[section][section_key] = int(configdict[section][section_key])
                        elif self.configtemplate[section][section_key] == 'float':
                            configdict[section][section_key] = float(configdict[section][section_key])
                        else:
                            logging.info('Error: Unrecognized data type encountered in input file template')
                            sys.exit()
                    elif type(self.configtemplate[section][section_key]) is list:
                        if type(configdict[section][section_key]) is str:
                            if configdict[section][section_key] not in self.configtemplate[section][section_key]:
                                logging.info('Error: Your input file contains an incorrect parameter keyword: %s' % str(configdict[section][section_key]))
                                self.errors_present = bool(True)
                        if type(configdict[section][section_key]) is list:
                            for param_value in configdict[section][section_key]:
                                if section_key == 'test_cases':
                                    if '_' in param_value:
                                        param_value = param_value.split('_')[0]
                                    if param_value not in self.configtemplate[section][section_key]:
                                        logging.info('Error: Your input file contains an incorrect parameter keyword: %s' % param_value)
                                        self.errors_present = bool(True)

            if depth == 2:
                for subsection_key in configdict[section].keys():
                    for param_name in configdict[section][subsection_key].keys():
                        subsection_key_template = subsection_key
                        if section == 'Data Setup':
                            subsection_key_template = 'string'
                        elif section == 'Test Parameters':
                            if '_' in subsection_key:
                                subsection_key_template = subsection_key.split('_')[0]
                        if type(self.configtemplate[section][subsection_key_template][param_name]) is str:
                            if self.configtemplate[section][subsection_key_template][param_name] == 'string':
                                configdict[section][subsection_key][param_name] = str(configdict[section][subsection_key][param_name])
                            if self.configtemplate[section][subsection_key_template][param_name] == 'bool':
                                configdict[section][subsection_key][param_name] = bool(du.strtobool(configdict[section][subsection_key][param_name]))
                            if self.configtemplate[section][subsection_key_template][param_name] == 'integer':
                                configdict[section][subsection_key][param_name] = int(configdict[section][subsection_key][param_name])
                            if self.configtemplate[section][subsection_key_template][param_name] == 'float':
                                configdict[section][subsection_key][param_name] = float(configdict[section][subsection_key][param_name])
        return configdict

        
    def _check_config_heading_compatibility(self, configdict):
        # Check that listed test_cases coincide with subsection names in Test_Parameters and Model_Parameters, and flag test cases that won't be run
        cases = ['models', 'test_cases']
        params = ['Model Parameters', 'Test Parameters']
        for param, case in zip(params, cases):
            test_cases = configdict['Models and Tests to Run'][case]
            test_parameter_subsections = configdict[param].keys()
            tests_being_run = list()
            if type(test_cases) is list:
                for test_case in test_cases:
                    if test_case not in test_parameter_subsections:
                        logging.info('Error: You have listed test case/model %s, which does not coincide with the corresponding subsection name in the Test Parameters/Model Parameters section. These two names need to be the same, and the keyword must be correct' % test_case)
                        self.errors_present = True
                    else:
                        tests_being_run.append(test_case)
            elif type(test_cases) is str:
                if test_cases not in test_parameter_subsections:
                    logging.info('Error: You have listed test case/model %s, which does not coincide with the corresponding subsection name in the Test Parameters/Model Parameters section. These two names need to be the same, and the keyword must be correct' % test_cases)
                    self.errors_present = True
                else:
                    tests_being_run.append(test_cases)
            for test_case in test_parameter_subsections:
                if test_case not in tests_being_run:
                    logging.info('Note to user: you have specified the test/model %s, which is not listed in your test_cases/models section. MASTML will run fine, but this test will not be performed' % test_case)
        return configdict

    def _check_target_feature(self, configdict):
        target_feature_name = configdict['General Setup']['target_feature']
        if ('regression' or 'classification') not in target_feature_name:
            logging.info('Error: You must include the designation "regression" or "classification" in your target feature name in your input file and data file. For example: "my_target_feature_regression"')
            self.errors_present = True
        return configdict

class ModelTestConstructor(object):
    """
    Class that takes parameters from configdict (configfile as dict) and performs calls to appropriate MASTML methods

    Args:
        configfile (MASTML configfile object) : a MASTML input file, as a configfile object

    Methods:
        get_machinelearning_model : obtains machine learning model by calling sklearn

            Args:
                model_type (str) : keyword string indicating sklearn model name
                y_feature (str) : name of target feature

            Returns:
                sklearn model object : an sklearn model object for fitting to data

        get_machinelearning_test : obtains test name to conduct from configdict

            Args:
                test_type (str) : keyword string specifying type of MASTML test to perform
                model (sklearn model object) : sklearn model object to use in test_type
                save_path (str) : path of save directory to store test output

    """
    def __init__(self, configdict):
        self.configdict = configdict

    def get_machinelearning_model(self, model_type, y_feature):
        if 'classification' in y_feature:
            logging.info('Error: Currently, MASTML only supports regression models. Classification models and metrics are under development')
            sys.exit()
            if 'classifier' in model_type:
                logging.info('got y_feature %s' % y_feature)
                logging.info('model type is %s' % model_type)
                logging.info('doing classification on %s' % y_feature)
                if model_type == 'support_vector_machine_model_classifier':
                    d = self.configdict['Model Parameters']['support_vector_machine_model_classifier']
                    return SVC(C=float(d['error_penalty']),
                               kernel=str(d['kernel']),
                               degree=int(d['degree']),
                               gamma=float(d['gamma']),
                               coef0=float(d['coef0']))
                if model_type == 'logistic_regression_model_classifier':
                    d = self.configdict['Model Parameters']['logistic_regression_model_classifier']
                    return LogisticRegression(penalty=str(d['penalty']),
                                              C=float(d['C']),
                                              class_weight=str(d['class_weight']))
                if model_type == 'decision_tree_model_classifier':
                    d = self.configdict['Model Parameters']['decision_tree_model_classifier']
                    return DecisionTreeClassifier(criterion=str(d['criterion']),
                                                  splitter=str(d['splitter']),
                                                  max_depth=int(d['max_depth']),
                                                  min_samples_leaf=int(d['min_samples_leaf']),
                                                  min_samples_split=int(d['min_samples_split']))
                if model_type == 'random_forest_model_classifier':
                    d = self.configdict['Model Parameters']['random_forest_model_classifier']
                    return RandomForestClassifier(criterion=str(d['criterion']),
                                                  n_estimators=int(d['n_estimators']),
                                                  max_depth=int(d['max_depth']),
                                                  min_samples_split=int(d['min_samples_split']),
                                                  min_samples_leaf=int(d['min_samples_leaf']),
                                                  max_leaf_nodes=int(d['max_leaf_nodes']))
                if model_type == 'extra_trees_model_classifier':
                    d = self.configdict['Model Parameters']['extra_trees_model_classifier']
                    return ExtraTreesClassifier(criterion=str(d['criterion']),
                                                n_estimators=int(d['n_estimators']),
                                                max_depth=int(d['max_depth']),
                                                min_samples_split=int(d['min_samples_split']),
                                                min_samples_leaf=int(d['min_samples_leaf']),
                                                max_leaf_nodes=int(d['max_leaf_nodes']))
                if model_type == 'adaboost_model_classifier':
                    d = self.configdict['Model Parameters']['adaboost_model_classifier']
                    return AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=int(d['base_estimator_max_depth'])),
                                              n_estimators=int(d['n_estimators']),
                                              learning_rate=float(d['learning_rate']),
                                              random_state=None)
                if model_type == 'nn_model_classifier':
                    d = self.configdict['Model Parameters']['nn_model_classifier']
                    return MLPClassifier(hidden_layer_sizes=int(d['hidden_layer_sizes']),
                                         activation=str(d['activation']),
                                         solver=str(d['solver']),
                                         alpha=float(d['alpha']),
                                         batch_size='auto',
                                         learning_rate='constant',
                                         max_iter=int(d['max_iterations']),
                                         tol=float(d['tolerance']))

        if 'regression' in y_feature:
            if 'regressor' in model_type:
                logging.info('got y_feature %s' % y_feature)
                logging.info('model type %s' % model_type)
                logging.info('doing regression on %s' % y_feature)
                if model_type == 'linear_model_regressor':
                    model = LinearRegression(fit_intercept=self.configdict['Model Parameters']['linear_model_regressor']['fit_intercept'])
                    return model
                if model_type == 'linear_model_lasso_regressor':
                    model = Lasso(alpha=self.configdict['Model Parameters']['linear_model_lasso_regressor']['alpha'],
                                  fit_intercept=self.configdict['Model Parameters']['linear_model_lasso_regressor']['fit_intercept'])
                    return model
                if model_type == 'support_vector_machine_model_regressor':
                    d = self.configdict['Model Parameters']['support_vector_machine_model_regressor']
                    return SVR(C=d['error_penalty'],
                               kernel=d['kernel'],
                               degree=d['degree'],
                               gamma=d['gamma'],
                               coef0=d['coef0'])
                if model_type == 'lkrr_model_regressor':
                    d = self.configdict['Model Parameters']['lkrr_model_regressor']
                    return KernelRidge(alpha=d['alpha'],
                                       gamma=d['gamma'],
                                       kernel=d['kernel'])
                if model_type == 'gkrr_model_regressor':
                    d = self.configdict['Model Parameters']['gkrr_model_regressor']
                    return KernelRidge(alpha=d['alpha'],
                                       coef0=d['coef0'],
                                       degree=d['degree'],
                                       gamma=d['gamma'],
                                       kernel=d['kernel'],
                                       kernel_params=None)
                if model_type == 'decision_tree_model_regressor':
                    d = self.configdict['Model Parameters']['decision_tree_model_regressor']
                    return DecisionTreeRegressor(criterion=d['criterion'],
                                                 splitter=d['splitter'],
                                                 max_depth=d['max_depth'],
                                                 min_samples_leaf=d['min_samples_leaf'],
                                                 min_samples_split=d['min_samples_split'])
                if model_type == 'extra_trees_model_regressor':
                    d = self.configdict['Model Parameters']['extra_trees_model_regressor']
                    return ExtraTreesRegressor(criterion=d['criterion'],
                                               n_estimators=d['n_estimators'],
                                               max_depth=d['max_depth'],
                                               min_samples_leaf=d['min_samples_leaf'],
                                               min_samples_split=d['min_samples_split'],
                                               max_leaf_nodes=d['max_leaf_nodes'])
                if model_type == 'randomforest_model_regressor':
                    d = self.configdict['Model Parameters']['randomforest_model_regressor']
                    return RandomForestRegressor(criterion=d['criterion'],
                                                 n_estimators=d['n_estimators'],
                                                 max_depth=d['max_depth'],
                                                 min_samples_split=d['min_samples_split'],
                                                 min_samples_leaf=d['min_samples_leaf'],
                                                 max_leaf_nodes=d['max_leaf_nodes'],
                                                 n_jobs=d['n_jobs'],
                                                 warm_start=d['warm_start'],
                                                 bootstrap=True)
                if model_type == 'adaboost_model_regressor':
                    d = self.configdict['Model Parameters']['adaboost_model_regressor']
                    return AdaBoostRegressor(base_estimator=d['base_estimator_max_depth'],
                                             n_estimators=d['n_estimators'],
                                             learning_rate=d['learning_rate'],
                                             loss=d['loss'],
                                             random_state=None)
                if model_type == 'nn_model_regressor':
                    d = self.configdict['Model Parameters']['nn_model_regressor']
                    return MLPRegressor(hidden_layer_sizes=d['hidden_layer_sizes'],
                                        activation=d['activation'],
                                        solver=d['solver'],
                                        alpha=d['alpha'],
                                        batch_size='auto',
                                        learning_rate='constant',
                                        max_iter=d['max_iterations'],
                                        tol=d['tolerance'])
                if model_type == 'gaussianprocess_model_regressor':
                    d = self.configdict['Model Parameters']['gaussianprocess_model_regressor']
                    test_kernel = None
                    if d['kernel'] == 'rbf':
                        test_kernel = skkernel.ConstantKernel(1.0, (1e-5, 1e5)) * skkernel.RBF(length_scale=d['RBF_length_scale'],
                            length_scale_bounds=tuple(float(i) for i in d['RBF_length_scale_bounds']))
                    return GaussianProcessRegressor(kernel=test_kernel,
                                                    alpha=d['alpha'],
                                                    optimizer=d['optimizer'],
                                                    n_restarts_optimizer=d['n_restarts_optimizer'],
                                                    normalize_y=d['normalize_y'],
                                                    copy_X_train=True)  # bool(self.configdict['Model Parameters']['gaussianprocess_model']['copy_X_train']),
                                                    # int(self.configdict['Model Parameters']['gaussianprocess_model']['random_state']
                if model_type == 'dummy_model':
                    model = None # model doesn't do anything
                    return model
                else:
                    model = None # TODO: make this throw an error
                    return model

        elif model_type == 'custom_model':
            model_dict = self.configdict['Model Parameters']['custom_model']
            package_name = model_dict.pop('package_name') #return and remove
            class_name = model_dict.pop('class_name') #return and remove
            import importlib
            custom_module = importlib.import_module(package_name)
            module_class_def = getattr(custom_module, class_name) 
            model = module_class_def(**model_dict) #pass all the rest as kwargs
            return model
        elif model_type == 'load_model':
            model_dict = self.configdict['Model Parameters']['load_model']
            model_location = model_dict['location'] #pickle location
            from sklearn.externals import joblib
            model = joblib.load(model_location)
            return model
        #else:
        #    raise TypeError('You have specified an invalid model_type name in your input file')

    def get_machinelearning_test(self, test_type, model, save_path, run_test=True, *args, **kwargs):
        mod_name = test_type.split("_")[0] #ex. KFoldCV_5fold goes to KFoldCV
        test_module = importlib.import_module('%s' % (mod_name))
        test_class_def = getattr(test_module, mod_name)
        logging.debug("Parameters passed by keyword:")
        logging.debug(kwargs)
        test_class = test_class_def(model=model,
                            save_path = save_path,
                            **kwargs)
        if run_test == True:
            test_class.run()
        return test_class

    def _process_config_keyword(self, keyword):
        keywordsetup = {}
        if not self.configdict[str(keyword)]:
            raise IOError('This dict does not contain the relevant key, %s' % str(keyword))
        for k, v in self.configdict[str(keyword)].items():
            keywordsetup[k] = v
        return keywordsetup
