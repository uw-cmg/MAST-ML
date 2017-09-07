__author__ = 'Ryan Jacobs, Tam Mayeshiba'

import sys
import os
import importlib
import logging
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC
from configobj import ConfigObj, ConfigObjError
from validate import Validator, VdtTypeError

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
        if not os.path.exists(path_to_file):
            print('You must specify a valid path')
            sys.exit()
        if os.path.exists(path_to_file+"/"+str(self.configfile)):
            original_dir = os.getcwd()
            os.chdir(path_to_file)
            try:
                config_dict = ConfigObj(self.configfile)
                os.chdir(original_dir)
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
                            logging.info('The parameter %s in your %s : %s section did not successfully convert to %s' % (section_heading, k, kk, datatype))
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
    # Add feature_number_index here so that can pass list of values for model parameters, one value for each feature to fit.
    # Need to also pass y_feature list here so can discern between regression and classification tasks.
    def get_machinelearning_model(self, model_type, target_feature_regression_count, target_feature_classification_count, y_feature):
        if 'classification' in y_feature:
            if 'classifier' in model_type:
                print('got y_feature', y_feature)
                print('model type is', model_type)
                print('doing classification on', y_feature)
                if model_type == 'support_vector_machine_model_classifier':
                    if type(self.configdict['Model Parameters']['support_vector_machine_model_classifier']['error_penalty']) is list:
                        model = SVC(C=float(self.configdict['Model Parameters']['support_vector_machine_model_classifier']['error_penalty'][target_feature_regression_count]),
                                    kernel=str(self.configdict['Model Parameters']['support_vector_machine_model_classifier']['kernel'][target_feature_regression_count]),
                                    degree=int(self.configdict['Model Parameters']['support_vector_machine_model_classifier']['degree'][target_feature_regression_count]),
                                    gamma=float(self.configdict['Model Parameters']['support_vector_machine_model_classifier']['gamma'][target_feature_regression_count]),
                                    coef0=float(self.configdict['Model Parameters']['support_vector_machine_model_classifier']['coef0'][target_feature_regression_count]))
                    else:
                        model = SVC(C=float(self.configdict['Model Parameters']['support_vector_machine_model_classifier']['error_penalty']),
                                    kernel=str(self.configdict['Model Parameters']['support_vector_machine_model_classifier']['kernel']),
                                    degree=int(self.configdict['Model Parameters']['support_vector_machine_model_classifier']['degree']),
                                    gamma=float(self.configdict['Model Parameters']['support_vector_machine_model_classifier']['gamma']),
                                    coef0=float(self.configdict['Model Parameters']['support_vector_machine_model_classifier']['coef0']))
                    return model
                if model_type == 'logistic_regression_model_classifier':
                    if type(self.configdict['Model Parameters']['logistic_regression_model_classifier']['penalty']) is list:
                        model = LogisticRegression(penalty=str(self.configdict['Model Parameters']['logistic_regression_model_classifier']['penalty'][target_feature_classification_count]),
                                                   C=float(self.configdict['Model Parameters']['logistic_regression_model_classifier']['C'][target_feature_classification_count]),
                                                   class_weight=str(self.configdict['Model Parameters']['logistic_regression_model_classifier']['class_weight'][target_feature_classification_count]))
                    else:
                        model = LogisticRegression(penalty=str(self.configdict['Model Parameters']['logistic_regression_model_classifier']['penalty']),
                                                   C=float(self.configdict['Model Parameters']['logistic_regression_model_classifier']['C']),
                                                   class_weight=str(self.configdict['Model Parameters']['logistic_regression_model_classifier']['class_weight']))
                    return model
                if model_type == 'decision_tree_model_classifier':
                    if type(self.configdict['Model Parameters']['decision_tree_model_classifier']['criterion']) is list:
                        model = DecisionTreeClassifier(criterion=str(self.configdict['Model Parameters']['decision_tree_model_classifier']['criterion'][target_feature_classification_count]),
                                                       splitter=str(self.configdict['Model Parameters']['decision_tree_model_classifier']['splitter'][target_feature_classification_count]),
                                                       max_depth=int(self.configdict['Model Parameters']['decision_tree_model_classifier']['max_depth'][target_feature_classification_count]),
                                                       min_samples_leaf=int(self.configdict['Model Parameters']['decision_tree_model_classifier']['min_samples_leaf'][target_feature_classification_count]),
                                                       min_samples_split=int(self.configdict['Model Parameters']['decision_tree_model_classifier']['min_samples_split'][target_feature_classification_count]))
                    else:
                        model = DecisionTreeClassifier(criterion=str(self.configdict['Model Parameters']['decision_tree_model_classifier']['criterion']),
                                                           splitter=str(self.configdict['Model Parameters']['decision_tree_model_classifier']['splitter']),
                                                           max_depth=int(self.configdict['Model Parameters']['decision_tree_model_classifier']['max_depth']),
                                                           min_samples_leaf=int(self.configdict['Model Parameters']['decision_tree_model_classifier']['min_samples_leaf']),
                                                           min_samples_split=int(self.configdict['Model Parameters']['decision_tree_model_classifier']['min_samples_split']))
                    return model
                if model_type == 'random_forest_model_classifier':
                    if type(self.configdict['Model Parameters']['random_forest_model_classifier']['criterion']) is list:
                        model = RandomForestClassifier(criterion=str(self.configdict['Model Parameters']['random_forest_model_classifier']['criterion'][target_feature_classification_count]),
                                                       n_estimators=int(self.configdict['Model Parameters']['random_forest_model_classifier']['n_estimators'][target_feature_classification_count]),
                                                       max_depth=int(self.configdict['Model Parameters']['random_forest_model_classifier']['max_depth'][target_feature_classification_count]),
                                                       min_samples_split=int(self.configdict['Model Parameters']['random_forest_model_classifier']['min_samples_split'][target_feature_classification_count]),
                                                       min_samples_leaf=int(self.configdict['Model Parameters']['random_forest_model_classifier']['min_samples_leaf'][target_feature_classification_count]),
                                                       max_leaf_nodes=int(self.configdict['Model Parameters']['random_forest_model_classifier']['max_leaf_nodes'][target_feature_classification_count]))
                    else:
                        model = RandomForestClassifier(criterion=str(self.configdict['Model Parameters']['random_forest_model_classifier']['criterion']),
                                                       n_estimators=int(self.configdict['Model Parameters']['random_forest_model_classifier']['n_estimators']),
                                                       max_depth=int(self.configdict['Model Parameters']['random_forest_model_classifier']['max_depth']),
                                                       min_samples_split=int(self.configdict['Model Parameters']['random_forest_model_classifier']['min_samples_split']),
                                                       min_samples_leaf=int(self.configdict['Model Parameters']['random_forest_model_classifier']['min_samples_leaf']),
                                                       max_leaf_nodes=int(self.configdict['Model Parameters']['random_forest_model_classifier']['max_leaf_nodes']))
                    return model
                if model_type == 'extra_trees_model_classifier':
                    if type(self.configdict['Model Parameters']['extra_trees_model_classifier']['criterion']) is list:
                        model = ExtraTreesClassifier(criterion=str(self.configdict['Model Parameters']['extra_trees_model_classifier']['criterion'][target_feature_classification_count]),
                                                     n_estimators=int(self.configdict['Model Parameters']['extra_trees_model_classifier']['n_estimators'][target_feature_classification_count]),
                                                     max_depth=int(self.configdict['Model Parameters']['extra_trees_model_classifier']['max_depth'][target_feature_classification_count]),
                                                     min_samples_split=int(self.configdict['Model Parameters']['extra_trees_model_classifier']['min_samples_split'][target_feature_classification_count]),
                                                     min_samples_leaf=int(self.configdict['Model Parameters']['extra_trees_model_classifier']['min_samples_leaf'][target_feature_classification_count]),
                                                     max_leaf_nodes=int(self.configdict['Model Parameters']['extra_trees_model_classifier']['max_leaf_nodes'][target_feature_classification_count]))
                    else:
                        model = ExtraTreesClassifier(criterion=str(self.configdict['Model Parameters']['extra_trees_model_classifier']['criterion']),
                                                     n_estimators=int(self.configdict['Model Parameters']['extra_trees_model_classifier']['n_estimators']),
                                                     max_depth=int(self.configdict['Model Parameters']['extra_trees_model_classifier']['max_depth']),
                                                     min_samples_split=int(self.configdict['Model Parameters']['extra_trees_model_classifier']['min_samples_split']),
                                                     min_samples_leaf=int(self.configdict['Model Parameters']['extra_trees_model_classifier']['min_samples_leaf']),
                                                     max_leaf_nodes=int(self.configdict['Model Parameters']['extra_trees_model_classifier']['max_leaf_nodes']))
                    return model
                if model_type == 'adaboost_model_classifier':
                    if type(self.configdict['Model Parameters']['adaboost_model_classifier']['base_estimator_max_depth']) is list:
                        model = AdaBoostClassifier(base_estimator= DecisionTreeClassifier(max_depth=int(self.configdict['Model Parameters']['adaboost_model_classifier']['base_estimator_max_depth'][target_feature_classification_count])),
                                                  n_estimators=int(self.configdict['Model Parameters']['adaboost_model_classifier']['n_estimators'][target_feature_classification_count]),
                                                  learning_rate=float(self.configdict['Model Parameters']['adaboost_model_classifier']['learning_rate'][target_feature_classification_count]),
                                                  random_state=None)
                    else:
                        model = AdaBoostClassifier(base_estimator= DecisionTreeClassifier(max_depth=int(self.configdict['Model Parameters']['adaboost_model_classifier']['base_estimator_max_depth'])),
                                                  n_estimators=int(self.configdict['Model Parameters']['adaboost_model_classifier']['n_estimators']),
                                                  learning_rate=float(self.configdict['Model Parameters']['adaboost_model_classifier']['learning_rate']),
                                                  random_state=None)
                    return model
                if model_type == 'nn_model_classifier':
                    if type(self.configdict['Model Parameters']['nn_model_classifier']['hidden_layer_sizes']) is list:
                        model = MLPClassifier(hidden_layer_sizes=int(self.configdict['Model Parameters']['nn_model_classifier']['hidden_layer_sizes'][target_feature_regression_count]),
                                         activation=str(self.configdict['Model Parameters']['nn_model_classifier']['activation'][target_feature_regression_count]),
                                         solver=str(self.configdict['Model Parameters']['nn_model_classifier']['solver'][target_feature_regression_count]),
                                         alpha=float(self.configdict['Model Parameters']['nn_model_classifier']['alpha'][target_feature_regression_count]),
                                         batch_size='auto',
                                         learning_rate='constant',
                                         max_iter=int(self.configdict['Model Parameters']['nn_model_classifier']['max_iter'][target_feature_regression_count]),
                                         tol=float(self.configdict['Model Parameters']['nn_model_classifier']['tol'][target_feature_regression_count]))
                    else:
                        model = MLPClassifier(hidden_layer_sizes=int(self.configdict['Model Parameters']['nn_model_classifier']['hidden_layer_sizes']),
                                         activation=str(self.configdict['Model Parameters']['nn_model_classifier']['activation']),
                                         solver=str(self.configdict['Model Parameters']['nn_model_classifier']['solver']),
                                         alpha=float(self.configdict['Model Parameters']['nn_model_classifier']['alpha']),
                                         batch_size='auto',
                                         learning_rate='constant',
                                         max_iter=int(self.configdict['Model Parameters']['nn_model_classifier']['max_iter']),
                                         tol=float(self.configdict['Model Parameters']['nn_model_classifier']['tol']))
                    return model

        if 'regression' in y_feature:
            if 'regressor' in model_type:
                print('got y_feature', y_feature)
                print('model type is', model_type)
                print('doing regression on', y_feature)
                if model_type == 'linear_model_regressor':
                    if type(self.configdict['Model Parameters']['linear_model_regressor']['fit_intercept']) is list:
                        model = LinearRegression(fit_intercept=bool(self.configdict['Model Parameters']['linear_model_regressor']['fit_intercept'][target_feature_regression_count]))
                    else:
                        model = LinearRegression(fit_intercept=bool(self.configdict['Model Parameters']['linear_model_regressor']['fit_intercept']))
                    return model
                if model_type == 'linear_model_lasso_regressor':
                    if type(self.configdict['Model Parameters']['linear_model_lasso_regressor']['alpha']) is list:
                        model = Lasso(alpha=float(self.configdict['Model Parameters']['linear_model_lasso_regressor']['alpha'][target_feature_regression_count]),
                                      fit_intercept=bool(self.configdict['Model Parameters']['linear_model_lasso_regressor']['fit_intercept'][target_feature_regression_count]))
                    else:
                        model = Lasso(alpha=float(self.configdict['Model Parameters']['linear_model_lasso_regressor']['alpha']),
                                      fit_intercept=bool(self.configdict['Model Parameters']['linear_model_lasso_regressor']['fit_intercept']))
                    return model
                if model_type == 'lkrr_model_regressor':
                    if type(self.configdict['Model Parameters']['lkrr_model_regressor']['alpha']) is list:
                        model = KernelRidge(alpha = float(self.configdict['Model Parameters']['lkrr_model_regressor']['alpha'][target_feature_regression_count]),
                                            gamma = float(self.configdict['Model Parameters']['lkrr_model_regressor']['gamma'][target_feature_regression_count]),
                                            kernel = str(self.configdict['Model Parameters']['lkrr_model_regressor']['kernel'][target_feature_regression_count]))
                    else:
                        model = KernelRidge(alpha=float(self.configdict['Model Parameters']['lkrr_model_regressor']['alpha']),
                                            gamma=float(self.configdict['Model Parameters']['lkrr_model_regressor']['gamma']),
                                            kernel=str(self.configdict['Model Parameters']['lkrr_model_regressor']['kernel']))
                    return model
                if model_type == 'gkrr_model_regressor':
                    if type(self.configdict['Model Parameters']['gkrr_model_regressor']['alpha']) is list:
                        model = KernelRidge(alpha=float(self.configdict['Model Parameters']['gkrr_model_regressor']['alpha'][target_feature_regression_count]),
                                            coef0=int(self.configdict['Model Parameters']['gkrr_model_regressor']['coef0'][target_feature_regression_count]),
                                            degree=int(self.configdict['Model Parameters']['gkrr_model_regressor']['degree'][target_feature_regression_count]),
                                            gamma=float(self.configdict['Model Parameters']['gkrr_model_regressor']['gamma'][target_feature_regression_count]),
                                            kernel=str(self.configdict['Model Parameters']['gkrr_model_regressor']['kernel'][target_feature_regression_count]),
                                            kernel_params=None)
                    else:
                        model = KernelRidge(alpha=float(self.configdict['Model Parameters']['gkrr_model_regressor']['alpha']),
                                            coef0=int(self.configdict['Model Parameters']['gkrr_model_regressor']['coef0']),
                                            degree=int(self.configdict['Model Parameters']['gkrr_model_regressor']['degree']),
                                            gamma=float(self.configdict['Model Parameters']['gkrr_model_regressor']['gamma']),
                                            kernel=str(self.configdict['Model Parameters']['gkrr_model_regressor']['kernel']),
                                            kernel_params=None)
                    return model
                if model_type == 'decision_tree_model_regressor':
                    if type(self.configdict['Model Parameters']['decision_tree_model_regressor']['criterion']) is list:
                        model = DecisionTreeRegressor(criterion=str(self.configdict['Model Parameters']['decision_tree_model_regressor']['criterion'][target_feature_regression_count]),
                                                       splitter=str(self.configdict['Model Parameters']['decision_tree_model_regressor']['splitter'][target_feature_regression_count]),
                                                       max_depth=int(self.configdict['Model Parameters']['decision_tree_model_regressor']['max_depth'][target_feature_regression_count]),
                                                       min_samples_leaf=int(self.configdict['Model Parameters']['decision_tree_model_regressor']['min_samples_leaf'][target_feature_regression_count]),
                                                       min_samples_split=int(self.configdict['Model Parameters']['decision_tree_model_regressor']['min_samples_split'][target_feature_regression_count]))
                    else:
                        model = DecisionTreeRegressor(criterion=str(self.configdict['Model Parameters']['decision_tree_model_regressor']['criterion']),
                                                       splitter=str(self.configdict['Model Parameters']['decision_tree_model_regressor']['splitter']),
                                                       max_depth=int(self.configdict['Model Parameters']['decision_tree_model_regressor']['max_depth']),
                                                       min_samples_leaf=int(self.configdict['Model Parameters']['decision_tree_model_regressor']['min_samples_leaf']),
                                                       min_samples_split=int(self.configdict['Model Parameters']['decision_tree_model_regressor']['min_samples_split']))
                    return model
                if model_type == 'extra_trees_model_regressor':
                    if type(self.configdict['Model Parameters']['extra_trees_model_regressor']['criterion']) is list:
                        model = ExtraTreesRegressor(criterion=str(self.configdict['Model Parameters']['extra_trees_model_regressor']['criterion'][target_feature_regression_count]),
                                                       n_estimators=int(self.configdict['Model Parameters']['extra_trees_model_regressor']['n_estimators'][target_feature_regression_count]),
                                                       max_depth=int(self.configdict['Model Parameters']['extra_trees_model_regressor']['max_depth'][target_feature_regression_count]),
                                                       min_samples_leaf=int(self.configdict['Model Parameters']['extra_trees_model_regressor']['min_samples_leaf'][target_feature_regression_count]),
                                                       min_samples_split=int(self.configdict['Model Parameters']['extra_trees_model_regressor']['min_samples_split'][target_feature_regression_count]),
                                                       max_leaf_nodes=int(self.configdict['Model Parameters']['extra_trees_model_regressor']['max_leaf_nodes'][target_feature_regression_count]))
                    else:
                        model = ExtraTreesRegressor(criterion=str(self.configdict['Model Parameters']['extra_trees_model_regressor']['criterion']),
                                                       n_estimators=int(self.configdict['Model Parameters']['extra_trees_model_regressor']['n_estimators']),
                                                       max_depth=int(self.configdict['Model Parameters']['extra_trees_model_regressor']['max_depth']),
                                                       min_samples_leaf=int(self.configdict['Model Parameters']['extra_trees_model_regressor']['min_samples_leaf']),
                                                       min_samples_split=int(self.configdict['Model Parameters']['extra_trees_model_regressor']['min_samples_split']),
                                                       max_leaf_nodes=int(self.configdict['Model Parameters']['extra_trees_model_regressor']['max_leaf_nodes']))
                    return model
                if model_type == 'randomforest_model_regressor':
                    if type(self.configdict['Model Parameters']['randomforest_model_regressor']['criterion']) is list:
                        model = RandomForestRegressor(criterion=str(self.configdict['Model Parameters']['randomforest_model_regressor']['criterion'][target_feature_regression_count]),
                                                  n_estimators=int(self.configdict['Model Parameters']['randomforest_model_regressor']['n_estimators'][target_feature_regression_count]),
                                                  max_depth=int(self.configdict['Model Parameters']['randomforest_model_regressor']['max_depth'][target_feature_regression_count]),
                                                  min_samples_split=int(self.configdict['Model Parameters']['randomforest_model_regressor']['min_samples_split'][target_feature_regression_count]),
                                                  min_samples_leaf=int(self.configdict['Model Parameters']['randomforest_model_regressor']['min_samples_leaf'][target_feature_regression_count]),
                                                  max_leaf_nodes=int(self.configdict['Model Parameters']['randomforest_model_regressor']['max_leaf_nodes'][target_feature_regression_count]),
                                                  n_jobs=int(self.configdict['Model Parameters']['randomforest_model_regressor']['n_jobs'][target_feature_regression_count]),
                                                  warm_start=bool(self.configdict['Model Parameters']['randomforest_model_regressor']['warm_start'][target_feature_regression_count]),
                                                  bootstrap=True)
                    else:
                        model = RandomForestRegressor(criterion=str(self.configdict['Model Parameters']['randomforest_model_regressor']['criterion']),
                                                  n_estimators=int(self.configdict['Model Parameters']['randomforest_model_regressor']['n_estimators']),
                                                  max_depth=int(self.configdict['Model Parameters']['randomforest_model_regressor']['max_depth']),
                                                  min_samples_split=int(self.configdict['Model Parameters']['randomforest_model_regressor']['min_samples_split']),
                                                  min_samples_leaf=int(self.configdict['Model Parameters']['randomforest_model_regressor']['min_samples_leaf']),
                                                  max_leaf_nodes=int(self.configdict['Model Parameters']['randomforest_model_regressor']['max_leaf_nodes']),
                                                  n_jobs=int(self.configdict['Model Parameters']['randomforest_model_regressor']['n_jobs']),
                                                  warm_start=bool(self.configdict['Model Parameters']['randomforest_model_regressor']['warm_start']),
                                                  bootstrap=True)
                    return model

                if model_type == 'adaboost_model_regressor':
                    if type(self.configdict['Model Parameters']['adaboost_model_regressor']['base_estimator_max_depth']) is list:
                        model = AdaBoostRegressor(base_estimator= DecisionTreeRegressor(max_depth=int(self.configdict['Model Parameters']['adaboost_model_regressor']['base_estimator_max_depth'][target_feature_regression_count])),
                                                  n_estimators=int(self.configdict['Model Parameters']['adaboost_model_regressor']['n_estimators'][target_feature_regression_count]),
                                                  learning_rate=float(self.configdict['Model Parameters']['adaboost_model_regressor']['learning_rate'][target_feature_regression_count]),
                                                  loss=str(self.configdict['Model Parameters']['adaboost_model_regressor']['loss'][target_feature_regression_count]),
                                                  random_state=None)
                    else:
                        model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=int(self.configdict['Model Parameters']['adaboost_model_regressor']['base_estimator_max_depth'])),
                                                  n_estimators=int(self.configdict['Model Parameters']['adaboost_model_regressor']['n_estimators']),
                                                  learning_rate=float(self.configdict['Model Parameters']['adaboost_model_regressor']['learning_rate']),
                                                  loss=str(self.configdict['Model Parameters']['adaboost_model_regressor']['loss']),
                                                  random_state=None)
                    return model

                if model_type == 'nn_model_regressor':
                    if type(self.configdict['Model Parameters']['nn_model_regressor']['hidden_layer_sizes']) is list:
                        model = MLPRegressor(hidden_layer_sizes=int(self.configdict['Model Parameters']['nn_model_regressor']['hidden_layer_sizes'][target_feature_regression_count]),
                                         activation=str(self.configdict['Model Parameters']['nn_model_regressor']['activation'][target_feature_regression_count]),
                                         solver=str(self.configdict['Model Parameters']['nn_model_regressor']['solver'][target_feature_regression_count]),
                                         alpha=float(self.configdict['Model Parameters']['nn_model_regressor']['alpha'][target_feature_regression_count]),
                                         batch_size='auto',
                                         learning_rate='constant',
                                         max_iter=int(self.configdict['Model Parameters']['nn_model_regressor']['max_iter'][target_feature_regression_count]),
                                         tol=float(self.configdict['Model Parameters']['nn_model_regressor']['tol'][target_feature_regression_count]))
                    else:
                        model = MLPRegressor(hidden_layer_sizes=int(self.configdict['Model Parameters']['nn_model_regressor']['hidden_layer_sizes']),
                                         activation=str(self.configdict['Model Parameters']['nn_model_regressor']['activation']),
                                         solver=str(self.configdict['Model Parameters']['nn_model_regressor']['solver']),
                                         alpha=float(self.configdict['Model Parameters']['nn_model_regressor']['alpha']),
                                         batch_size='auto',
                                         learning_rate='constant',
                                         max_iter=int(self.configdict['Model Parameters']['nn_model_regressor']['max_iter']),
                                         tol=float(self.configdict['Model Parameters']['nn_model_regressor']['tol']))
                    return model

                else:
                    model = None
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
