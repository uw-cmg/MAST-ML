__author__ = 'Ryan Jacobs, Tam Mayeshiba'
__maintainer__ = 'Ryan Jacobs'
__version__ = '1.0'
__email__ = 'rjacobs3@wisc.edu'
__date__ = 'October 14th, 2017'

import os
import importlib
import logging
import sklearn
from configobj import ConfigObj, ConfigObjError
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import sklearn.gaussian_process.kernels as skkernel

def get_config_dict(directory, filename, logger):
    """
    Reads in contents of MASTML input file and parse it
    """
    if not os.path.exists(directory):
        message = 'Specified directory %s does not exist.' % directory
        logger.error(message)
        raise OSError(message)

    if not os.path.exists(directory + '/' + str(filename)):
        message = 'Conf file %s does not exist in directory %s' % (directory, filename)
        logging.error(message)
        raise OSError(message)

    original_dir = os.getcwd()
    os.chdir(directory)

    try:
        config_dict = ConfigObj(filename)
    except (ConfigObjError, IOError) as e:
        logger.error('Could not read in conf file %s') % str(self.configfile)
        raise e
    finally:
        os.chdir(original_dir)

    return config_dict

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

        if self.configdict['General Setup']['is_classification']:
            logging.warn('MAST-ML classifiers are still untested')
            logging.info('got y_feature %s' % y_feature)
            logging.info('model type is %s' % model_type)
            logging.info('doing classification on %s' % y_feature)
        else:
            logging.info('got y_feature %s' % y_feature)
            logging.info('model type %s' % model_type)
            logging.info('doing regression on %s' % y_feature)


        name_to_model = {
            'linear_model_regressor': LinearRegression,
            'k_nearest_neighbors_classifier': sklearn.neighbors.KNeighborsClassifier,
            'support_vector_machine_model_classifier': SVC,
            'logistic_regression_model_classifier': LogisticRegression,
            'decision_tree_model_classifier': DecisionTreeClassifier,
            'random_forest_model_classifier': RandomForestClassifier,
            'extra_trees_model_classifier': ExtraTreesClassifier,
            'adaboost_model_classifier': AdaBoostClassifier,
            'nn_model_classifier': MLPClassifier,
            'linear_model_lasso_regressor': Lasso,
            'support_vector_machine_model_regressor': SVR,
            'lkrr_model_regressor': KernelRidge,
            'gkrr_model_regressor': KernelRidge,
            'decision_tree_model_regressor': DecisionTreeRegressor,
            'extra_trees_model_regressor': ExtraTreesRegressor,
            'randomforest_model_regressor': RandomForestRegressor,
            'adaboost_model_regressor': AdaBoostRegressor,
            'nn_model_regressor': MLPRegressor,
        }

        model_parameters = self.configdict['Model Parameters'][model_type]

        # MARK place to add new models

        if model_type in name_to_model:
            Model = name_to_model[model_type]
            try:
                return Model(**model_parameters)
            except TypeError as e:
                logging.error("Invalid parameter for model %s" % model_type)
                raise e

        if model_type == 'gaussianprocess_model_regressor': # Special case
            test_kernel = None
            if d['kernel'] == 'rbf':
                test_kernel = skkernel.ConstantKernel(1.0, (1e-5, 1e5)) * skkernel.RBF(length_scale=d['RBF_length_scale'],
                    length_scale_bounds=tuple(float(i) for i in d['RBF_length_scale_bounds']))
            return GaussianProcessRegressor(kernel=test_kernel,
                                            alpha=model_parameters['alpha'],
                                            optimizer=model_parameters['optimizer'],
                                            n_restarts_optimizer=model_parameters['n_restarts_optimizer'],
                                            normalize_y=model_parameters['normalize_y'])

        raise TypeError('You have specified invalid models in your input file: ' + model_type)

    def get_machinelearning_test(self, test_type, model, save_path, run_test=True, *args, **kwargs):
        mod_name = test_type.split("_")[0] #ex. KFoldCV_5fold goes to KFoldCV
        test_module = importlib.import_module('MLTests.%s' % mod_name)
        test_class_def = getattr(test_module, mod_name)
        logging.debug("Parameters passed by keyword:")
        logging.debug(kwargs)
        test_class = test_class_def(
            model=model,
            save_path=save_path,
            is_classification=self.configdict['General Setup']['is_classification'],
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
