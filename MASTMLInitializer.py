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
import sklearn
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR
import sklearn.gaussian_process.kernels as skkernel
from sklearn.gaussian_process import GaussianProcessRegressor
from configobj import ConfigObj, ConfigObjError
import distutils.util as du
from ConfigTemplate import configtemplate

def get_config_dict(directory, filename, logger):
    """
    Reads in contents of MASTML input file and parse it
    """
    if not os.path.exists(directory):
        message = 'Specified directory %s does not exist.' % directory
        logger.error(message)
        raise OSError(message)

    if not os.path.exists(directory + '/' + str(self.configfile)):
        message = 'Conf file %s does not exist in directory %s' % (directory, filename)
        logging.error(message)
        raise OSError(message)

    original_dir = os.getcwd()
    os.chdir(path_to_file)

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
        # maybe TODO: require conf file model parameters to conform to sklearn standards, then we
        # can just directly execute and don't have to duplicate the sklearn docs

        if self.configdict['General Setup']['is_classification']:
            logging.warn('MAST-ML classifiers are still untested')
            logging.info('got y_feature %s' % y_feature)
            logging.info('model type is %s' % model_type)
            logging.info('doing classification on %s' % y_feature)
        else:
            logging.info('got y_feature %s' % y_feature)
            logging.info('model type %s' % model_type)
            logging.info('doing regression on %s' % y_feature)

        d = self.configdict['Model Parameters'][model_type]

        # MARK place to add new models

        if model_type == 'linear_model_regressor':
            return LinearRegression(fit_intercept=d['fit_intercept'])

        if model_type == 'k_nearest_neighbors_classifier':
            return sklearn.neighbors.KNeighborsClassifier()

        if model_type == 'support_vector_machine_model_classifier':
            return SVC(C=float(d['error_penalty']),
                       kernel=str(d['kernel']),
                       degree=int(d['degree']),
                       gamma=float(d['gamma']),
                       coef0=float(d['coef0']))

        if model_type == 'logistic_regression_model_classifier':
            return LogisticRegression(penalty=str(d['penalty']),
                                      C=float(d['C']),
                                      class_weight=str(d['class_weight']))

        if model_type == 'decision_tree_model_classifier':
            return DecisionTreeClassifier(criterion=str(d['criterion']),
                                          splitter=str(d['splitter']),
                                          max_depth=int(d['max_depth']),
                                          min_samples_leaf=int(d['min_samples_leaf']),
                                          min_samples_split=int(d['min_samples_split']))

        if model_type == 'random_forest_model_classifier':
            return RandomForestClassifier(criterion=str(d['criterion']),
                                          n_estimators=int(d['n_estimators']),
                                          max_depth=int(d['max_depth']),
                                          min_samples_split=int(d['min_samples_split']),
                                          min_samples_leaf=int(d['min_samples_leaf']),
                                          max_leaf_nodes=int(d['max_leaf_nodes']))

        if model_type == 'extra_trees_model_classifier':
            return ExtraTreesClassifier(criterion=str(d['criterion']),
                                        n_estimators=int(d['n_estimators']),
                                        max_depth=int(d['max_depth']),
                                        min_samples_split=int(d['min_samples_split']),
                                        min_samples_leaf=int(d['min_samples_leaf']),
                                        max_leaf_nodes=int(d['max_leaf_nodes']))

        if model_type == 'adaboost_model_classifier':
            return AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=int(d['base_estimator_max_depth'])),
                                      n_estimators=int(d['n_estimators']),
                                      learning_rate=float(d['learning_rate']),
                                      random_state=None)

        if model_type == 'nn_model_classifier':
            return MLPClassifier(hidden_layer_sizes=int(d['hidden_layer_sizes']),
                                 activation=str(d['activation']),
                                 solver=str(d['solver']),
                                 alpha=float(d['alpha']),
                                 batch_size='auto',
                                 learning_rate='constant',
                                 max_iter=int(d['max_iterations']),
                                 # maybe TODO: just use the sklearn names for these parameters,
                                 # instead of our own names
                                 tol=float(d['tolerance']))

        if model_type == 'linear_model_lasso_regressor':
            return Lasso(alpha=d['alpha'],
                         fit_intercept=d['fit_intercept'])

        if model_type == 'support_vector_machine_model_regressor':
            return SVR(C=d['error_penalty'],
                       kernel=d['kernel'],
                       degree=d['degree'],
                       gamma=d['gamma'],
                       coef0=d['coef0'])

        if model_type == 'lkrr_model_regressor':
            return KernelRidge(alpha=d['alpha'],
                               gamma=d['gamma'],
                               kernel=d['kernel'])

        if model_type == 'gkrr_model_regressor':
            return KernelRidge(alpha=d['alpha'],
                               coef0=d['coef0'],
                               degree=d['degree'],
                               gamma=d['gamma'],
                               kernel=d['kernel'],
                               kernel_params=None)


        if model_type == 'decision_tree_model_regressor':
            return DecisionTreeRegressor(criterion=d['criterion'],
                                         splitter=d['splitter'],
                                         max_depth=d['max_depth'],
                                         min_samples_leaf=d['min_samples_leaf'],
                                         min_samples_split=d['min_samples_split'])

        if model_type == 'extra_trees_model_regressor':
            return ExtraTreesRegressor(criterion=d['criterion'],
                                       n_estimators=d['n_estimators'],
                                       max_depth=d['max_depth'],
                                       min_samples_leaf=d['min_samples_leaf'],
                                       min_samples_split=d['min_samples_split'],
                                       max_leaf_nodes=d['max_leaf_nodes'])

        if model_type == 'randomforest_model_regressor':
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
            return AdaBoostRegressor(base_estimator=d['base_estimator_max_depth'],
                                     n_estimators=d['n_estimators'],
                                     learning_rate=d['learning_rate'],
                                     loss=d['loss'],
                                     random_state=None)

        if model_type == 'nn_model_regressor':
            return MLPRegressor(hidden_layer_sizes=d['hidden_layer_sizes'],
                                activation=d['activation'],
                                solver=d['solver'],
                                alpha=d['alpha'],
                                batch_size='auto',
                                learning_rate='constant',
                                max_iter=d['max_iterations'],
                                tol=d['tolerance'])

        if model_type == 'gaussianprocess_model_regressor':
            test_kernel = None
            if d['kernel'] == 'rbf':
                test_kernel = skkernel.ConstantKernel(1.0, (1e-5, 1e5)) * skkernel.RBF(length_scale=d['RBF_length_scale'],
                    length_scale_bounds=tuple(float(i) for i in d['RBF_length_scale_bounds']))
            return GaussianProcessRegressor(kernel=test_kernel,
                                            alpha=d['alpha'],
                                            optimizer=d['optimizer'],
                                            n_restarts_optimizer=d['n_restarts_optimizer'],
                                            normalize_y=d['normalize_y'],
                                            copy_X_train=True)

        raise TypeError('You have specified invalid models in your input file: ' + model_type)

    def get_machinelearning_test(self, test_type, model, save_path, run_test=True, *args, **kwargs):
        mod_name = test_type.split("_")[0] #ex. KFoldCV_5fold goes to KFoldCV
        test_module = importlib.import_module('%s' % (mod_name))
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
