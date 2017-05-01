__author__ = 'Ryan Jacobs'

from configobj import ConfigObj, ConfigObjError
import sys
import os
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import sklearn.tree as tree
import neurolab as nl

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
                                               max_depth=int(self.configdict['Model Parameters']['decision_tree_model']['max_depth']),
                                               min_samples_leaf=int(self.configdict['Model Parameters']['decision_tree_model']['min_samples_leaf']),
                                               min_samples_split=int(self.configdict['Model Parameters']['decision_tree_model']['min_samples_split']))
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
            train = str(self.configdict['Model Parameters']['nn_model_neurolab']['minmax']['training_method'])
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

    def fit_machinelearning_model(self, data):
        if self.configdict['Models and Tests to Run']['models']['nn_model_neurolab']:
            model, training_method, epochs, show, goal = self.get_machinelearning_model(model_type='nn_model_neurolab')
            model_fitted = model.train('x_train', 'y_train', epochs=epochs, show=show, goal=goal)
            return model_fitted

    # This method will call the different classes corresponding to each test type, which are being organized by Tam
    def get_machinelearning_test(self, test_type, model, data, save_path):
        if test_type == 'KFold_CV':
            import KFoldCV as KFoldCV
            KFoldCV.execute(model=model, data=data, savepath=save_path)
        if test_type == 'FullFit':
            import FullFit as FullFit
            FullFit.execute(model=model, data=data, savepath=save_path)
        return None