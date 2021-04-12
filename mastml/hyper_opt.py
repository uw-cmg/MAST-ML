"""
This module contains methods for optimizing hyperparameters of models

HyperOptUtils:
    This class contains various helper utilities for setting up and running hyperparameter optimization

GridSearch:
    This class performs a basic grid search over the parameters and value ranges of interest to find the best
    set of model hyperparameters in the provided grid of values

RandomizedSearch:
    This class performs a randomized search over the parameters and value ranges of interest to find the best
    set of model hyperparameters in the provided grid of values. Often faster than GridSearch. Instead of a grid
    of values, it takes a probability distribution name as input (e.g. "norm")

BayesianSearch:
    This class performs a Bayesian search over the parameters and value ranges of interest to find the best
    set of model hyperparameters in the provided grid of values. Often faster than GridSearch.

"""

import sklearn.model_selection as ms
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import scipy.stats
import pandas as pd
import numpy as np
import os
from ast import literal_eval

from mastml.models import SklearnModel
from mastml.metrics import Metrics

class HyperOptUtils():
    """
    Helper class providing useful methods for other hyperparameter optimization classes.

    Args:
       param_names: (list), list containing names of hyperparams to optimize

       param_values: (list), list containing values of hyperparams to optimize

    Methods:
        _search_space_generator : parses GridSearch param_dict and checks values
            Args:
                params: (dict), dict of {param_name : param_value} pairs.

            Returns:
                params_: (dict), dict of {param_name : param_value} pairs.

        _save_output : saves hyperparameter optimization output and best values to csv file
            Args:
                savepath: (str), path of output directory

                data: (dict), dict of {estimator_name : hyper_opt.GridSearch.fit()} object, or equivalent

            Returns:
                None

        _get_grid_param_dict : configures the param_dict for GridSearch
            Args:
                None

            Returns:
                param_dict: (dict), dict of {param_name : param_value} pairs.

        _get_randomized_param_dict : configures the param_dict for RandomSearch
            Args:
                None

            Returns:
                param_dict: (dict), dict of {param_name : param_value} pairs.

        _get_bayesian_param_dict : configures the param_dict for BayesianSearch
            Args:
                None

            Returns:
                param_dict: (dict), dict of {param_name : param_value} pairs.

    """
    def __init__(self, param_names, param_values):
        self.param_names = param_names
        self.param_values = param_values

    def _search_space_generator(self, params):
        params_ = dict()
        for param_name, param_vals in params.items():
            if 'int' in param_vals:
                dtype = 'int'
            elif 'float' in param_vals:
                dtype = 'float'
            elif 'str' in param_vals:
                dtype = 'str'
                param_vals.remove('str')
            elif 'tup' in param_vals:
                is_tuple = True
                param_vals.remove('tup')
            else:
                print('Error: You must specify datatype as int, float or str (last entry in param values for a given parameter)')
            try:
                if param_vals[3] == "lin":
                    params_[param_name] = np.linspace(float(param_vals[0]), float(param_vals[1]), num=int(param_vals[2]), dtype=dtype)
                elif param_vals[3] == "log":
                    params_[param_name] = np.logspace(float(param_vals[0]), float(param_vals[1]), num=int(param_vals[2]), dtype=dtype)
                else:
                    if is_tuple is True:
                        param_vals = [literal_eval(param_val) for param_val in param_vals]
                        params_[param_name] = np.array(param_vals)
                    else:
                        print('You must specify either lin or log scaling for GridSearch, or be specifying a set of tuples')
                        exit()
            except:
                params_[param_name] = param_vals
        return params_

    def _save_output(self, savepath, data):
        for key in data:
            d = data[key]
            c = dict((k, d.cv_results_[k]) for k in ('mean_test_score', 'std_test_score'))
                     # Bayesian search does not report test scores and will error out
                     #('mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score'))
            out = pd.DataFrame(c, d.cv_results_['params'])
            try:
                best = pd.DataFrame(d.best_params_, index=['Best Parameters'])
            except:
                best = pd.DataFrame(d.best_params_)
            out.to_excel(os.path.join(savepath, self.__class__.__name__+"_"+str(key)+'_output.xlsx'))
            best.to_excel(os.path.join(savepath, self.__class__.__name__+"_"+str(key)+'_bestparams.xlsx'))
        return

    def _get_grid_param_dict(self):
        param_dict = dict()
        try:
            for name, value_string in zip(self.param_names.split(';'), self.param_values.split(';')):
                param_dict[name] = value_string
        except:
            print('Error: An error occurred when trying to parse the hyperparam input values.'
                            ' Please check your input file for errors. Remember values need to be delimited by semicolons')
            exit()

        # Clean spaces in names and value strings
        param_dict_ = dict()
        for name, value_string in param_dict.items():
            if name[0] == ' ':
                name = name[1:]
            if name[-1] == ' ':
                name = name[0:-1]
            if value_string[0] == ' ':
                value_string = value_string[1:]
            if value_string[-1] == ' ':
                value_string = value_string[0:-1]
            param_dict_[name] = value_string.split(' ')

        param_dict = param_dict_
        return param_dict

    def _get_randomized_param_dict(self):
        param_dict = self._get_grid_param_dict()
        param_dict_ = dict()

        # Now fetch scipy.stats probability distributions from provided strings
        for param_name, param_val in param_dict.items():
            # Try making param_val a scipy.stats object. If it isn't assume it's to be made into a list (e.g. case of trying different string args).
            try:
                dist = getattr(scipy.stats, param_val[0])
                param_dict_[param_name] = dist
            except AttributeError:
                param_dict_[param_name] = param_val
        param_dict = param_dict_
        return param_dict

    def _get_bayesian_param_dict(self):
        param_dict = self._get_grid_param_dict()
        param_dict_ = dict()

        # Construct prior distribution for Bayesian search
        for param_name, param_val in param_dict.items():
            param_val_split = param_val
            is_str = False
            is_int = False
            is_float = False
            if param_val_split[-1] == 'int':
                is_int = True
            elif param_val_split[-1] == 'float':
                is_float = True
            elif param_val_split[-1] == 'str':
                is_str = True
            else:
                print('An error occurred with parsing your param_dict for hyperparam optimization. You must choose one of'
                      '[int, float, str] as second to last entry for Bayesian Search')

            prior = str(param_val_split[-2])
            if prior == 'log':
                prior = 'log-uniform'
                # If someone specifies log spacing, assume they mean to have floats and not ints for linear spacing, and
                # override is_int = True from above
                if is_float is True:
                    start = float(10**(float(param_val_split[0])))
                    end = float(10**(float(param_val_split[1])))
                elif is_int is True:
                    start = int(param_val_split[0])
                    end = int(param_val_split[1])
            if prior == 'lin':
                prior = 'uniform'

            if is_int is True:
                start = int(param_val_split[0])
                end = int(param_val_split[1])
                param_val_ = Integer(start, end)
            elif is_float is True:
                if prior == 'uniform':
                    start = float(param_val_split[0])
                    end = float(param_val_split[1])
                elif prior == 'log-uniform':
                    start = float(10**(float(param_val_split[0])))
                    end = float(10**(float(param_val_split[1])))
                param_val_ = Real(start, end, prior=prior)
            elif is_str is True:
                param_val_ = Categorical([s for s in param_val_split if s not in ['int', 'float', 'str', 'lin', 'log']])
            else:
                print('Your hyperparam input values were not parsed correctly, possibly due to unreasonable value choices'
                          '(e.g. negative values when only positive values make sense). Please check your input file and '
                          're-run MAST-ML.')
                exit()

            param_dict_[param_name] = param_val_

        param_dict = param_dict_
        return param_dict

class GridSearch(HyperOptUtils):
    """
    Class to conduct a grid search to find optimized model hyperparameter values

    Args:
        param_names: (list), list containing names of hyperparams to optimize

        param_values: (list), list containing values of hyperparams to optimize

        scoring: (str), string denoting name of regression metric to evaluate learning curves. See mastml.metrics.Metrics._metric_zoo for full list

        n_jobs: (int), number of jobs to run in parallel. Can speed up calculation when using multiple cores

    Methods:
        fit : optimizes hyperparameters
            Args:
                X: (pd.DataFrame), dataframe of X feature data

                y: (pd.Series), series of target y data

                model: (mastml.models object), a MAST-ML model, e.g. SklearnModel or EnsembleModel

                cv: (scikit-learn cross-validation object), a scikit-learn cross-validation object

                savepath: (str), path of output directory

            Returns:
                best_estimator (mastml.models object) : the optimized MAST-ML model

    """
    def __init__(self, param_names, param_values, scoring=None, n_jobs=1):
        super(GridSearch, self).__init__(param_names=param_names, param_values=param_values)
        self.param_names = param_names
        self.param_values = param_values
        self.scoring = scoring
        self.n_jobs = int(n_jobs)

    def fit(self, X, y, model, cv=None, savepath=None):
        rst = dict()
        param_dict = self._get_grid_param_dict()

        if savepath is None:
            savepath = os.getcwd()

        estimator_name = model.model.__class__.__name__
        param_dict = self._search_space_generator(param_dict)

        if cv is None:
            cv = ms.RepeatedKFold()

        metrics = Metrics(metrics_list=None)._metric_zoo()
        if self.scoring is None:
            scoring = make_scorer(metrics['mean_absolute_error'][1],
                                  greater_is_better=metrics['mean_absolute_error'][0])  # Note using True b/c if False then sklearn multiplies by -1
        else:
            scoring = make_scorer(metrics[self.scoring][1],
                                  greater_is_better=metrics[self.scoring][0])  # Note using True b/c if False then sklearn multiplies by -1

        model = GridSearchCV(model.model,
                             param_dict,
                             scoring=scoring,
                             cv=cv,
                             refit=True,
                             n_jobs=self.n_jobs,
                             verbose=0)

        try:
            rst[estimator_name] = model.fit(X, y)
        except:
            print('Hyperparameter optimization failed, likely due to inappropriate domain of values to optimize'
                               ' one or more parameters over. Please check your input file and the sklearn docs for the mode'
                               ' you are optimizing for the domain of correct values')
            exit()

        best_estimator = rst[estimator_name].best_estimator_

        self._save_output(savepath, rst)

        # Need to rebuild the estimator as SklearnModel
        best_estimator = SklearnModel(model=best_estimator.__class__.__name__, **best_estimator.get_params())

        return best_estimator

class RandomizedSearch(HyperOptUtils):
    """
    Class to conduct a randomized search to find optimized model hyperparameter values

    Args:
        param_names: (list), list containing names of hyperparams to optimize

        param_values: (list), list containing values of hyperparams to optimize

        scoring: (str), string denoting name of regression metric to evaluate learning curves. See mastml.metrics.Metrics._metric_zoo for full list

        n_iter: (int), number denoting the number of evaluations in the search space to perform. Higher numbers will take longer but will be more accurate

        n_jobs: (int), number of jobs to run in parallel. Can speed up calculation when using multiple cores

    Methods:
        fit : optimizes hyperparameters
            Args:
                X: (pd.DataFrame), dataframe of X feature data

                y: (pd.Series), series of target y data

                model: (mastml.models object), a MAST-ML model, e.g. SklearnModel or EnsembleModel

                cv: (scikit-learn cross-validation object), a scikit-learn cross-validation object

                savepath: (str), path of output directory

            Returns:
                best_estimator (mastml.models object) : the optimized MAST-ML model

        """
    def __init__(self, param_names, param_values, scoring=None, n_iter=50, n_jobs=1):
        super(RandomizedSearch, self).__init__(param_names=param_names, param_values=param_values)

        self.param_names = param_names
        self.param_values = param_values
        self.scoring = scoring
        self.n_iter = int(n_iter)
        self.n_jobs = int(n_jobs)

    def fit(self, X, y, model, cv=None, savepath=None, refit=True):
        rst = dict()
        param_dict = self._get_randomized_param_dict()

        if savepath is None:
            savepath = os.getcwd()

        estimator_name = model.model.__class__.__name__

        if cv is None:
            cv = ms.RepeatedKFold()

        metrics = Metrics(metrics_list=None)._metric_zoo()
        if self.scoring is None:
            scoring = make_scorer(metrics['mean_absolute_error'][1],
                                  greater_is_better=metrics['mean_absolute_error'][0])  # Note using True b/c if False then sklearn multiplies by -1
        else:
            scoring = make_scorer(metrics[self.scoring][1],
                                  greater_is_better=metrics[self.scoring][0])  # Note using True b/c if False then sklearn multiplies by -1

        model = RandomizedSearchCV(model.model,
                                   param_dict,
                                   n_iter=self.n_iter,
                                   scoring=scoring,
                                   cv=cv,
                                   refit=refit,
                                   n_jobs=self.n_jobs,
                                   verbose=0)

        try:
            rst[estimator_name] = model.fit(X, y)
        except:
            print('Hyperparameter optimization failed, likely due to inappropriate domain of values to optimize'
                               ' one or more parameters over. Please check your input file and the sklearn docs for the mode'
                               ' you are optimizing for the domain of correct values')
            exit()

        best_estimator = rst[estimator_name].best_estimator_

        # Need to rebuild the best estimator back into SklearnModel object
        best_estimator = SklearnModel(model=best_estimator.__class__.__name__, **best_estimator.get_params())

        self._save_output(savepath, rst)
        return best_estimator

# NOTE: there is a known problem where BayesSearchCV in skopt doesn't work with sklearn 0.24 (deprecated iid parameter).
# They are working on fixing this (as of 2/4/21). See updates at https://github.com/scikit-optimize/scikit-optimize/issues/978
class BayesianSearch(HyperOptUtils):
    """
    Class to conduct a Bayesian search to find optimized model hyperparameter values

    Args:
        param_names: (list), list containing names of hyperparams to optimize

        param_values: (list), list containing values of hyperparams to optimize

        scoring: (str), string denoting name of regression metric to evaluate learning curves. See mastml.metrics.Metrics._metric_zoo for full list

        n_iter: (int), number denoting the number of evaluations in the search space to perform. Higher numbers will take longer but will be more accurate

        n_jobs: (int), number of jobs to run in parallel. Can speed up calculation when using multiple cores

    Methods:
        fit : optimizes hyperparameters
            Args:
                X: (pd.DataFrame), dataframe of X feature data

                y: (pd.Series), series of target y data

                model: (mastml.models object), a MAST-ML model, e.g. SklearnModel or EnsembleModel

                cv: (scikit-learn cross-validation object), a scikit-learn cross-validation object

                savepath: (str), path of output directory

            Returns:
                best_estimator (mastml.models object) : the optimized MAST-ML model
    """

    def __init__(self, param_names, param_values, scoring=None, n_iter=50, n_jobs=1):
        print('Warning: As of 2/4/21, Bayesian search from skopt is not compatible with'
              ' sklearn>=0.24. Downgrade to sklearn 0.23.2 should fix the issue but may cause'
              ' other unforseen compatibility issues in the MAST-ML code')

        super(BayesianSearch, self).__init__(param_names=param_names, param_values=param_values)
        self.param_names = param_names
        self.param_values = param_values
        self.scoring = scoring
        self.n_iter = int(n_iter)
        self.n_jobs = int(n_jobs)

    def fit(self, X, y, model, cv, savepath=None):
        rst = dict()
        param_dict = self._get_bayesian_param_dict()

        if savepath is None:
            savepath = os.getcwd()

        estimator_name = model.__class__.__name__

        if cv is None:
            cv = ms.RepeatedKFold()

        metrics = Metrics(metrics_list=None)._metric_zoo()
        if self.scoring is None:
            scoring = make_scorer(metrics['mean_absolute_error'][1],
                                  greater_is_better=metrics['mean_absolute_error'][0])  # Note using True b/c if False then sklearn multiplies by -1
        else:
            scoring = make_scorer(metrics[self.scoring][1],
                                  greater_is_better=metrics[self.scoring][0])  # Note using True b/c if False then sklearn multiplies by -1

        model = BayesSearchCV(estimator=model.model,
                              search_spaces=param_dict,
                              n_iter=self.n_iter,
                              scoring=scoring,
                              cv=cv,
                              refit=True,
                              n_jobs=self.n_jobs,
                              verbose=1)

        try:
            rst[estimator_name] = model.fit(X, y)
        except:
            print('Hyperparameter optimization failed, likely due to inappropriate domain of values to optimize'
                               ' one or more parameters over. Please check your input file and the sklearn docs for the mode'
                               ' you are optimizing for the domain of correct values')
            exit()

        best_estimator = rst[estimator_name].best_estimator_

        # Need to rebuild the estimator as SklearnModel
        best_estimator = SklearnModel(model=best_estimator.__class__.__name__, **best_estimator.get_params())

        self._save_output(savepath, rst)
        return best_estimator

    @property
    def _estimator_name(self):
        return self.estimator.__class__.__name__
