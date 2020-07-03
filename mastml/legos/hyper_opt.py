import sklearn.model_selection as ms
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import scipy.stats
import pandas as pd
import numpy as np
import os
from mastml import utils
import logging

log = logging.getLogger('mastml')

class HyperOptUtils():
    """
    Helper class providing useful methods for other hyperparameter optimization classes.

    Args:

       param_names (list) : list containing names of hyperparams to optimize

       param_values (list) : list containing values of hyperparams to optimize

       param_dict (dict) : dict of {param_name : param_value} pairs. Constructed using one of the _get_*_param_dict methods.

    Methods:

        _search_space_generator : parses GridSearch param_dict and checks values

            Args:

                params (dict) : dict of {param_name : param_value} pairs.

            Returns:

                params_ (dict) : dict of {param_name : param_value} pairs.

        _save_output : saves hyperparameter optimization output and best values to csv file

            Args:

                savepath (str) : path of output directory

                data (dict) : dict of {estimator_name : hyper_opt.GridSearch.fit()} object, or equivalent

        _get_grid_param_dict : configures the param_dict for GridSearch

            Returns:

                param_dict (dict) : dict of {param_name : param_value} pairs.

        _get_randomized_param_dict : configures the param_dict for RandomSearch

            Returns:

                param_dict (dict) : dict of {param_name : param_value} pairs.

        _get_bayesian_param_dict : configures the param_dict for BayesianSearch

            Returns:

                param_dict (dict) : dict of {param_name : param_value} pairs.

    """
    def __init__(self, param_names, param_values):
        self.param_names = param_names
        self.param_values = param_values

    def _search_space_generator(self, params):
        params_ = dict()
        for param_name, param_vals in params.items():
            dtype = param_vals[4]
            try:
                if param_vals[3] == "lin":
                    params_[param_name] = np.linspace(float(param_vals[0]), float(param_vals[1]), num=int(param_vals[2]), dtype=dtype)
                elif param_vals[3] == "log":
                    params_[param_name] = np.logspace(float(param_vals[0]), float(param_vals[1]), num=int(param_vals[2]), dtype=dtype)
                else:
                    log.error('You must specify either lin or log scaling for GridSearch')
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
            best = pd.DataFrame(d.best_params_, index=['Best Parameters'])
            with open(savepath, 'w') as f:
                out.to_csv(f)
            with open(savepath, 'a') as f:
                best.to_csv(f)

    def _get_grid_param_dict(self):
        param_dict = dict()
        try:
            for name, value_string in zip(self.param_names.split(';'), self.param_values.split(';')):
                param_dict[name] = value_string
        except:
            log.error('Error: An error occurred when trying to parse the hyperparam input values.'
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
                log.error('Your hyperparam input values were not parsed correctly, possibly due to unreasonable value choices'
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

        estimator (sklearn estimator object) : an sklearn estimator

        cv (sklearn cross-validator object or iterator) : an sklearn cross-validator

        param_names (list) : list containing names of hyperparams to optimize

        param_values (list) : list containing values of hyperparams to optimize

        scoring (sklearn scoring object or str) : an sklearn scorer

    Methods:

        fit : optimizes hyperparameters

            Args:

                X (np array) : array of X data

                y (np array) : array of y data

                savepath (str) : path of output directory

            Returns:

                best_estimator (sklearn estimator object) : the optimized sklearn estimator

        _estimator_name : returns string of estimator name

    """
    def __init__(self, estimator, cv, param_names, param_values, scoring=None, n_jobs=1):
        super(GridSearch, self).__init__(param_names=param_names, param_values=param_values)
        self.estimator = estimator
        self.cv = cv
        self.param_names = param_names
        self.param_values = param_values
        self.scoring = scoring
        self.n_jobs = int(n_jobs)

    def fit(self, X, y, savepath=None, refit=True, iid=True):
        rst = dict()
        param_dict = self._get_grid_param_dict()

        if savepath is None:
            savepath = os.getcwd()

        estimator_name = self._estimator_name
        param_dict = self._search_space_generator(param_dict)

        if self.cv is None:
            self.cv = ms.RepeatedKFold()

        model = GridSearchCV(self.estimator, param_dict, scoring=self.scoring, cv=self.cv, refit=refit,
                             iid=iid, n_jobs=self.n_jobs, verbose=2)

        try:
            rst[estimator_name] = model.fit(X, y)
        except:
            log.error('Hyperparameter optimization failed, likely due to inappropriate domain of values to optimize'
                               ' one or more parameters over. Please check your input file and the sklearn docs for the mode'
                               ' you are optimizing for the domain of correct values')
            exit()

        best_estimator = rst[estimator_name].best_estimator_

        self._save_output(savepath, rst)
        return best_estimator

    @property
    def _estimator_name(self):
        return self.estimator.__class__.__name__

class RandomizedSearch(HyperOptUtils):
    """
        Class to conduct a randomized search to find optimized model hyperparameter values

        Args:

            estimator (sklearn estimator object) : an sklearn estimator

            cv (sklearn cross-validator object or iterator) : an sklearn cross-validator

            param_names (list) : list containing names of hyperparams to optimize

            param_values (list) : list containing values of hyperparams to optimize

            scoring (sklearn scoring object or str) : an sklearn scorer

            n_iter (int) : number of optimizer iterations

        Methods:

            fit : optimizes hyperparameters

                Args:

                    X (np array) : array of X data

                    y (np array) : array of y data

                    savepath (str) : path of output directory

                Returns:

                    best_estimator (sklearn estimator object) : the optimized sklearn estimator

            _estimator_name : returns string of estimator name

        """
    def __init__(self, estimator, cv, param_names, param_values, scoring=None, n_iter=50, n_jobs=1):
        super(RandomizedSearch, self).__init__(param_names=param_names, param_values=param_values)
        self.estimator = estimator
        self.cv = cv
        self.param_names = param_names
        self.param_values = param_values
        self.scoring = scoring
        self.n_iter = int(n_iter)
        self.n_jobs = int(n_jobs)

    def fit(self, X, y, savepath=None, refit=True):
        rst = dict()
        param_dict = self._get_randomized_param_dict()

        if savepath is None:
            savepath = os.getcwd()

        estimator_name = self._estimator_name

        if self.cv is None:
            self.cv = ms.RepeatedKFold()

        model = RandomizedSearchCV(self.estimator, param_dict, n_iter=self.n_iter, scoring=self.scoring, cv=self.cv,
                                   refit=refit, n_jobs=self.n_jobs, verbose=2)

        try:
            rst[estimator_name] = model.fit(X, y)
        except:
            log.error('Hyperparameter optimization failed, likely due to inappropriate domain of values to optimize'
                               ' one or more parameters over. Please check your input file and the sklearn docs for the mode'
                               ' you are optimizing for the domain of correct values')
            exit()

        best_estimator = rst[estimator_name].best_estimator_

        self._save_output(savepath, rst)
        return best_estimator

    @property
    def _estimator_name(self):
        return self.estimator.__class__.__name__

class BayesianSearch(HyperOptUtils):
    """
    Class to conduct a Bayesian search to find optimized model hyperparameter values

    Args:

        estimator (sklearn estimator object) : an sklearn estimator

        cv (sklearn cross-validator object or iterator) : an sklearn cross-validator

        param_names (list) : list containing names of hyperparams to optimize

        param_values (list) : list containing values of hyperparams to optimize

        scoring (sklearn scoring object or str) : an sklearn scorer

        n_iter (int) : number of optimizer iterations

    Methods:

        fit : optimizes hyperparameters

            Args:

                X (np array) : array of X data

                y (np array) : array of y data

                savepath (str) : path of output directory

            Returns:

                best_estimator (sklearn estimator object) : the optimized sklearn estimator

        _estimator_name : returns string of estimator name
    """

    def __init__(self, estimator, cv, param_names, param_values, scoring=None, n_iter=50, n_jobs=1):
        super(BayesianSearch, self).__init__(param_names=param_names, param_values=param_values)
        self.estimator = estimator
        self.cv = cv
        self.param_names = param_names
        self.param_values = param_values
        self.scoring = scoring
        self.n_iter = int(n_iter)
        self.n_jobs = int(n_jobs)

    def fit(self, X, y, savepath=None, refit=True):
        rst = dict()
        param_dict = self._get_bayesian_param_dict()

        if savepath is None:
            savepath = os.getcwd()

        estimator_name = self._estimator_name

        if self.cv is None:
            self.cv = ms.RepeatedKFold()

        model = BayesSearchCV(estimator=self.estimator, search_spaces=param_dict, n_iter=self.n_iter,
                              scoring=self.scoring, cv=self.cv, refit=refit, n_jobs=self.n_jobs, verbose=2)

        try:
            rst[estimator_name] = model.fit(X, y)
        except:
            log.error('Hyperparameter optimization failed, likely due to inappropriate domain of values to optimize'
                               ' one or more parameters over. Please check your input file and the sklearn docs for the mode'
                               ' you are optimizing for the domain of correct values')
            exit()

        best_estimator = rst[estimator_name].best_estimator_

        self._save_output(savepath, rst)
        return best_estimator

    @property
    def _estimator_name(self):
        return self.estimator.__class__.__name__

name_to_constructor = {'GridSearch': GridSearch, 'RandomizedSearch': RandomizedSearch, 'BayesianSearch': BayesianSearch}