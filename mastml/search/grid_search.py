__author__ = 'Tam Mayeshiba'
__maintainer__ = 'Ryan Jacobs'
__version__ = '1.0'
__email__ = 'rjacobs3@wisc.edu'
__date__ = 'October 14th, 2017'

import os
import numpy as np
import pandas as pd
import copy
import logging
import sys
import time
import itertools
from sklearn.externals import joblib
import sklearn.model_selection

from .. import plot_helper

logger = logging.getLogger('mastml')

class GridSearch:
    """Class to perform parameter optimization by grid search. Only up to 4 parameters may be optimized at a time.

    Args:
        training_dataset (DataHandler object): Training dataset handler
        testing_dataset (DataHandler object): Testing dataset handler
        model (sklearn model object): sklearn model
        save_path (str): Save path

        param_strings (list of str): list of [parameter string made up of semicolon-delimited pieces]

            Piece 1: The word 'model' or a custom feature class.method string, e.g. DBTT.calculate_EffectiveFluence,
            where the custom module has the same name as the custom class, and resides inside the custom_features folder

            Piece 2: The parameter name

            Piece 3: The parameter type. Use only: 'int', 'float', 'bool', 'str' (Some sklearn model hyperparameters are type-sensitive)

            Piece 4: The series type. Use:
            'discrete': List will be given in piece 5
            'bool' and 'str' MUST use 'discrete'
            'continuous': Range and step will be given in piece 5
            'continuous-log: Range and step will be given in piece 5

            Piece 5: A colon-delimited list of
            (a) a discrete list of values to grid over, OR
            (b) start, end, number of points: numpy's np.linspace or np.logspace function will be used to generate this list,
            using an inclusive start and inclusive end. A parameter with only one discrete value will not be considered as an 'optimized' parameter.

        fix_random_for_testing (int): 0 - use random numbers
                                      1 - fix randomizer for testing
        num_cvtests (int): Number of CV tests for each validation step
        num_folds (int): Number of folds for K-fold cross validation; leave blank to use LO% CV
        percent_leave_out (int): Percentage to leave out for LO% CV; leave blank to use K-fold CV
        mark_outlying_points (list of int): Number of outlying points to mark in best and worst tests, e.g. [0,3]
        num_bests (int): Number of best individuals to track
        processors (int): Number of processors to use 1 - single processor (serial); 2 - use multiprocessing with this many processors, all on a SINGLE node
        pop_upper_limit (int): Upper limit for population size.

    Returns:
        Analysis in the save_path folder
        Plots results in a predicted vs. measured square plot.

    Raises:
        ValueError if testing target data is None; CV must have testing target data

    """
    def __init__(self, param_strings, training_dataset, testing_dataset, model,
                 save_path=None, xlabel="Measured", ylabel="Predicted", fix_random_for_testing=0,
                 num_cvtests=5, mark_outlying_points='0,3', num_folds=None, percent_leave_out=None,
                 processors=1, pop_upper_limit=1000000, num_bests=10):
        """
        Additional class attributes to parent class:
            Set by keyword:
                self.fix_random_for_testing
                self.num_cvtests
                self.mark_outlying_points
                self.num_folds
                self.percent_leave_out
                self.processors
                self.pop_upper_limit
                self.num_bests
                self.param_strings
            Set in code:
                self.opt_dict
                self.opt_param_list
                self.nonopt_param_list
                self.flat_results
                self.pop_params
                self.pop_size
                self.pop_stats
                self.pop_rmses
                self.best_indivs
                self.best_params
                ?self.random_state
        """
        if not(training_dataset == testing_dataset):
            raise ValueError("Only testing_dataset will be used. Use the same values for training_dataset and testing_dataset")

        self.single_fit_init(
            training_dataset=training_dataset, #only testing_dataset is used (???)
            testing_dataset=testing_dataset,
            model=model,
            save_path = save_path,
            xlabel=xlabel,
            ylabel=ylabel)
        self.fix_random_for_testing = int(fix_random_for_testing)
        if self.fix_random_for_testing == 1:
            self.random_state = np.random.RandomState(0)
        else:
            self.random_state = np.random.RandomState()
        self.num_cvtests = int(num_cvtests)
        self.mark_outlying_points = mark_outlying_points
        self.num_folds = num_folds
        self.percent_leave_out = percent_leave_out
        self.processors=int(processors)
        self.pop_upper_limit = int(pop_upper_limit)
        self.num_bests = int(num_bests)

        ### MARK Don't do this
        #self.param_strings = dict()
        #for argname in kwargs.keys():
        #    if 'param_' in argname:
        #        param_num = int(argname.split("_")[1].strip())
        #        self.param_strings[param_num] = kwargs[argname]

        self.param_strings = param_strings # should be a list of strings in that crazy format

        # Sets later in code
        self.opt_dict=None
        self.opt_param_list=None
        self.nonopt_param_list=None
        self.pop_params=None
        self.pop_size=None
        self.pop_stats=None
        self.pop_rmses=None
        self.flat_results=None
        self.best_indivs=None
        self.best_params=None
        return

    ### SingleFit section

    def single_fit_init(self, training_dataset=None, testing_dataset=None, model=None, save_path=None,
                        xlabel="Measured", ylabel="Predicted", plot_filter_out=None, scaler=None):
        """
        Initialize class.
            Attributes that can be set through keywords:
                self.training_dataset
                self.testing_dataset
                self.model
                self.save_path
                self.xlabel
                self.ylabel
                self.plot_filter_out
            Other attributes:
                self.analysis_name <str>
                self.trained_model <sklearn model>: trained model
                self.statistics <dict of float>: statistics dictionary
                self.readme_list <dict of str>: stores lines for readme
        """
        # Keyword-set attributes
        # training csv
        if training_dataset is None:
            raise ValueError("training_dataset is not set")
        if type(training_dataset) is list: #allow inheriting classes to get multiple datasets; MASTML will pass list of data objects
            self.training_dataset= copy.deepcopy(training_dataset[0]) #first item
        else:
            self.training_dataset = copy.deepcopy(training_dataset)

        # testing csv
        if testing_dataset is None:
            raise ValueError("testing_dataset is not set")
        if type(testing_dataset) is list:
            self.testing_dataset = copy.deepcopy(testing_dataset[0])
        else:
            self.testing_dataset=copy.deepcopy(testing_dataset)
        self.scaler = scaler

        # model
        if model is None:
            raise ValueError("No model.")
        self.model=model
        # paths
        if save_path is None:
            save_path = os.getcwd()
        save_path = os.path.abspath(save_path)
        self.save_path = save_path
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        # plot customization
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_filter_out = self.get_plot_filter(plot_filter_out)
        # Self-set attributes
        self.analysis_name = os.path.basename(self.save_path)
        self.trained_model=None
        self.statistics=dict()
        self.readme_list=list()
        self.testing_dataset_filtered=None
        #
        logger.info("-------- %s --------" % self.analysis_name)
        logger.info("Starting analysis at %s" % time.asctime())
        return

    def get_plot_filter(self, plot_filter_out):
        if (plot_filter_out is None or plot_filter_out == 'None' or plot_filter_out == ''):
            self.plot_filter_out = None
            return
        pf_tuple_list = list()
        if type(plot_filter_out) is list:
            pfo_list = plot_filter_out
        elif type(plot_filter_out) is str:
            pfo_list = plot_filter_out.split(",")
        for pfo_item in pfo_list:
            if type(pfo_item) is tuple:
                pf_tuple_list.append(pfo_item)
            else:
                pflist = pfo_item.split(";")
                feature = pflist[0].strip()
                symbol = pflist[1].strip()
                rawvalue = pflist[2].strip()
                try:
                    value = float(rawvalue)
                except ValueError:
                    value = rawvalue
                pf_tuple_list.append((feature, symbol, value))
        return pf_tuple_list

    def save_model(self):
        pname = os.path.join(self.save_path, "model.pickle")
        with open(pname,'wb') as pfile:
            joblib.dump(self.model, pfile)

    def print_readme(self):
        with open(os.path.join(self.save_path,"README"), 'w') as rfile:
            rfile.writelines(self.readme_list)

    def run(self):
        self.set_up()
        self.evaluate_pop()
        self.get_best_indivs()
        self.print_results()
        self.plot()
        self.print_readme()
        return

    def print_results(self):
        self.print_best_params()
        self.save_best_model()
        self.print_best_dataframe()
        self.flatten_results()
        return

    def set_up(self):
        self.set_up_prior_to_population()
        self.set_up_pop_params()
        #logger.debug("Population: %s" % self.pop_params)
        logger.debug("Population size: %i" % len(self.pop_params))
        return

    def set_up_prior_to_population(self):
        self.readme_list.append("%s\n" % time.asctime())
        self.set_up_opt_dict()
        logger.debug("opt dict: %s" % self.opt_dict)
        logger.debug("opt param list: %s" % self.opt_param_list)
        logger.debug("nonopt param list: %s" % self.nonopt_param_list)
        return

    def evaluate_pop(self):
        """make model and new testing dataset for each pop member
            and evaluate
        """
        self.pop_stats=dict()
        self.pop_rmses=dict()

        if self.processors == 1:
            for ikey in self.pop_params.keys():
                #sys.stdout.write("\rIndividual %s/%i" % (ikey, self.pop_size)) # loading bar HACK
                sys.stdout.write(f"\rMaking individuals [{'+'*int(ikey)}{'-'*(self.pop_size-int(ikey))}]") # loading bar HACK
                sys.stdout.flush()
                indiv_params = self.pop_params[ikey]
                [indiv_rmse, indiv_stats] = self.evaluate_indiv(indiv_params, ikey)
                self.pop_stats[ikey] = indiv_stats
                self.pop_rmses[ikey] = indiv_rmse
            print()
            logger.debug(f'finished generation of {self.pop_size} individuals.')
        else:
            from multiprocessing import Process, Manager
            manager = Manager()
            pop_stats_dict = manager.dict()
            pop_rmses_dict = manager.dict()
            pop_done = manager.Value('i',0)
            #create a number of lists each with the size of number of processors
            pop_keys = list(self.pop_params.keys())
            num_lists = int(np.ceil(self.pop_size/self.processors)) #round up
            flat_lists = list()
            pkct = 0
            for listct in range(0, num_lists):
                flat_list = list()
                for item_ct in range(0, self.processors):
                    if pkct < self.pop_size: #last list may have fewer entries
                        flat_list.append(pop_keys[pkct])
                        pkct = pkct + 1
                flat_lists.append(flat_list)
            for listct in range(0, num_lists):
                pkey_list = flat_lists[listct]
                indiv_p_list = list()
                for iidx in range(0, len(pkey_list)):
                    ikey = pkey_list[iidx]
                    logger.debug("Individual %s/%i" % (ikey, self.pop_size))
                    indiv_params = self.pop_params[ikey]
                    indiv_p = Process(target=self.evaluate_indiv_multiprocessing, args=(indiv_params, ikey, pop_stats_dict, pop_rmses_dict, pop_done))
                    indiv_p_list.append(indiv_p)
                    indiv_p.start()
                for indiv_p in indiv_p_list:
                    indiv_p.join()
            self.pop_stats = pop_stats_dict
            self.pop_rmses = pop_rmses_dict
        return

    def evaluate_indiv_multiprocessing(self, indiv_params, indiv_key, pop_stats_dict, pop_rmses_dict, pop_done):
        [indiv_rmse, indiv_stats] = self.evaluate_indiv(indiv_params, indiv_key)
        pop_stats_dict[indiv_key] = indiv_stats
        pop_rmses_dict[indiv_key] = indiv_rmse
        pop_done.value += 1
        logger.debug("Individual %s done (multiprocessing), %i/%i" % (indiv_key, pop_done.value, self.pop_size))
        return

    def evaluate_indiv(self, indiv_params, indiv_key):
        """Evaluate an individual
        """
        indiv_model = copy.deepcopy(self.model)
        try:
            indiv_model.set_params(**indiv_params['model'])
        except ValueError as e:
            logger.debug('good params: ', indiv_model.get_params().keys())
            logger.debug('your params: ', indiv_params['model'].keys())
            raise e

        indiv_dh = self.get_indiv_datahandler(indiv_params)
        #logging.debug(indiv_dh)
        indiv_path = os.path.join(self.save_path, "indiv_%s" % indiv_key)

        if self.num_folds is not None: # CZECK_mARK replace with sklearn
            splitter = sklearn.model_selection.KFold(n_splits=self.num_folds)
        elif self.percent_leave_out is not None: # CZECK-mARK replace with sklearn shuffesplit
            splitter = sklearn.model_selection.ShuffleSplit(percent_test=self.percent_leave_out)
        else:
            raise ValueError("Both self.num_folds and self.percent_leave_out are None. One or the other must be specified.")

        X = indiv_dh.data[indiv_dh.input_features].values
        y = indiv_dh.data[indiv_dh.target_feature].values
        for train_index, test_index in splitter.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            indiv_model.fit(X_train, y_train)
            y_test_pred = indiv_model.predict(X_test)
            mycv_rmse = sklearn.metrics.mean_squared_error(y_test, y_test_pred)**0.5
            mycv_stats = dict()

        indiv_param_list = self.print_params(indiv_params)
        try:
            os.makedirs(indiv_path) # TODO Why do I need to make this now? I didn't before....
        except FileExistsError:
            pass
        with open(os.path.join(indiv_path,"param_values"), 'w') as indiv_pfile:
            indiv_pfile.writelines(indiv_param_list)
        files_to_clean=list()
        files_to_clean.append("best_and_worst_test_data.csv")
        for cfile in files_to_clean:
            cdir = os.path.join(indiv_path, cfile)
            if os.path.isfile(cdir):
                os.remove(cdir)
        return [mycv_rmse, mycv_stats]

    def get_best_indivs(self):
        how_many = min(self.num_bests, len(self.pop_rmses.keys()))
        if how_many < self.num_bests:
            logger.info("Only %i best values will be returned because population size is limited to %i." % (how_many, how_many))
        largeval=1e10
        lowest = list()
        params = copy.deepcopy(self.pop_params)
        rmses = copy.deepcopy(self.pop_rmses)
        lct=0
        while lct < how_many:
            minval = largeval
            minikey = None
            for ikey in rmses.keys():
                ival = rmses[ikey]
                if ival < minval:
                    minval = ival
                    minikey = ikey
            lowest.append((minikey, rmses[minikey], params[minikey]))
            rmses[minikey]=largeval
            lct = lct + 1
        self.readme_list.append("----Minimum RMSE params----\n")
        for lowitem in lowest:
            self.readme_list.append("%s: %3.3f, %s\n" % (lowitem[0],lowitem[1],lowitem[2]))
        self.readme_list.append("-----------------------\n")
        self.best_indivs = lowest
        self.best_params = self.best_indivs[0][2] #first index, third col
        return

    def print_best_params(self):
        bplist = self.print_params(self.best_params)
        with open(os.path.join(self.save_path,"OPTIMIZED_PARAMS"),'w') as pfile:
            pfile.writelines(bplist)
        return

    def print_params(self, param_dict):
        param_list =list()
        for loc in param_dict.keys():
            for param in param_dict[loc].keys():
                val = param_dict[loc][param]
                paramstr = "%s;%s;%s\n" % (loc, param, val)
                param_list.append(paramstr)
        return param_list

    def save_best_model(self):
        self.model.set_params(**self.best_params['model'])
        self.save_model()
        return

    def print_best_dataframe(self):
        best_dh = copy.deepcopy(self.testing_dataset)
        best_df = self.get_afm_updated_dataset(best_dh.data, self.best_params)
        namestr = "best_params_updated_dataframe.csv"
        updated_name = os.path.join(self.save_path,namestr)
        best_df.to_csv(updated_name)
        self.readme_list.append("Updated dataframe printed as %s\n" % namestr)
        return

    def get_afm_updated_dataset(self, indiv_df, indiv_params):
        """Update dataframe with additional feature methods
        """
        for afm in indiv_params.keys():
            if afm == 'model': #model dealt with separately
                continue
            afm_kwargs = dict(indiv_params[afm])
            (feature_name,feature_data)=cf_help.get_custom_feature_data(afm,
                        starting_dataframe = indiv_df,
                        addl_feature_method_kwargs = dict(afm_kwargs))

            indiv_df[afm] = feature_data
        return indiv_df

    def get_indiv_datahandler(self, indiv_params):
        indiv_dh = copy.deepcopy(self.testing_dataset)
        indiv_dataframe = self.get_afm_updated_dataset(indiv_dh.data, indiv_params)
        indiv_dh.data = indiv_dataframe
        for afm in indiv_params.keys():
            if afm == 'model':
                continue
            indiv_dh.input_features.append(afm) # MARK
        indiv_dh.set_up_data_from_features()
        return indiv_dh

    def get_split_name(self, combined_name):
        """ model.param or package.class.param to
            (model, param) or (package.class, param)
        """
        name_split = combined_name.split(".")
        param_name = name_split.pop(-1) #get and remove
        location = str.join(".", name_split) #join back up
        return (location, param_name)

    def grow_param_dict(self, old_dict, add_combined_name):
        """Increase the parameter dictionary by having each entry
            branch into new entries
        """
        add_vals = self.opt_dict[add_combined_name]
        (location, param_name) = self.get_split_name(add_combined_name)
        if len(old_dict.keys()) == 0:
            new_dict = dict()
            for addct in range(0, len(add_vals)):
                newstr = "%i" % addct
                new_dict[newstr]=dict()
                if not location in new_dict[newstr].keys():
                    new_dict[newstr][location]=dict()
                new_dict[newstr][location][param_name] = add_vals[addct]
        else:
            new_dict = dict()
            for oldstr in old_dict.keys():
                for addct in range(0, len(add_vals)):
                    newstr = "%s_%i" % (oldstr, addct)
                    new_dict[newstr] = dict()
                    for oloc in old_dict[oldstr].keys():
                        if not oloc in new_dict[newstr].keys():
                            new_dict[newstr][oloc] = dict()
                        for oparam in old_dict[oldstr][oloc].keys():
                            new_dict[newstr][oloc][oparam] = old_dict[oldstr][oloc][oparam]
                    if not location in new_dict[newstr].keys():
                        new_dict[newstr][location]=dict()
                    new_dict[newstr][location][param_name] = add_vals[addct]
        return new_dict

    def grow_param_dict_nonopt(self, old_dict, add_combined_name):
        """Add nonoptimized parameters (single value) to parameter
            dictionary.
        """
        add_value = self.opt_dict[add_combined_name]
        (location, param_name) = self.get_split_name(add_combined_name)
        if len(old_dict.keys()) == 0:
            raise ValueError("No optimized parameters in grid search? You shouldn't be using Grid Search, silly.")
        new_dict = dict()
        for oldstr in old_dict.keys():
            newstr = oldstr
            new_dict[newstr] = dict()
            for oloc in old_dict[oldstr].keys():
                if not oloc in new_dict[newstr].keys():
                    new_dict[newstr][oloc] = dict()
                for oparam in old_dict[oldstr][oloc].keys():
                    new_dict[newstr][oloc][oparam] = old_dict[oldstr][oloc][oparam]
            if not location in new_dict[newstr].keys():
                new_dict[newstr][location]=dict()
            new_dict[newstr][location][param_name] = add_value
        return new_dict

    def set_up_pop_params(self):
        self.pop_params=dict()
        for opt_param in self.opt_param_list:
            self.pop_params = self.grow_param_dict(self.pop_params, opt_param)
        for nonopt_param in self.nonopt_param_list:
            self.pop_params = self.grow_param_dict_nonopt(self.pop_params, nonopt_param)
        return

    def set_up_opt_dict(self):
        """Set up parameter value dictionary based on parameter strings, into
            self.opt_dict
            Also sets:
                self.opt_param_list (only those parameters which will be
                                    optimized)
                self.nonopt_param_list
                self.pop_size
        """
        self.opt_dict=dict()
        self.opt_param_list=list()
        self.nonopt_param_list=list()
        pop_size = None

        #print(self.param_strings)

        for paramct, paramstr in enumerate(self.param_strings):
            #print(paramct, paramstr)

            logger.debug(paramstr)
            paramsplit = paramstr.strip().split(";")
            location = paramsplit[0].strip()
            paramname = paramsplit[1].strip()
            paramtype = paramsplit[2].strip().lower()
            if not(paramtype in ['int','float','bool','str','tuple']):
                raise ValueError("Parameter type %s must be one of int, float, bool, str, or tuple. Exiting." % paramtype)
            rangetype = paramsplit[3].strip().lower()
            if not(rangetype in ['discrete','continuous','continuous-log']):
                raise ValueError("Range type %s must be 'discrete' or 'continuous' or 'continuous-log'. Exiting." % rangetype)
            if paramtype in ['bool', 'str']:
                if not(rangetype in ['discrete']):
                    raise ValueError("Range type %s must be discrete for parameter type of %s" % (rangetype, paramtype))
            gridinfo = paramsplit[4].strip()
            combined_name = "%s.%s" % (location, paramname)
            if combined_name in self.opt_dict.keys():
                raise KeyError("Parameter %s for optimization of %s appears to be listed twice. Exiting." % (paramname, location))
            #print(gridinfo)
            #print(paramtype)
            gridsplit = gridinfo.split(":") #split colon-delimited
            #print(gridsplit)
            if paramtype == 'tuple':
                gridsplit = [int(i) for i in gridsplit]
                gridsplit = np.array(gridsplit)
            else:
                gridsplit = np.array(gridsplit, paramtype)
            #print(gridsplit)
            #print(gridsplit.shape, type(gridsplit))
            if rangetype == 'discrete':
                gridvals = gridsplit
                #print(gridvals)
            elif rangetype == 'continuous':
                if paramtype == 'tuple':
                    meshes = list()
                    if gridsplit.shape[0]%3 != 0:
                        raise ValueError("Tuple designation should appear in groups of 3 as start:end:spacing for each value in the tuple."
                                         "For example, to denote a tuple of hidden layer sizes ranging from (20, 2) to (200, 10) with spacings of 5,"
                                         "use 20:200:5:2:10;5")
                    num_tuple = int(gridsplit.shape[0]/3)
                    for i in range(num_tuple):
                        meshes.append((list(np.linspace(gridsplit[i+i*(num_tuple-1)],gridsplit[i+1+i*(num_tuple-1)],gridsplit[i+2+i*(num_tuple-1)], dtype='int'))))
                    #print('meshes')
                    #print(meshes)
                    gridvals = list()
                    for mesh in meshes:
                        other_meshes = meshes
                        other_meshes.remove(mesh)
                        tup = list(itertools.product(mesh, *other_meshes))
                        gridvals.append(tup)
                    gridvals = gridvals[0]
                    for tup in gridvals:
                        #print(tup, len(tup))
                        if len(tup) != num_tuple:
                            gridvals.remove(tup)
                    #print(gridvals)
                else:
                    gridvals = np.linspace(start=gridsplit[0],
                                        stop=gridsplit[1],
                                        num=gridsplit[2],
                                        endpoint=True,
                                        dtype=paramtype)
            elif rangetype == 'continuous-log':
                gridvals = np.logspace(start=gridsplit[0],
                                        stop=gridsplit[1],
                                        num=gridsplit[2],
                                        endpoint=True,
                                        dtype=paramtype)
            #print(gridvals)
            numvals = len(gridvals)
            if numvals < 1:
                raise ValueError("Parameter %s does not evaluate to having any values." % paramstr)
            elif numvals > 1: #optimization to be done
                self.opt_param_list.append(combined_name)
                self.opt_dict[combined_name] = gridvals
            elif numvals == 1: #
                self.nonopt_param_list.append(combined_name)
                self.opt_dict[combined_name] = gridvals[0]
            if pop_size is None:
                pop_size = numvals
            else:
                pop_size = pop_size * numvals
            if pop_size > self.pop_upper_limit:
                raise ValueError("Population size has reached the upper limit of %i. Exiting." % self.pop_upper_limit)
        self.pop_size = pop_size
        return

    def plot(self):
        self.readme_list.append("----- Plotting -----\n")
        logger.debug(list)
        cols=list() #repeated code; may want to reduce
        for opt_param in self.opt_param_list:
            self.plot_single_rmse(opt_param)
        if len(self.opt_param_list) == 2:
            self.plot_2d_rmse_heatmap(self.opt_param_list)
        elif len(self.opt_param_list) == 3:
            self.plot_3d_rmse_heatmap(self.opt_param_list)
        return

    def is_log_param(self, col):
        """Check to see if flattened column was a log parameter
        """
        location=col.split(".")[0]
        param=col.split(".")[1]
        for init_param in self.param_strings:
            if location in init_param and param in init_param and 'log' in init_param:
                return True
        return False

    def plot_3d_rmse_heatmap(self, cols):
        #adjust for log params if necessary
        xcol = cols[0]
        ycol = cols[1]
        zcol = cols[2]
        xdata = self.flat_results[xcol]
        xlabel = xcol
        if self.is_log_param(xcol):
            xdata_raw = np.array(self.flat_results[xcol].values,'float')
            xdata = np.log10(xdata_raw)
            xlabel = "log10 %s" % xcol
        ydata = self.flat_results[ycol]
        ylabel = ycol
        if self.is_log_param(ycol):
            ydata_raw = np.array(self.flat_results[ycol].values,'float')
            ydata = np.log10(ydata_raw)
            ylabel = "log10 %s" % ycol
        zdata = self.flat_results[zcol]
        zlabel = zcol
        if self.is_log_param(zcol):
            zdata_raw = np.array(self.flat_results[zcol].values,'float')
            zdata = np.log10(zdata_raw)
            zlabel = "log10 %s" % zcol

        plotlabel = 'rmse_heatmap_3d'
        savepath = os.path.join(self.save_path, plotlabel+'.png')
        if isinstance(xdata[0], str):
            strings = list(set(xdata))
            mapping = {string: i for i,string in enumerate(strings)}
            xdata = xdata.map(lambda s: mapping[s])
        plot_helper.plot_3d_heatmap(xdata, ydata, zdata, self.flat_results['rmse'], savepath,
                                    xlabel, ylabel, zlabel, 'rmse')
        self.readme_list.append("Plot %s.png created\n" % plotlabel)

    def plot_2d_rmse_heatmap(self, cols):
        #adjust for log params if necessary
        xcol = cols[0]
        ycol = cols[1]
        xdata = np.array(self.flat_results[xcol])
        xlabel = xcol
        if self.is_log_param(xcol):
            xdata_raw = np.array(self.flat_results[xcol].values,'float')
            xdata = np.log10(xdata_raw)
            xlabel = "log10 %s" % xcol
        ydata = np.array(self.flat_results[ycol])
        ylabel = ycol
        if self.is_log_param(ycol):
            ydata_raw = np.array(self.flat_results[ycol].values,'float')
            ydata = np.log10(ydata_raw)
            ylabel = "log10 %s" % ycol

        plotlabel = "rmse_heatmap"
        savepath = os.path.join(self.save_path, f'{plotlabel}.png')
        plot_helper.plot_2d_heatmap(xdata, ydata, self.flat_results['rmse'], savepath, xlabel, ylabel, 'rmse')
        self.readme_list.append("Plot %s.png created\n" % plotlabel)
        return

    def plot_single_rmse(self, col):
        #adjust for log params if necessary
        xdata = self.flat_results[col]
        xlabel = col
        if self.is_log_param(col):
            import numpy as np
            xdata_raw = np.array(self.flat_results[col].values,'float')
            xdata = np.log10(xdata_raw)
            xlabel = "log10 %s" % col

        #ph_kwargs = dict()
        #ph_kwargs['xlabel'] = xlabel
        #ph_kwargs['ylabel'] = 'RMSE'
        #ph_kwargs['labellist'] = [xlabel]
        #ph_kwargs['xdatalist'] = [xdata]
        #ph_kwargs['ydatalist'] = [self.flat_results['rmse']]
        #ph_kwargs['xerrlist'] = list([None])
        #ph_kwargs['yerrlist'] = list([None])
        #ph_kwargs['notelist'] = list()
        #ph_kwargs['guideline'] = 0
        #plotlabel=plotlabel.replace(".","_") #mask periods
        #ph_kwargs['plotlabel'] = plotlabel
        #ph_kwargs['save_path'] = self.save_path
        #myph = PlotHelper(**ph_kwargs)
        #myph.multiple_overlay()
        #self.readme_list.append("Plot %s.png created\n" % plotlabel)

        plotlabel="rmse_vs_%s" % col
        savepath = os.path.join(self.save_path, f'{plotlabel}.png')
        plot_helper.plot_1d_heatmap(xdata, self.flat_results['rmse'], savepath)
        self.readme_list.append("Plot %s.png created\n" % plotlabel)
        return

    def flatten_results(self):
        """Flatten results into a csv
        """
        cols=list()
        for opt_param in self.opt_param_list:
            cols.append(opt_param)
        cols.append('rmse')
        cols.append('key')
        flat_results = pd.DataFrame(index=range(0, self.pop_size), columns=cols)
        pct = 0
        for pkey in self.pop_params.keys():
            params = self.pop_params[pkey]
            rmse = self.pop_rmses[pkey]
            for loc in params.keys():
                for param in params[loc].keys():
                    colname = "%s.%s" % (loc, param)
                    val = params[loc][param]
                    #Need to make tuples into single element to insert in df
                    if type(val) is tuple:
                        val_cat = ''
                        for v in val:
                            val_cat += str(v)+','
                        val = val_cat
                    flat_results.loc[pct, colname] = val
            flat_results.loc[pct, 'rmse'] = rmse
            flat_results.loc[pct, 'key'] = pkey
            pct = pct + 1
        flat_results.to_csv(os.path.join(self.save_path, "results.csv"))
        self.readme_list.append("Printed RMSE results to results.csv\n")
        self.flat_results = flat_results
        return
