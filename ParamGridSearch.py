import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from KFoldCV import KFoldCV
from LeaveOutPercentCV import LeaveOutPercentCV
from plot_data.PlotHelper import PlotHelper
from SingleFit import SingleFit
from SingleFit import timeit
from custom_features import cf_help
from FeatureOperations import FeatureIO
from DataHandler import DataHandler
import pandas as pd
import copy
import logging
logger = logging.getLogger()

class ParamGridSearch(SingleFit):
    """Parameter optimization by grid search
        Only 4 parameters may be optimized at a time.
   
    Args:
        training_dataset, (Should be the same as testing_dataset)
        testing_dataset, (Should be the same as training_dataset)
        model,
        save_path,
        xlabel, 
        ylabel, see parent class.
        param_1 <str>: parameter string made up of semicolon-delimited pieces
            Piece 1: The word 'model' or a custom feature class.method string, 
                        e.g. DBTT.calculate_EffectiveFluence,
                        where the custom module has the same name as the 
                        custom class, and resides inside 
                        the custom_features folder
            Piece 2: The parameter name
            Piece 3: The parameter type. Use only:
                        'int', 'float' (boolean and string not supported)
                        (Some sklearn model
                        hyperparameters are type-sensitive)
            Piece 4: The series type. Use:
                        'discrete': List will be given in piece 5
                        'continuous': Range and step will be given in piece 5
                        'continuous-log: Range and step will be given in piece 5
            Piece 5: A colon-delimited list of 
                    (a) a discrete list of values to grid over, OR
                    (b) start, end, number of points: 
                        numpy's np.linspace or np.logspace
                        function will be used to generate this list,
                        using an inclusive start and inclusive end
        param_2 <str>
        param_3 <str>
        param_4 <str>
        fix_random_for_testing <int>: 0 - use random numbers
                                      1 - fix randomizer for testing
        num_cvtests <int>: Number of CV tests for each validation step
        additional_feature_methods <str>: comma-delimited string, or
            a list, of semicolon-delimited pieces, formatted like:
            class.method;parameter1:value1;parameter2:value2;... 
            These values will be passed on and not optimized.
        mark_outlying_points <int>: See KFoldCV
    Returns:
        Analysis in the save_path folder
        Plots results in a predicted vs. measured square plot.
    Raises:
        ValueError if testing target data is None; CV must have
                testing target data
    """
    def __init__(self, 
        training_dataset=None,
        testing_dataset=None,
        model=None,
        save_path=None,
        xlabel="Measured",
        ylabel="Predicted",
        param_1=None,
        param_2=None,
        param_3=None,
        param_4=None,
        additional_feature_methods=None,
        fix_random_for_testing=0,
        num_cvtests=5,
        mark_outlying_points='0,3',
        num_folds=None,
        percent_leave_out=None,
        processors=1,
        *args, **kwargs):
        """
        Additional class attributes to parent class:
            Set by keyword:
                self.param_1
                self.param_2
                self.param_3
                self.param_4
                self.additional_feature_methods
                self.fix_random_for_testing
                self.num_cvtests
                self.mark_outlying_points
                self.num_folds
                self.percent_leave_out
                self.processors
            Set in code:
                self.opt_dict
                self.afm_dict
                self.pop_params
                self.pop_size
                self.pop_stats
                self.pop_rmses
                self.pop_upper_limit
                ?self.random_state
        """
        if not(training_dataset == testing_dataset):
            raise ValueError("Only testing_dataset will be used. Use the same values for training_dataset and testing_dataset")
        SingleFit.__init__(self, 
            training_dataset=training_dataset, #only testing_dataset is used
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
        self.param_1 = param_1
        self.param_2 = param_2
        self.param_3 = param_3
        self.param_4 = param_4
        if type(additional_feature_methods) is list:
            self.additional_feature_methods = list(additional_feature_methods)
        elif type(additional_feature_methods) is str:
            self.additional_feature_methods = additional_feature_methods.split(",")
        else:
            self.additional_feature_methods = additional_feature_methods
        self.num_cvtests = int(num_cvtests)
        self.mark_outlying_points = mark_outlying_points
        self.num_folds = num_folds
        self.percent_leave_out = percent_leave_out
        self.processors=int(processors)
        # Sets later in code
        self.opt_dict=None
        self.afm_dict=None
        self.pop_params=None
        self.pop_size=None
        self.pop_stats=None
        self.pop_rmses=None
        self.pop_upper_limit=1e6
        return 

    @timeit
    def run(self):
        self.set_up()
        self.evaluate_pop()
        self.get_best_indivs()
        self.print_readme()
        return
    @timeit
    def set_up(self):
        SingleFit.set_up(self)
        self.set_up_opt_dict()
        logger.debug("opt dict: %s" % self.opt_dict)
        self.set_up_afm_dict()
        logger.debug("afm dict: %s" % self.afm_dict)
        self.set_up_pop_params()
        logger.debug("Population size: %i" % len(self.pop_params))
        return
    


    @timeit
    def evaluate_pop(self):
        """make model and new testing dataset for each pop member
            and evaluate
        """
        self.pop_stats=list()
        self.pop_rmses=list()
        for pidx in range(0, self.pop_size):
            indiv_params = self.pop_params[pidx]
            indiv_model = copy.deepcopy(self.model)
            indiv_model.set_params(**indiv_params['model'])
            indiv_dh = self.get_indiv_datahandler(indiv_params)
            #logging.debug(indiv_dh)
            indiv_path = os.path.join(self.save_path, "indiv_%i" % pidx)
            if not(self.num_folds is None):
                mycv = KFoldCV(training_dataset= indiv_dh,
                        testing_dataset= indiv_dh,
                        model = indiv_model,
                        save_path = indiv_path,
                        xlabel = self.xlabel,
                        ylabel = self.ylabel,
                        mark_outlying_points = self.mark_outlying_points,
                        num_cvtests = self.num_cvtests,
                        fix_random_for_testing = self.fix_random_for_testing,
                        num_folds = self.num_folds)
                mycv.run()
                self.pop_rmses.append(mycv.statistics['avg_fold_avg_rmses'])
            elif not (self.percent_leave_out is None):
                mycv = LeaveOutPercentCV(training_dataset= indiv_dh,
                        testing_dataset= indiv_dh,
                        model = indiv_model,
                        save_path = indiv_path,
                        xlabel = self.xlabel,
                        ylabel = self.ylabel,
                        mark_outlying_points = self.mark_outlying_points,
                        num_cvtests = self.num_cvtests,
                        fix_random_for_testing = self.fix_random_for_testing,
                        percent_leave_out = self.percent_leave_out)
                mycv.run()
                self.pop_rmses.append(mycv.statistics['avg_rmse'])
            else:
                raise ValueError("Both self.num_folds and self.percent_leave_out are None. One or the other must be specified.")
            with open(os.path.join(indiv_path,"param_values"), 'w') as indiv_pfile:
                for loc in indiv_params.keys():
                    for param in indiv_params[loc].keys():
                        val = indiv_params[loc][param]
                        indiv_pfile.write("%s, %s: %s\n" % (loc, param, val))
            self.pop_stats.append(mycv.statistics)
        return

    def get_best_indivs(self):
        how_many = min(10, len(self.pop_rmses))
        largeval=1e10
        lowest = list()
        params = copy.deepcopy(self.pop_params)
        rmses = copy.deepcopy(self.pop_rmses)
        lct=0
        while lct < how_many:
            minidx = np.argmin(rmses)
            lowest.append((minidx, rmses[minidx], params[minidx]))
            rmses[minidx]=largeval
            lct = lct + 1
        self.readme_list.append("----Minimum RMSE params----\n")
        for lowitem in lowest:
            self.readme_list.append("%s: %3.3f, %s\n" % (lowitem[0],lowitem[1],lowitem[2]))
        self.readme_list.append("-----------------------\n")
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
            fio = FeatureIO(indiv_df)
            indiv_df = fio.add_custom_features([afm], feature_data)
        return indiv_df

    def get_indiv_datahandler(self, indiv_params):
        indiv_dh = copy.deepcopy(self.testing_dataset)
        indiv_dataframe = self.get_afm_updated_dataset(indiv_dh.data, indiv_params)
        indiv_dh.data = indiv_dataframe
        for afm in indiv_params.keys():
            if afm == 'model':
                continue
            indiv_dh.input_features.append(afm)
        indiv_dh.set_up_data_from_features()
        return indiv_dh

    def set_up_pop_params(self):
        self.pop_params=dict()
        flat_params=list()
        for location in self.opt_dict.keys():
            for param in self.opt_dict[location].keys():
                paramlist=list()
                for val in self.opt_dict[location][param]:
                    paramlist.append([location,param,val])
                flat_params.append(paramlist)
        logger.debug("Flattened:")
        for flat_item in flat_params:
            logger.debug(flat_item)
        num_params = len(flat_params)
        pop_size=1
        for fplist in flat_params:
            pop_size = pop_size * len(fplist)
        if pop_size > self.pop_upper_limit:
            raise ValueError("Over %i grid points. Exiting.")
        pct = 0
        for aidx in range(0, len(flat_params[0])):
            alocation = flat_params[0][aidx][0]
            aparam = flat_params[0][aidx][1]
            aval = flat_params[0][aidx][2]
            single_dict=dict()
            single_dict[alocation]=dict()
            single_dict[alocation][aparam] = aval
            if num_params > 1:
                for bidx in range(0, len(flat_params[1])):
                    blocation = flat_params[1][bidx][0]
                    bparam = flat_params[1][bidx][1]
                    bval = flat_params[1][bidx][2]
                    if not blocation in single_dict.keys():
                        single_dict[blocation]=dict()
                    single_dict[blocation][bparam] = bval
                    if num_params > 2:
                        for cidx in range(0, len(flat_params[2])):
                            clocation = flat_params[2][cidx][0]
                            cparam = flat_params[2][cidx][1]
                            cval = flat_params[2][cidx][2]
                            if not clocation in single_dict.keys():
                                single_dict[clocation]=dict()
                            single_dict[clocation][cparam] = cval
                            if num_params > 3:
                                for didx in range(0, len(flat_params[3])):
                                    dlocation = flat_params[3][didx][0]
                                    dparam = flat_params[3][didx][1]
                                    dval = flat_params[3][didx][2]
                                    if not dlocation in single_dict.keys():
                                        single_dict[dlocation]=dict()
                                    single_dict[dlocation][dparam] = dval
                                    if num_params > 4:
                                        raise ValueError("Too many params")
                                    else:
                                        self.pop_params[pct]=copy.deepcopy(single_dict)
                                        pct = pct + 1

                            else:
                                self.pop_params[pct]=copy.deepcopy(single_dict)
                                pct = pct +1
                    else:
                        self.pop_params[pct]=copy.deepcopy(single_dict)
                        pct = pct + 1
            else:
                self.pop_params[pct]=copy.deepcopy(single_dict)
                pct = pct + 1
        self.pop_size = pct
        if not(self.pop_size == pop_size):
            raise ValueError("Flat population size does not match dictionary population size. Exiting.")
        for noct in range(0, self.pop_size):
            for afm_loc in self.afm_dict.keys():
                if not afm_loc in self.pop_params[noct].keys():
                    self.pop_params[noct][afm_loc] = dict()
                for afm_param in self.afm_dict[afm_loc].keys():
                    if afm_param in self.pop_params[noct][afm_loc].keys():
                        raise ValueError("Parameter %s for module %s appears twice. Exiting." % (afm_param, afm_loc))
                    self.pop_params[noct][afm_loc][afm_param] = self.afm_dict[afm_loc][afm_param]
        #logger.debug(self.pop_params)
        return

    def set_up_opt_dict(self):
        self.opt_dict=dict()
        params = list()
        if not (self.param_1 is None):
            params.append(self.param_1)
        if not (self.param_2 is None):
            params.append(self.param_2)
        if not (self.param_3 is None):
            params.append(self.param_3)
        if not (self.param_4 is None):
            params.append(self.param_4)
        if len(params) == 0:
            raise ValueError("No parameters to optimize. Exiting")
        for paramstr in params:
            logger.debug(paramstr)
            paramsplit = paramstr.strip().split(";")
            location = paramsplit[0].strip()
            paramname = paramsplit[1].strip()
            paramtype = paramsplit[2].strip().lower()
            if not(paramtype in ['int','float']):
                raise ValueError("Parameter type %s must be 'int' or 'float'. Exiting." % paramtype)
            rangetype = paramsplit[3].strip().lower()
            if not(rangetype in ['discrete','continuous','continuous-log']):
                raise ValueError("Range type %s must be 'discrete' or 'continuous' or 'continuous-log'. Exiting." % rangetype)
            gridinfo = paramsplit[4].strip()
            if not location in self.opt_dict.keys():
                self.opt_dict[location] = dict()
            if paramname in self.opt_dict[location].keys():
                raise KeyError("Parameter %s for optimization of %s appears to be listed twice. Exiting." % (paramname, location))
            gridsplit = gridinfo.split(":") #split colon-delimited
            gridsplit = np.array(gridsplit, paramtype)
            if rangetype == 'discrete':
                gridvals = gridsplit
            elif rangetype == 'continuous':
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
            self.opt_dict[location][paramname] = gridvals
        return
    
    def set_up_afm_dict(self):
        self.afm_dict=dict()
        if self.additional_feature_methods is None:
            return
        for paramstr in self.additional_feature_methods:
            logger.debug(paramstr)
            paramsplit = paramstr.strip().split(";")
            location = paramsplit[0].strip()
            if not location in self.afm_dict.keys():
                self.afm_dict[location]=dict()
            for argidx in range(1, len(paramsplit)):
                argitem = paramsplit[argidx]
                paramname = argitem.split(":")[0]
                paramval = argitem.split(":")[1]
                self.afm_dict[location][paramname]=paramval
        return


    @timeit
    def fit(self):
        self.cv_fit_and_predict()
        return

    @timeit
    def predict(self):
        #Predictions themselves are covered in self.fit()
        self.get_statistics()
        self.print_statistics()
        self.readme_list.append("----- Output data -----\n")
        self.print_best_worst_output_csv("best_and_worst")
        return

    @timeit
    def plot(self):
        self.readme_list.append("----- Plotting -----\n")
        notelist=list()
        notelist.append("Mean over %i LO %i%% tests:" % (self.num_cvtests, self.percent_leave_out))
        notelist.append("RMSE:")
        notelist.append("    {:.2f} $\pm$ {:.2f}".format(self.statistics['avg_rmse'], self.statistics['std_rmse']))
        notelist.append("Mean error:")
        notelist.append("    {:.2f} $\pm$ {:.2f}".format(self.statistics['avg_mean_error'], self.statistics['std_mean_error']))
        self.plot_best_worst_overlay(notelist=list(notelist))
        return

    def set_up_cv(self):
        if self.testing_dataset.target_data is None:
            raise ValueError("Testing target data cannot be none for cross validation.")
        indices = np.arange(0, len(self.testing_dataset.target_data))
        self.readme_list.append("----- CV setup -----\n")
        self.readme_list.append("%i CV tests,\n" % self.num_cvtests)
        self.readme_list.append("leaving out %i percent\n" % self.percent_leave_out)
        test_fraction = self.percent_leave_out / 100.0
        self.cvmodel = ShuffleSplit(n_splits = 1, 
                            test_size = test_fraction, 
                            random_state = None)
        for cvtest in range(0, self.num_cvtests):
            self.cvtest_dict[cvtest] = dict()
            for train, test in self.cvmodel.split(indices):
                fdict=dict()
                fdict['train_index'] = train
                fdict['test_index'] = test
                self.cvtest_dict[cvtest]= dict(fdict)
        return
    
    def cv_fit_and_predict(self):
        for cvtest in self.cvtest_dict.keys():
            prediction_array = np.zeros(len(self.testing_dataset.target_data))
            prediction_array[:] = np.nan
            fdict = self.cvtest_dict[cvtest]
            input_train=self.testing_dataset.input_data.iloc[fdict['train_index']]
            target_train = self.testing_dataset.target_data[fdict['train_index']]
            input_test=self.testing_dataset.input_data.iloc[fdict['test_index']]
            target_test = self.testing_dataset.target_data[fdict['test_index']]
            fit = self.model.fit(input_train, target_train)
            predict_test = self.model.predict(input_test)
            rmse = np.sqrt(mean_squared_error(predict_test, target_test))
            merr = np.mean(predict_test - target_test)
            prediction_array[fdict['test_index']] = predict_test
            self.cvtest_dict[cvtest]["rmse"] = rmse
            self.cvtest_dict[cvtest]["mean_error"] = merr
            self.cvtest_dict[cvtest]["prediction_array"] = prediction_array
        return

    def get_statistics(self):
        cvtest_rmses = list()
        cvtest_mean_errors = list()
        for cvtest in range(0, self.num_cvtests):
            cvtest_rmses.append(self.cvtest_dict[cvtest]["rmse"])
            cvtest_mean_errors.append(self.cvtest_dict[cvtest]["mean_error"])
        highest_rmse = max(cvtest_rmses)
        self.worst_test_index = cvtest_rmses.index(highest_rmse)
        lowest_rmse = min(cvtest_rmses)
        self.best_test_index = cvtest_rmses.index(lowest_rmse)
        self.statistics['avg_rmse'] = np.mean(cvtest_rmses)
        self.statistics['std_rmse'] = np.std(cvtest_rmses)
        self.statistics['avg_mean_error'] = np.mean(cvtest_mean_errors)
        self.statistics['std_mean_error'] = np.std(cvtest_mean_errors)
        self.statistics['rmse_best'] = lowest_rmse
        self.statistics['rmse_worst'] = highest_rmse
        return

    def print_best_worst_output_csv(self, label=""):
        """
        """
        olabel = "%s_test_data.csv" % label
        ocsvname = os.path.join(self.save_path, olabel)
        self.testing_dataset.add_feature("Best Prediction", 
                    self.cvtest_dict[self.best_test_index]['prediction_array'])
        self.testing_dataset.add_feature("Worst Prediction", 
                    self.cvtest_dict[self.worst_test_index]['prediction_array'])
        cols = self.testing_dataset.print_data(ocsvname, ["Best Prediction", "Worst Prediction"])
        self.readme_list.append("%s file created with columns:\n" % olabel)
        for col in cols:
            self.readme_list.append("    %s\n" % col)
        return


    def plot_best_worst_overlay(self, notelist=list()):
        kwargs2 = dict()
        kwargs2['xlabel'] = self.xlabel
        kwargs2['ylabel'] = self.ylabel
        kwargs2['labellist'] = ["Best test","Worst test"]
        kwargs2['xdatalist'] = list([self.testing_dataset.target_data, 
                            self.testing_dataset.target_data])
        kwargs2['ydatalist'] = list(
                [self.cvtest_dict[self.best_test_index]['prediction_array'],
                self.cvtest_dict[self.worst_test_index]['prediction_array']])
        kwargs2['xerrlist'] = list([None,None])
        kwargs2['yerrlist'] = list([None,None])
        kwargs2['notelist'] = list(notelist)
        kwargs2['guideline'] = 1
        kwargs2['plotlabel'] = "best_worst_overlay"
        kwargs2['save_path'] = self.save_path
        if not (self.mark_outlying_points is None):
            kwargs2['marklargest'] = self.mark_outlying_points
            if self.testing_dataset.labeling_features is None:
                raise ValueError("Must specify some labeling features if you want to mark the largest outlying points")
            labels = self.testing_dataset.data[self.testing_dataset.labeling_features[0]]
            kwargs2['mlabellist'] = list([labels,labels])
        myph = PlotHelper(**kwargs2)
        myph.multiple_overlay()
        self.readme_list.append("Plot best_worst_overlay.png created,\n")
        self.readme_list.append("    showing the best and worst of %i tests.\n" % self.num_cvtests)
        return

