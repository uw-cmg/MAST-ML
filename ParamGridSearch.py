import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from plot_data.PlotHelper import PlotHelper
from SingleFit import SingleFit
from SingleFit import timeit
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
        fix_random_for_testing=0,
        num_cvtests=5,
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
                self.fix_random_for_testing
                self.num_cvtests
                self.num_folds
                self.percent_leave_out
                self.processors
            Set in code:
                self.opt_dict
                self.pop_params
                ?self.grid_dict
                ?self.cv_divisions
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
        self.num_cvtests = int(num_cvtests)
        self.num_folds = num_folds
        self.percent_leave_out = percent_leave_out
        self.processors=int(processors)
        # Sets later in code
        self.opt_dict=None
        self.pop_params=None
        self.cv_divisions=dict()
        return 

    @timeit
    def run(self):
        self.set_up()
        return
    @timeit
    def set_up(self):
        SingleFit.set_up(self)
        self.set_up_opt_dict()
        self.set_up_pop_params()
        logger.debug("Population size: %i" % len(self.pop_params))
        #self.cv_divisions = self.get_cv_divisions(self.num_cvtests)
        #self.set_up_cv()
        return
    


    @timeit
    def run_grid(self):
        """make model and new testing dataset for each pop member"""
        return

    def get_updated_dataset(self):
        return updated_dataset

    def set_up_pop_params(self):
        self.pop_params=dict()
        flat_params=list()
        for location in self.opt_dict.keys():
            for param in self.opt_dict[location].keys():
                paramlist=list()
                for val in self.opt_dict[location][param]:
                    paramlist.append([location,param,val])
                flat_params.append(paramlist)
        logger.debug(flat_params)
        num_params = len(flat_params)
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
                                        self.pop_params[pct]=dict(single_dict)
                                        pct = pct + 1

                            else:
                                self.pop_params[pct]=dict(single_dict)
                                pct = pct +1
                    else:
                        self.pop_params[pct]=dict(single_dict)
                        pct = pct + 1
            else:
                self.pop_params[pct]=dict(single_dict)
                pct = pct + 1
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

    def get_cv_divisions(self, num_runs=0):
        """Preset all CV divisions so that all individuals in all
            generations have the same CV divisions.
            Presetting division ensures that comparisons are consistent,
            with only the genome changing between individual evaluations.
        """
        cv_divisions = list()
        if not(self.num_folds is None):
            for kn in range(num_runs):
                kf = KFold(n_splits=self.num_folds, shuffle=True, random_state = self.random_state)
                splits = kf.split(range(len(self.testing_dataset.target_data)))
                sdict=dict()
                sdict['train_index'] = list()
                sdict['test_index'] = list()
                for train, test in splits:
                    sdict['train_index'].append(train)
                    sdict['test_index'].append(test)
                cv_divisions.append(dict(sdict))
        elif not(self.percent_leave_out is None):
            test_fraction = self.percent_leave_out / 100.0
            for kn in range(num_runs):
                plo = ShuffleSplit(n_splits=1, 
                                test_size = test_fraction,
                                random_state = self.random_state)
                splits = plo.split(range(len(self.testing_dataset.target_data)))
                sdict=dict()
                sdict['train_index'] = list()
                sdict['test_index'] = list()
                for train, test in splits:
                    sdict['train_index'].append(train)
                    sdict['test_index'].append(test)
                cv_divisions.append(dict(sdict))
        else:
            raise ValueError("Neither percent_leave_out nor num_folds appears to be set.")
        return cv_divisions

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

