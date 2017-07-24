import os
import numpy as np
from sklearn.metrics import mean_squared_error
from plot_data.PlotHelper import PlotHelper
from SingleFit import SingleFit
from SingleFit import timeit

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
            Piece 5: A comma-delimited list of 
                    (a) a discrete list of values to grid over, OR
                    (b) start, end, number of points: numpy's np.linspace 
                        function will be used to generate this list,
                        using an inclusive start and inclusive end
        param_2 <str>
        param_3 <str>
        param_4 <str>
        fix_random_for_testing <int>: 0 - use random numbers
                                      1 - fix randomizer for testing
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
        *args, **kwargs):
        """
        Additional class attributes to parent class:
            Set by keyword:
            self.param_1
            self.param_2
            self.param_3
            self.param_4
            self.fix_random_for_testing
            Set in code:
            self.opt_dict
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
        if int(fix_random_for_testing) == 1:
            np.random.seed(0)
        self.param_1 = param_1
        self.param_2 = param_2
        self.param_3 = param_3
        self.param_4 = param_4
        # Sets later in code
        self.opt_dict=None
        return 

    @timeit
    def run(self):
        self.set_up()
        return
    @timeit
    def set_up(self):
        SingleFit.set_up(self)
        self.set_up_opt_dict()
        self.set_up_cv()
        return
    
    def set_up_opt_dict(self):
        params = list()
        params.append(self.param_1)
        params.append(self.param_2)
        params.append(self.param_3)
        params.append(self.param_4)
        for paramstr in params:
            paramsplit = paramstr.strip().split(";")
            location = paramsplit[0].strip()
            paramname = paramsplit[1].strip()
            paramtype = paramsplit[2].strip().lower()
            if not(paramtype in ['int','float']):
                raise ValueError("Parameter type %s must be 'int' or 'float'. Exiting." % paramtype)
            rangetype = paramsplit[3].strip().lower()
            if not(rangetype in ['discrete','continuous']):
                raise ValueError("Range type %s must be 'discrete' or 'continuous'. Exiting." % rangetype)
            gridinfo = paramsplit[4].strip()
            if not location in self.opt_dict.keys():
                self.opt_dict[location] = dict()
            if paramname in self.opt_dict[location].keys():
                raise KeyError("Parameter %s for optimization of %s appears to be listed twice. Exiting." % (paramname, location))
            if rangetype == 'discrete':
                gridvals = gridinfo.split(",") #split comma-delimited
                gridvals = np.array(gridvals, paramtype)
            elif rangetype == 'continuous':
                gridbounds = gridinfo.split(",")
                gridvals = np.linspace(start=gridbounds[0],
                                        stop=gridbounds[1],
                                        num=gridbounds[2],
                                        endpoint=True,
                                        dtype=paramtype)
            self.opt_dict[location][paramname] = gridvals
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

