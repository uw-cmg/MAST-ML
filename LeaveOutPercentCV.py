import os
import matplotlib
import numpy as np
import data_parser
import matplotlib.pyplot as plt
from mean_error import mean_error
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
import data_analysis.printout_tools as ptools
import plot_data.plot_predicted_vs_measured as plotpm
import plot_data.plot_xy as plotxy
from SingleFit import SingleFit
from SingleFit import timeit
from sklearn.metrics import r2_score

class LeaveOutPercentCV(SingleFit):
    """Leave out percent cross validation
   
    Args:
        training_dataset, (Should be the same as testing_dataset)
        testing_dataset, (Should be the same as training_dataset)
        model,
        save_path,
        input_features,
        target_feature,
        target_error_feature,
        labeling_features, 
        xlabel, 
        ylabel,
        stepsize, see parent class.
        mark_outlying_points <list of int>: Number of outlying points to mark in best and worst tests, e.g. [0,3]
        percent_leave_out <int>: Percent to leave out
        num_cvtests <int>: Number of CV tests (K folds in a KFoldCV is 1 test)
        fix_random_for_testing <int>: 1- fix random shuffle for testing purposes
                                      0 (default) Use random shuffle
 
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
        input_features=None,
        target_feature=None,
        target_error_feature=None,
        labeling_features=None,
        xlabel="Measured",
        ylabel="Predicted",
        stepsize=1,
        mark_outlying_points=None,
        percent_leave_out=20,
        num_cvtests=10,
        fix_random_for_testing=0,
        *args, **kwargs):
        """
        Additional class attributes to parent class:
            Set by keyword:
            self.mark_outlying_points <list of int>: Number of outlying points in best and worst fits to mark
            self.percent_leave_out <int>: Percent to leave out of training set
                                    (becomes the percent that is tested on)
            self.num_cvtests <int>: Number of KFold tests to perform
            Set in code:
            self.cvmodel <sklearn CV model>: Cross-validation model
            self.cvtest_dict <dict>: Dictionary of CV test information and
                                        results, one entry for each cvtest.
            self.best_test_index <int>: Test number of best test
            self.worst_test_index <int>: Test number of worst test
        """
        if not(training_dataset == testing_dataset):
            raise ValueError("Only testing_dataset will be used. Use the same values for training_dataset and testing_dataset")
        SingleFit.__init__(self, 
            training_dataset=training_dataset, #only testing_dataset is used
            testing_dataset=testing_dataset,
            model=model, 
            save_path = save_path,
            input_features = input_features, 
            target_feature = target_feature,
            target_error_feature = target_error_feature,
            labeling_features = labeling_features,
            xlabel=xlabel,
            ylabel=ylabel,
            stepsize=stepsize)
        self.mark_outlying_points = mark_outlying_points
        self.percent_leave_out = int(percent_leave_out)
        self.num_cvtests = int(num_cvtests)
        if int(fix_random_for_testing) == 1:
            np.random.seed(0)
        # Sets later in code
        self.cvmodel = None
        self.cvtest_dict=dict()
        self.best_test_index = None
        self.worst_test_index = None
        return 

    @timeit
    def set_up(self):
        SingleFit.set_up(self)
        self.set_up_cv()
        return
    @timeit
    def fit(self):
        self.cv_fit_and_predict()
        return

    def predict(self):
        #Predictions themselves are covered in self.fit()
        self.get_statistics()
        self.print_statistics()
        self.readme_list.append("----- Output data -----\n")
        self.print_output_csv(label="best", cvtest_entry=self.cvtest_dict[self.best_test_index])
        self.print_output_csv(label="worst", cvtest_entry=self.cvtest_dict[self.worst_test_index])
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
        if self.testing_target_data is None:
            raise ValueError("Testing target data cannot be none for cross validation.")
        indices = np.arange(0, len(self.testing_target_data))
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
            prediction_array = np.zeros(len(self.testing_target_data))
            prediction_array[:] = np.nan
            fdict = self.cvtest_dict[cvtest]
            input_train = self.testing_input_data[fdict['train_index']]
            target_train = self.testing_target_data[fdict['train_index']]
            input_test = self.testing_input_data[fdict['test_index']]
            target_test = self.testing_target_data[fdict['test_index']]
            fit = self.model.fit(input_train, target_train)
            predict_test = self.model.predict(input_test)
            rmse = np.sqrt(mean_squared_error(predict_test, target_test))
            merr = mean_error(predict_test, target_test)
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

    def print_output_csv(self, label="", cvtest_entry=None):
        """
            Modify once dataframe is in place
        """
        olabel = "%s_test_data.csv" % label
        ocsvname = os.path.join(self.save_path, olabel)
        self.readme_list.append("%s file created with columns:\n" % olabel)
        headerline = ""
        printarray = None
        if len(self.labeling_features) > 0:
            self.readme_list.append("   labeling features: %s\n" % self.labeling_features)
            print_features = list(self.labeling_features)
        else:
            print_features = list()
        print_features.extend(self.input_features)
        self.readme_list.append("   input features: %s\n" % self.input_features)
        if not (self.testing_target_data is None):
            print_features.append(self.target_feature)
            self.readme_list.append("   target feature: %s\n" % self.target_feature)
            if not (self.target_error_feature is None):
                print_features.append(self.target_error_feature)
                self.readme_list.append("   target error feature: %s\n" % self.target_error_feature)
        for feature_name in print_features:
            headerline = headerline + feature_name + ","
            feature_vector = np.asarray(self.testing_dataset.get_data(feature_name)).ravel()
            if printarray is None:
                printarray = feature_vector
            else:
                printarray = np.vstack((printarray, feature_vector))
        headerline = headerline + "Prediction,"
        self.readme_list.append("   prediction: Prediction\n")
        printarray = np.vstack((printarray, cvtest_entry['prediction_array']))
        printarray=printarray.transpose()
        ptools.mixed_array_to_csv(ocsvname, headerline, printarray)
        return

    def plot_best_worst_overlay(self, notelist=list()):
        kwargs2 = dict()
        kwargs2['xlabel'] = self.xlabel
        kwargs2['ylabel'] = self.ylabel
        kwargs2['labellist'] = ["Best test","Worst test"]
        kwargs2['xdatalist'] = list([self.testing_target_data, 
                            self.testing_target_data])
        kwargs2['ydatalist'] = list(
                [self.cvtest_dict[self.best_test_index]['prediction_array'],
                self.cvtest_dict[self.worst_test_index]['prediction_array']])
        kwargs2['xerrlist'] = list([None,None])
        kwargs2['yerrlist'] = list([None,None])
        kwargs2['notelist'] = list(notelist)
        kwargs2['guideline'] = 1
        kwargs2['plotlabel'] = "best_worst_overlay"
        kwargs2['save_path'] = self.save_path
        kwargs2['stepsize'] = self.stepsize
        if not (self.mark_outlying_points is None):
            kwargs2['marklargest'] = self.mark_outlying_points
            if (self.labeling_features is None) or (len(self.labeling_features) == 0):
                raise ValueError("Must specify some labeling features if you want to mark the largest outlying points")
            labels = np.asarray(self.testing_dataset.get_data(self.labeling_features[0])).ravel()
            kwargs2['mlabellist'] = list([labels,labels])
        plotxy.multiple_overlay(**kwargs2)
        self.readme_list.append("Plot best_worst_overlay.png created,\n")
        self.readme_list.append("    showing the best and worst of %i tests.\n" % self.num_cvtests)
        return

def execute(model, data, savepath, lwr_data="", 
            num_runs=200,
            leave_out_percent=20,
            xlabel="Measured",
            ylabel="Predicted",
            label_field_name=None,
            stepsize=1,
            marklargest=None,
            *args, **kwargs):
    """Basic cross validation
        Args:
            model, data, savepath, lwr_data: see AllTests.py
            num_runs <int>: number of runs to repeat each cross-validation
                            split 
                            Default 200.
            leave_out_percent <float>: percent, for a leave-out percentage
                            for shufflesplit.
                    The training size will be (100 - leave_out_percent)/100.
                    The test size will be leave_out_percent/100.
                    e.g. 20 for 20 percent.
            xlabel <str>: x-axis label for predicted-vs-measured plot
            ylabel <str>: y-axis label
            label_field_name <str>: field name for labeling points
            stepsize <float>: step size for plot grid
            marklargest <str>: (optional) number of outliers to mark using the 
                                data in label_field_name; 
                                format is int,int 
                                where the first integer is
                                the number to mark in the "best test" case, and
                                second integer is the number to mark in the
                                "worst test" case; e.g. "0,6" (optional)
    """
    num_runs = int(num_runs)
    leave_out_percent = float(leave_out_percent)
    stepsize = float(stepsize)
    
    # get data
    Ydata = np.array(data.get_y_data()).ravel()
    Xdata = np.array(data.get_x_data())
    dlen = len(Xdata)
    indices = np.arange(0, dlen)

    print("Using shufflesplit, or leave out %i percent" % leave_out_percent)
    test_fraction = leave_out_percent / 100.0
    cvmodel = ShuffleSplit(n_splits = 1, 
                            test_size = test_fraction, 
                            random_state = None)

    Y_predicted_best = list()
    Y_predicted_worst = list()

    maxRMS =  0
    minRMS =  1000000

    Mean_RMS_List = list()
    Mean_ME_List = list()
    for numrun in range(num_runs):
        print("Run %i/%i" % (numrun+1, num_runs))
        run_rms_list = list()
        run_me_list = list()
        Run_Y_Pred = np.empty(dlen)
        Run_Abs_Err = np.empty(dlen)
        Run_Y_Pred.fill(np.nan)
        Run_Abs_Err.fill(np.nan)
        # split into testing and training sets
        ntest=0
        for train, test in cvmodel.split(indices):
            ntest = ntest + 1
            print("Test %i.%i" % (numrun+1,ntest))
            X_train, X_test = Xdata[train], Xdata[test]
            Y_train, Y_test = Ydata[train], Ydata[test]
            # train on training sets
            model.fit(X_train, Y_train)
            Y_test_Pred = model.predict(X_test)
            rms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
            me = mean_error(Y_test_Pred,Y_test)
            run_rms_list.append(rms)
            run_me_list.append(me)
            Run_Y_Pred[test] = Y_test_Pred
            Run_Abs_Err[test] = np.absolute(Y_test - Y_test_Pred)

        mean_run_rms = np.mean(run_rms_list)
        mean_run_me = np.mean(run_me_list)
        Mean_RMS_List.append(mean_run_rms)
        Mean_ME_List.append(mean_run_me)
        if mean_run_rms > maxRMS:
            maxRMS = mean_run_rms
            Y_predicted_worst = Run_Y_Pred
            Worst_Abs_Err = Run_Abs_Err

        if mean_run_rms < minRMS:
            minRMS = mean_run_rms
            Y_predicted_best = Run_Y_Pred
            Best_Abs_Err = Run_Abs_Err

    avgRMS = np.mean(Mean_RMS_List)
    medRMS = np.median(Mean_RMS_List)
    sd = np.std(Mean_RMS_List)
    meanME = np.mean(Mean_ME_List)

    if not (os.path.isdir(savepath)):
        os.mkdir(savepath)
    print("The average RMSE was {:.3f}".format(avgRMS))
    print("The median RMSE was {:.3f}".format(medRMS))
    print("The max RMSE was {:.3f}".format(maxRMS))
    print("The min RMSE was {:.3f}".format(minRMS))
    print("The std deviation of the RMSE values was {:.3f}".format(sd))
    print("The average mean error was {:.3f}".format(meanME))

    notelist_best = list()
    notelist_best.append("Min RMSE: {:.2f}".format(minRMS))
    notelist_best.append("Mean RMSE: {:.2f}".format(avgRMS))
    notelist_best.append("Std. Dev.: {:.2f}".format(sd))
    
    notelist_worst = list()
    notelist_worst.append("Max RMSE.: {:.2f}".format(maxRMS))
    kwargs=dict()
    kwargs['xlabel'] = xlabel
    kwargs['ylabel'] = ylabel
    kwargs['notelist_best'] = notelist_best
    kwargs['notelist_worst'] = notelist_worst
    kwargs['savepath'] = savepath
    kwargs['stepsize'] = stepsize

    plotpm.best_worst(Ydata, Y_predicted_best, Y_predicted_worst, **kwargs)

    kwargs2 = dict()
    kwargs2['xlabel'] = xlabel
    kwargs2['ylabel'] = ylabel
    kwargs2['labellist'] = ["Best test","Worst test"]
    notelist=list()
    notelist.append("Mean RMSE over %i LO-%i%% tests:" % (num_runs,leave_out_percent))
    notelist.append("{:.2f} $\pm$ {:.2f}".format(avgRMS, sd))
    kwargs2['xdatalist'] = list([Ydata,Ydata])
    kwargs2['ydatalist'] = list([Y_predicted_best,Y_predicted_worst])
    kwargs2['xerrlist'] = list([None,None])
    kwargs2['yerrlist'] = list([None,None])
    kwargs2['notelist'] = notelist
    kwargs2['guideline'] = 1
    kwargs2['plotlabel'] = "best_worst_overlay"
    kwargs2['savepath'] = savepath
    kwargs2['stepsize'] = stepsize


    if label_field_name == None: #help label data
        label_field_name = data.x_features[0]

    labels = np.asarray(data.get_data(label_field_name)).ravel()
    
    if not (marklargest is None):
        kwargs2['marklargest'] = marklargest
        kwargs2['mlabellist'] = list([labels,labels])
    plotxy.multiple_overlay(**kwargs2)
    
    
    csvname = os.path.join(savepath,"Leave%iPercentOut_CV_data.csv" % int(leave_out_percent))
    headerline = "%s,Measured,Predicted best,Absolute error best,Predicted worst,Absolute error worst" % label_field_name
    myarray = np.array([labels, Ydata,
                Y_predicted_best, Best_Abs_Err, 
                Y_predicted_worst,Worst_Abs_Err]).transpose()
    ptools.mixed_array_to_csv(csvname, headerline, myarray)
    return
