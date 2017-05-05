import os
import matplotlib
import numpy as np
import data_parser
import matplotlib.pyplot as plt
from mean_error import mean_error
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import data_analysis.printout_tools as ptools
import plot_data.plot_predicted_vs_measured as plotpm
import plot_data.plot_xy as plotxy
from SingleFit import SingleFit
from SingleFit import timeit
from sklearn.metrics import r2_score

class KFoldCV(SingleFit):
    """KFold cross validation
   
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
        num_folds <int>: Number of folds for KFold CV
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
        num_folds=2,
        num_cvtests=10,
        fix_random_for_testing=0,
        *args, **kwargs):
        """
        Additional class attributes to parent class:
            Set by keyword:
            self.num_folds <int>: Number of folds for KFold test
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
        self.num_folds = int(num_folds)
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
        return

    def set_up_cv(self):
        if self.testing_target_data is None:
            raise ValueError("Testing target data cannot be none for cross validation.")
        indices = np.arange(0, len(self.testing_target_data))
        self.readme_list.append("----- CV setup -----\n")
        self.readme_list.append("%i CV tests,\n" % self.num_cvtests)
        self.readme_list.append("each with %i folds\n" % self.num_folds)
        self.cvmodel = KFold(n_splits = self.num_folds, shuffle=True, 
                                random_state = None)
        for cvtest in range(0, self.num_cvtests):
            self.cvtest_dict[cvtest] = dict()
            foldidx=0
            fold_rmses = np.zeros(self.num_folds)
            for train, test in self.cvmodel.split(indices):
                fdict=dict()
                fdict['train_index'] = train
                fdict['test_index'] = test
                self.cvtest_dict[cvtest][foldidx] = dict(fdict)
                foldidx = foldidx + 1
        return
    
    def cv_fit_and_predict(self):
        for cvtest in self.cvtest_dict.keys():
            fold_rmses = np.zeros(self.num_folds)
            fold_array = np.zeros(len(self.testing_target_data))
            prediction_array = np.zeros(len(self.testing_target_data))
            for fold in self.cvtest_dict[cvtest].keys():
                fdict = self.cvtest_dict[cvtest][fold]
                input_train = self.testing_input_data[fdict['train_index']]
                target_train = self.testing_target_data[fdict['train_index']]
                input_test = self.testing_input_data[fdict['test_index']]
                target_test = self.testing_target_data[fdict['test_index']]
                fit = self.model.fit(input_train, target_train)
                predict_test = self.model.predict(input_test)
                rmse = np.sqrt(mean_squared_error(predict_test, target_test))
                self.cvtest_dict[cvtest][fold]['rmse'] = rmse
                fold_rmses[fold] = rmse
                fold_array[fdict['test_index']] = fold
                prediction_array[fdict['test_index']] = predict_test
            self.cvtest_dict[cvtest]["avg_rmse"] = np.mean(fold_rmses)
            self.cvtest_dict[cvtest]["fold_array"] = fold_array
            self.cvtest_dict[cvtest]["prediction_array"] = prediction_array
        return

    def get_statistics(self):
        cvtest_avg_rmses = list()
        for cvtest in range(0, self.num_cvtests):
            cvtest_avg_rmses.append(self.cvtest_dict[cvtest]["avg_rmse"])
        highest_rmse = max(cvtest_avg_rmses)
        self.worst_test_index = cvtest_avg_rmses.index(highest_rmse)
        lowest_rmse = min(cvtest_avg_rmses)
        self.best_test_index = cvtest_avg_rmses.index(lowest_rmse)
        self.statistics['avg_avg_rmses'] = np.mean(cvtest_avg_rmses)
        self.statistics['std_avg_rmses'] = np.std(cvtest_avg_rmses)
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
        headerline = headerline + "Fold"
        self.readme_list.append("   fold number: Fold\n")
        printarray = np.vstack((printarray, cvtest_entry['fold_array']))
        printarray=printarray.transpose()
        ptools.mixed_array_to_csv(ocsvname, headerline, printarray)

def execute(model, data, savepath, lwr_data="", 
            num_cvtests=200,
            num_folds=5,
            xlabel="Measured",
            ylabel="Predicted",
            stepsize=1,
            numeric_field_name=None,
            *args, **kwargs):
    """Basic cross validation
        Args:
            model, data, savepath, lwr_data: see AllTests.py
            num_cvtests <int>: number of runs to repeat each cross-validation
                            split (e.g. num_cvtests iterations of 5fold CV, where
                            each 5-fold CV has 5 test-train sets)
                            Default 200.
            num_folds <int>: number of folds.
            xlabel <str>: x-axis label for predicted-vs-measured plot
            ylabel <str>: y-axis label
            stepsize <float>: step size for plot grid
            numeric_field_name <str>: Field name of a numeric field which
                            may help identify data
    """
    num_cvtests = int(num_cvtests)
    num_folds = int(num_folds)
    stepsize = float(stepsize)
    
    # get data
    Ydata = np.array(data.get_y_data()).ravel()
    Xdata = np.array(data.get_x_data())
    dlen = len(Xdata)
    indices = np.arange(0, dlen)

    print("Using %i %i-folds." % (num_cvtests,num_folds))
    cvmodel = KFold(n_splits=num_folds, shuffle=True)

    Y_predicted_best = list()
    Y_predicted_worst = list()
    Y_predicted_best_fold_numbers = list()
    Y_predicted_worst_fold_numbers = list()

    maxRMS =  0
    minRMS =  1000000

    Mean_RMS_List = list()
    Mean_ME_List = list()
    for numrun in range(num_cvtests):
        print("Run %i/%i" % (numrun+1, num_cvtests))
        run_rms_list = list()
        run_me_list = list()
        Run_Y_Pred = np.empty(dlen)
        Run_Fold_Numbers = np.empty(dlen)
        Run_Abs_Err = np.empty(dlen)
        Run_Y_Pred.fill(np.nan)
        Run_Fold_Numbers.fill(np.nan)
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
            Run_Fold_Numbers[test] = ntest
            Run_Abs_Err[test] = np.absolute(Y_test - Y_test_Pred)

        mean_run_rms = np.mean(run_rms_list)
        mean_run_me = np.mean(run_me_list)
        Mean_RMS_List.append(mean_run_rms)
        Mean_ME_List.append(mean_run_me)
        if mean_run_rms > maxRMS:
            maxRMS = mean_run_rms
            Y_predicted_worst = Run_Y_Pred
            Y_predicted_worst_fold_numbers = Run_Fold_Numbers
            Worst_Abs_Err = Run_Abs_Err

        if mean_run_rms < minRMS:
            minRMS = mean_run_rms
            Y_predicted_best = Run_Y_Pred
            Y_predicted_best_fold_numbers = Run_Fold_Numbers
            Best_Abs_Err = Run_Abs_Err

    avgRMS = np.mean(Mean_RMS_List)
    medRMS = np.median(Mean_RMS_List)
    sd = np.std(Mean_RMS_List)
    meanME = np.mean(Mean_ME_List)
    sdME = np.std(Mean_ME_List)

    if not (os.path.isdir(savepath)):
        os.mkdir(savepath)
    print("The average mean-over-{:d}-folds RMSE was {:.3f} over {:d} tests".format(num_folds, avgRMS, num_cvtests))
    print("The median mean RMSE was {:.3f}".format(medRMS))
    print("The max mean RMSE was {:.3f}".format(maxRMS))
    print("The min mean RMSE was {:.3f}".format(minRMS))
    print("The std deviation of the average mean RMSE values was {:.3f}".format(sd))
    print("The average mean error was {:.3f}".format(meanME))
    print("The std deviation of the average mean error was {:.3f}".format(sdME))

    notelist_best = list()
    notelist_best.append("Min RMSE: {:.2f}".format(minRMS))
    notelist_best.append("Mean RMSE: {:.2f}".format(avgRMS))
    notelist_best.append("Std. Dev.: {:.2f}".format(sd))
    
    notelist_worst = list()
    notelist_worst.append("Max RMSE: {:.2f}".format(maxRMS))
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
    kwargs2['label1'] = "Test with lowest fold-average RMSE"
    kwargs2['label2'] = "Test with highest fold-average RMSE"
    notelist=list()
    notelist.append("Mean over %i tests of:" % num_cvtests)
    notelist.append("  {:d}-fold-average RMSE:".format(num_folds))
    notelist.append("    {:.2f} $\pm$ {:.2f}".format(avgRMS, sd))
    notelist.append("  {:d}-fold-average mean error:".format(num_folds))
    notelist.append("    {:.2f} $\pm$ {:.2f}".format(meanME, sdME))
    kwargs2['notelist'] = notelist
    kwargs2['guideline'] = 1
    kwargs2['fill'] = 1
    kwargs2['equalsize'] = 1
    kwargs2['plotlabel'] = "best_worst_overlay"
    kwargs2['savepath'] = savepath
    kwargs2['stepsize'] = stepsize
    plotxy.dual_overlay(Ydata, Y_predicted_best, Ydata, Y_predicted_worst, **kwargs2)

    if numeric_field_name == None:
        numeric_field_name = data.x_features[0]

    labels = np.asarray(data.get_data(numeric_field_name)).ravel()
    csvname = os.path.join(savepath,"KFold_CV_data.csv")
    headerline = "%s,Measured,Predicted best,Absolute error best,Fold numbers best,Predicted worst,Absolute error worst,Fold numbers worst" % numeric_field_name
    myarray = np.array([labels, Ydata,
                Y_predicted_best, Best_Abs_Err, 
                Y_predicted_best_fold_numbers,
                Y_predicted_worst,Worst_Abs_Err,
                Y_predicted_worst_fold_numbers]).transpose()
    ptools.array_to_csv(csvname, headerline, myarray)
    return
