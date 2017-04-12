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

def execute(model, data, savepath, lwr_data="", 
            num_runs=200,
            num_folds=5,
            xlabel="Measured",
            ylabel="Predicted",
            stepsize=1,
            numeric_field_name=None,
            *args, **kwargs):
    """Basic cross validation
        Args:
            model, data, savepath, lwr_data: see AllTests.py
            num_runs <int>: number of runs to repeat each cross-validation
                            split (e.g. num_runs iterations of 5fold CV, where
                            each 5-fold CV has 5 test-train sets)
                            Default 200.
            num_folds <int>: number of folds.
            xlabel <str>: x-axis label for predicted-vs-measured plot
            ylabel <str>: y-axis label
            stepsize <float>: step size for plot grid
            numeric_field_name <str>: Field name of a numeric field which
                            may help identify data
    """
    num_runs = int(num_runs)
    num_folds = int(num_folds)
    stepsize = float(stepsize)
    
    # get data
    Ydata = np.array(data.get_y_data()).ravel()
    Xdata = np.array(data.get_x_data())
    dlen = len(Xdata)
    indices = np.arange(0, dlen)

    print("Using %i %i-folds." % (num_runs,num_folds))
    cvmodel = KFold(n_splits=num_folds, shuffle=True)

    Y_predicted_best = list()
    Y_predicted_worst = list()
    Y_predicted_best_fold_numbers = list()
    Y_predicted_worst_fold_numbers = list()

    maxRMS =  0
    minRMS =  1000000

    Mean_RMS_List = list()
    Mean_ME_List = list()
    for numrun in range(num_runs):
        print("Run %i/%i" % (numrun+1, num_runs))
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
    print("The average mean-over-{:d}-folds RMSE was {:.3f} over {:d} tests".format(num_folds, avgRMS, num_runs))
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
    notelist.append("Mean over %i tests of:" % num_runs)
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
