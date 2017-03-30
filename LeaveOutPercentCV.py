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

def execute(model, data, savepath, lwr_data="", 
            num_runs=200,
            leave_out_percent=20,
            xlabel="Measured",
            ylabel="Predicted",
            numeric_field_name=None,
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


    if numeric_field_name == None: #help label data
        numeric_field_name = data.x_features[0]

    labels = np.asarray(data.get_data(numeric_field_name)).ravel()
    
    if not (marklargest is None):
        kwargs2['marklargest'] = marklargest
        kwargs2['mlabellist'] = list([labels,labels])
    plotxy.multiple_overlay(**kwargs2)
    
    
    csvname = os.path.join(savepath,"Leave%iPercentOut_CV_data.csv" % int(leave_out_percent))
    headerline = "%s,Measured,Predicted best,Absolute error best,Predicted worst,Absolute error worst" % numeric_field_name
    myarray = np.array([labels, Ydata,
                Y_predicted_best, Best_Abs_Err, 
                Y_predicted_worst,Worst_Abs_Err]).transpose()
    ptools.mixed_array_to_csv(csvname, headerline, myarray)
    return
