import os
import matplotlib
import numpy as np
import data_parser
import matplotlib.pyplot as plt
from mean_error import mean_error
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
import data_analysis.printout_tools as ptools

def execute(model, data, savepath, lwr_data="", 
            cvtype="kfold",
            num_runs=200,
            num_folds=None,
            leave_out_percent=None,
            *args, **kwargs):
    """Basic cross validation
        Args:
            model, data, savepath, lwr_data: see AllTests.py
            cvtype <str>: cross-validation type:
                kfold, leaveoneout, shufflesplit
            num_runs <int>: number of runs to repeat each cross-validation
                            split (e.g. num_runs iterations of 5fold CV, where
                            each 5-fold CV has 5 test-train sets)
                            Default 200.
            num_folds <int>: number of folds. Only for cvtype of "kfold".
            leave_out_percent <float>: percent, for a leave-out percentage
                            for shufflesplit.
                            The test size will be (100 - leave_out_percent)/100.
                            Only for cvtype of "shufflesplit"
                            e.g. 20 for 20 percent.
    """
    num_runs = int(num_runs)

    cvtype = cvtype.lower()
    cvallowed = ['kfold','leaveoneout','shufflesplit']
    if not cvtype in cvallowed:
        raise ValueError("cvtype must be one of %s" % cvallowed)
    # get data
    Ydata = np.array(data.get_y_data()).ravel()
    Xdata = np.array(data.get_x_data())
    dlen = len(Xdata)
    indices = np.arange(0, dlen)

    cvmodel = None
    if cvtype == 'kfold':
        num_folds = int(num_folds)
        print("Using %i folds." % num_folds)
        cvmodel = KFold(n_splits=num_folds, shuffle=True)
    elif cvtype == 'leaveoneout':
        print("Using leave one out.")
        cvmodel = LeaveOneOut()
    elif cvtype == 'shufflesplit':
        leave_out_percent = float(leave_out_percent)
        print("Using shufflesplit, or leave out %i percent" % leave_out_percent)
        test_fraction = (100.0 - leave_out_percent) / 100.0
        cvmodel = ShuffleSplit(n_splits = 1, 
                                test_size = test_fraction, 
                                random_state = 0)
    else:
        raise ValueError("cvtype must be one of %s" % cvallowed)

    Y_predicted_best = list()
    Y_predicted_worst = list()
    Y_predicted_best_fold_numbers = list()
    Y_predicted_worst_fold_numbers = list()

    maxRMS =  1
    minRMS =  100

    Mean_RMS_List = list()
    Mean_ME_List = list()
    for numrun in range(num_runs):
        print("Run %i/%i" % (numrun+1, num_runs))
        run_rms_list = list()
        run_me_list = list()
        Run_Y_Pred = np.zeros(dlen)
        Run_Fold_Numbers = np.zeros(dlen)
        Run_Abs_Err = np.zeros(dlen)
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
            if cvtype == 'kfold':
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

    if not (os.path.isdir(savepath)):
        os.mkdir(savepath)
    print("The average RMSE was {:.3f}".format(avgRMS))
    print("The median RMSE was {:.3f}".format(medRMS))
    print("The max RMSE was {:.3f}".format(maxRMS))
    print("The min RMSE was {:.3f}".format(minRMS))
    print("The std deviation of the RMSE values was {:.3f}".format(sd))
    print("The average mean error was {:.3f}".format(meanME))

    matplotlib.rcParams.update({'font.size': 18})
    f, ax = plt.subplots(1, 2, figsize = (11,5))
    ax[0].scatter(Ydata, Y_predicted_best, c='black', s=10)
    ax[0].plot(ax[0].get_ylim(), ax[0].get_ylim(), ls="--", c=".3")
    ax[0].set_title('Best Fit')
    ax[0].text(.05, .88, 'Min RMSE: {:.2f}'.format(minRMS), transform=ax[0].transAxes)
    ax[0].text(.05, .81, 'Mean RMSE: {:.2f}'.format(avgRMS), transform=ax[0].transAxes)
    ax[0].text(.05, .74, 'Std. Dev.: {:.2f}'.format(sd), transform=ax[0].transAxes)
    #ax[0].text(.05, .88, 'Min RMSE: {:.2f} MPa'.format(minRMS), transform=ax[0].transAxes)
    #ax[0].text(.05, .81, 'Mean RMSE: {:.2f} MPa'.format(avgRMS), transform=ax[0].transAxes)
    #ax[0].text(.05, .74, 'Std. Dev.: {:.2f} MPa'.format(sd), transform=ax[0].transAxes)
    ax[0].set_xlabel('Measured (MPa)')
    ax[0].set_ylabel('Predicted (MPa)')

    ax[1].scatter(Ydata, Y_predicted_worst, c='black', s=10)
    ax[1].plot(ax[1].get_ylim(), ax[1].get_ylim(), ls="--", c=".3")
    ax[1].set_title('Worst Fit')
    ax[1].text(.05, .88, 'Max RMSE: {:.2f}'.format(maxRMS), transform=ax[1].transAxes)
    ax[1].set_xlabel('Measured (MPa)')
    ax[1].set_ylabel('Predicted (MPa)')

    f.tight_layout()
    f.savefig(os.path.join(savepath, "cv_best_worst"), dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close()

    alloys = np.asarray(data.get_data("alloy_number")).ravel()
    csvname = os.path.join(savepath,"CV_data.csv")
    headerline = "Alloy number,Measured,Predicted best,Absolute error best,Fold numbers best (only if kfold),Predicted worst,Absolute error worst,Fold numbers worst (only if kfold)"
    myarray = np.array([alloys, Ydata,
                Y_predicted_best, Best_Abs_Err, 
                Y_predicted_best_fold_numbers,
                Y_predicted_worst,Worst_Abs_Err,
                Y_predicted_worst_fold_numbers]).transpose()
    ptools.array_to_csv(csvname, headerline, myarray)
    return
