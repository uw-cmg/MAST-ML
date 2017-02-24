import matplotlib
import numpy as np
import data_parser
import matplotlib.pyplot as plt
from mean_error import mean_error
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import data_analysis.printout_tools as ptools

def execute(model, data, savepath, lwr_data="", *args, **kwargs):
    if "num_runs" in kwargs.keys():
        num_runs = int(kwargs["num_runs"])
    else:
        num_runs = 200
    if "num_folds" in kwargs.keys():
        num_folds = int(kwargs["num_folds"])
    else:
        num_runs = 5

    # get data
    Ydata = np.array(data.get_y_data()).ravel()
    Xdata = np.array(data.get_x_data())

    Y_predicted_best = []
    Y_predicted_worst = []
    Y_predicted_best_fold_numbers = []
    Y_predicted_worst_fold_numbers = []
    Worst_RMS=[]
    Best_RMS=[]

    maxRMS = 1
    minRMS = 100

    RMS_List = []
    ME_List = []
    for n in range(num_runs):
        kf = cross_validation.KFold(len(Xdata), n_folds=num_folds, shuffle=True)
        K_fold_rms_list = []
        K_fold_me_list = []
        Overall_Y_Pred = np.zeros(len(Xdata))
        Pred_Fold_Numbers = np.zeros(len(Xdata))
        Overall_Abs_Err = np.zeros(len(Xdata))
        # split into testing and training sets
        nfold=0
        for train_index, test_index in kf:
            nfold = nfold + 1
            X_train, X_test = Xdata[train_index], Xdata[test_index]
            Y_train, Y_test = Ydata[train_index], Ydata[test_index]
            # train on training sets
            model.fit(X_train, Y_train)
            Y_test_Pred = model.predict(X_test)
            rms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
            me = mean_error(Y_test_Pred,Y_test)
            K_fold_rms_list.append(rms)
            K_fold_me_list.append(me)
            Overall_Y_Pred[test_index] = Y_test_Pred
            Pred_Fold_Numbers[test_index] = nfold
            Overall_Abs_Err[test_index] = np.absolute(Y_test - Y_test_Pred)

        RMS_List.append(np.mean(K_fold_rms_list))
        ME_List.append(np.mean(K_fold_me_list))
        if np.mean(K_fold_rms_list) > maxRMS:
            maxRMS = np.mean(K_fold_rms_list)
            Y_predicted_worst = Overall_Y_Pred
            Y_predicted_worst_fold_numbers = Pred_Fold_Numbers
            Worst_Abs_Err = Overall_Abs_Err

        if np.mean(K_fold_rms_list) < minRMS:
            minRMS = np.mean(K_fold_rms_list)
            Y_predicted_best = Overall_Y_Pred
            Y_predicted_best_fold_numbers = Pred_Fold_Numbers
            Best_Abs_Err = Overall_Abs_Err

    avgRMS = np.mean(RMS_List)
    medRMS = np.median(RMS_List)
    sd = np.std(RMS_List)
    meanME = np.mean(ME_List)

    print("Using {}x {}-Fold CV: ".format(num_runs, num_folds))
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
    f.savefig(savepath.format("cv_best_worst"), dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close()

    alloys = np.asarray(data.get_data("alloy_number")).ravel()
    csvname = "KFoldCV_data.csv"
    headerline = "Alloy number,Measured,Predicted best,Absolute error best,Fold numbers best,Predicted worst,Absolute error worst,Fold numbers worst"
    myarray = np.array([alloys, Ydata,
                Y_predicted_best, Best_Abs_Err, 
                Y_predicted_best_fold_numbers,
                Y_predicted_worst,Worst_Abs_Err,
                Y_predicted_worst_fold_numbers]).transpose()
    ptools.array_to_csv(csvname, headerline, myarray)
    return
