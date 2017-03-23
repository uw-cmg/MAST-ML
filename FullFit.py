import matplotlib
import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import data_analysis.printout_tools as ptools
import plot_data.plot_predicted_vs_measured as plotpm
import plot_data.plot_xy as plotxy
import os

def execute(model, data, savepath, 
        xlabel="Measured",
        ylabel="Predicted",
        stepsize=1,
        numericlabelfield=None,
        *args, **kwargs):
    """Full fit
    """
    stepsize = float(stepsize)

    # Train the model using the training sets
    Xdata = data.get_x_data()
    ydata = np.asarray(data.get_y_data()).ravel()
    model.fit(Xdata, ydata)
    ypredict = model.predict(Xdata)
    rmse = np.sqrt(mean_squared_error(ypredict, ydata))
    y_abs_err = np.absolute(ypredict - ydata)
    mean_error = np.mean(ypredict - ydata) #mean error sees if fit is shifted in one direction or another; so, do not want absolute here

    notelist=list()
    notelist.append("RMSE: %3.2f" % rmse)
    notelist.append("Mean error: %3.2f" % mean_error)
    print("RMSE: %3.2f" % rmse)
    print("Mean error: %3.2f" % mean_error)

    kwargs=dict()
    kwargs['xlabel'] = xlabel
    kwargs['ylabel'] = ylabel
    #kwargs['stepsize'] = stepsize
    kwargs['notelist'] = notelist
    kwargs['guideline'] = 1
    kwargs['savepath'] = savepath
    plotxy.single(ydata, ypredict, **kwargs)

    #datasets = ['IVAR', 'ATR-1', 'ATR-2']
    #colors = ['#BCBDBD', '#009AFF', '#FF0A09']
    #fig, ax = plt.subplots()

    #calculate rms for each dataset
    #TTM want data to be separated out earlier. Just calculate RMS.
    #for dataset in range(max(np.asarray(data.get_data("Data Set code")).ravel()) + 1):
    #    data.remove_all_filters()
    #    data.add_inclusive_filter("Data Set code", '=', dataset)
    #    Ypredict = model.predict(data.get_x_data())
    #    Ydata = np.asarray(data.get_y_data()).ravel()
    #    # calculate rms
    #    rms = np.sqrt(mean_squared_error(Ypredict, Ydata))
    #    # graph outputs
    #    ax.scatter(Ydata, Ypredict, s=7, color=colors[dataset], label= datasets[dataset], lw = 0)
    #    #ax.text(.05, .83 - .05*dataset, '{} RMS: {:.3f}'.format(datasets[dataset],rms), fontsize=14, transform=ax.transAxes)
    # calculate rms
    #rms = np.sqrt(mean_squared_error(Ypredict, Ydata))
    # graph outputs
    #dataset=0 #TTM dummy index value
    #matplotlib.rcParams.update({'font.size': 18})
    #ax.scatter(Ydata, Ypredict, s=7, color=colors[dataset], label= datasets[dataset], lw = 0)
    #ax.text(.05, .83 - .05*dataset, '{} RMS: {:.3f}'.format(datasets[dataset],rms), transform=ax.transAxes)
    #
    #ax.legend()
    #ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3")
    #ax.set_xlabel('Measured (MPa)')
    #ax.set_ylabel('Predicted (MPa)')
    #ax.set_title('Full Fit')
    #ax.text(.05, .88, 'Overall RMS: %.4f' % (overall_rms), transform=ax.transAxes)
    #fig.savefig(savepath.format(ax.get_title()), dpi=300, bbox_inches='tight')
    #plt.clf()
    #plt.close()

    if numericlabelfield == None:
        numericlabelfield = data.x_features[0]

    labels = np.asarray(data.get_data(numericlabelfield)).ravel()

    csvname = os.path.join(savepath, "FullFit_data.csv")
    headerline = "%s,Measured,Predicted,Absolute error" % numericlabelfield
    myarray = np.array([labels, ydata, ypredict, y_abs_err]).transpose()
    ptools.array_to_csv(csvname, headerline, myarray)
    return
