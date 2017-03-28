import matplotlib
import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import data_analysis.printout_tools as ptools
import plot_data.plot_predicted_vs_measured as plotpm
import plot_data.plot_xy as plotxy
import portion_data.get_test_train_data as gttd
import os

def do_single_fit(model, xdata, ydata, xlabel="", ylabel="", 
            stepsize=1,
            guideline=1,
            savepath="",
            plotlabel="single_fit",
            **kwargs):
    model.fit(xdata, ydata)
    ypredict = model.predict(xdata)
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
    kwargs['stepsize'] = stepsize
    kwargs['notelist'] = notelist
    kwargs['guideline'] = guideline
    kwargs['savepath'] = savepath
    kwargs['plotlabel'] = plotlabel
    plotxy.single(ydata, ypredict, **kwargs)
    return [ypredict, y_abs_err, rmse, mean_error]

def execute(model, data, savepath, 
        xlabel="Measured",
        ylabel="Predicted",
        stepsize=1,
        group_field_name=None,
        label_field_name=None,
        numeric_field_name=None,
        *args, **kwargs):
    """Full fit
    """
    stepsize = float(stepsize)

    # Train the model using the training sets
    Xdata = np.asarray(data.get_x_data())
    ydata = np.asarray(data.get_y_data()).ravel()
    
    kwargs=dict()
    kwargs['xlabel'] = xlabel
    kwargs['ylabel'] = ylabel
    kwargs['stepsize'] = stepsize
    kwargs['savepath'] = savepath
    kwargs['guideline'] = 1
    [ypredict, y_abs_err, rmse, mean_error] = do_single_fit(model, Xdata, ydata, **kwargs)
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

    if numeric_field_name == None: #help identify each point
        numeric_field_name = data.x_features[0]

    labels = np.asarray(data.get_data(numeric_field_name)).ravel()

    if group_field_name == None:
        headerline = "%s,Measured,Predicted,Absolute error" % numeric_field_name
        myarray = np.array([labels, ydata, ypredict, y_abs_err]).transpose()
    else:
        groupdata = np.asarray(data.get_data(group_field_name)).ravel()
        headerline = "%s,%s,Measured,Predicted,Absolute error" % (numeric_field_name, group_field_name)
        myarray = np.array([labels, groupdata, ydata, ypredict, y_abs_err]).transpose()
    
    csvname = os.path.join(savepath, "FullFit_data.csv")
    ptools.mixed_array_to_csv(csvname, headerline, myarray)
    
    if group_field_name == None:
        return

    if label_field_name == None:
        labeldata = np.copy(groupdata)
    else:
        labeldata = np.asarray(data.get_data(label_field_name)).ravel()
    
    indices = gttd.get_field_logo_indices(data, group_field_name)
    
    groups = list(indices.keys())
    groups.sort()

   
    xdatalist=list()
    ydatalist=list()
    labellist=list()
    xerrlist=list()
    yerrlist=list()
    group_notelist=list()
    group_notelist.append("RMSE for per-group fitting:")
    for group in groups:
        g_index = indices[group]["test_index"]
        g_label = labeldata[g_index[0]]
        kwargs['plotlabel'] = "per_group_fit_%s_%s" % (group, g_label)
        [g_ypredict, g_y_abs_err, g_rmse, g_mean_error] = do_single_fit(model, Xdata[g_index], ydata[g_index], **kwargs)
        g_myarray = np.array([labeldata[g_index], groupdata[g_index], ydata[g_index], g_ypredict, g_y_abs_err]).transpose()
        csvname = os.path.join(savepath, "GroupFit_data_%s_%s.csv" % (group, g_label))
        ptools.mixed_array_to_csv(csvname, headerline, g_myarray)
        xdatalist.append(ydata[g_index]) #actual
        ydatalist.append(g_ypredict) #predicted
        labellist.append(g_label)
        xerrlist.append(None)
        yerrlist.append(None)
        group_notelist.append("%15s: %3.2f" % (g_label, g_rmse))
    kwargs['xdatalist'] = xdatalist
    kwargs['ydatalist'] = ydatalist
    kwargs['xerrlist'] = xerrlist
    kwargs['yerrlist'] = yerrlist
    kwargs['labellist'] = labellist
    kwargs['notelist'] = group_notelist
    plotxy.multiple_overlay(**kwargs) 
    return
