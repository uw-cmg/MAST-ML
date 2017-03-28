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

def do_single_fit(model, 
            trainx, 
            trainy, 
            testx=None, 
            testy=None,
            xlabel="", ylabel="", 
            stepsize=1,
            guideline=1,
            savepath="",
            plotlabel="single_fit",
            **kwargs):
    """Single fit with test/train data and plotting.
    """
    if testx is None:
        testx = np.copy(trainx)
    if testy is None:
        testy = np.copy(trainy)
    model.fit(trainx, trainy)
    ypredict = model.predict(testx)
    rmse = np.sqrt(mean_squared_error(ypredict, testy))
    y_abs_err = np.absolute(ypredict - testy)
    mean_error = np.mean(ypredict - testy) #mean error sees if fit is shifted in one direction or another; so, do not want absolute here

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
    plotxy.single(testy, ypredict, **kwargs)
    return [ypredict, y_abs_err, rmse, mean_error]


def execute(model, data, savepath, 
        xlabel="Measured",
        ylabel="Predicted",
        stepsize=1,
        group_field_name=None,
        label_field_name=None,
        numeric_field_name=None,
        *args, **kwargs):
    """Do full fit.
        Main function is split off from execute in order to allow external use.
    """
    stepsize=float(stepsize)
    if numeric_field_name == None: #help identify each point
        numeric_field_name = data.x_features[0]
    Xdata = np.asarray(data.get_x_data())
    ydata = np.asarray(data.get_y_data()).ravel()
    #
    kwargs_f = dict()
    kwargs_f['xlabel'] = xlabel
    kwargs_f['ylabel'] = ylabel
    kwargs_f['stepsize'] = stepsize
    kwargs_f['numeric_field_name'] = numeric_field_name
    kwargs_f['group_field_name'] = group_field_name
    kwargs_f['label_field_name'] = label_field_name
    kwargs_f['savepath'] = savepath
    kwargs_f['numericdata'] = np.asarray(data.get_data(numeric_field_name)).ravel()
    if not(group_field_name == None):
        kwargs_f['groupdata'] = np.asarray(data.get_data(group_field_name)).ravel()
        group_indices = gttd.get_field_logo_indices(data, group_field_name)
        kwargs_f['group_indices'] = group_indices
        if label_field_name == None:
            label_field_name = group_field_name
        kwargs_f['labeldata'] = np.asarray(data.get_data(label_field_name)).ravel()
    do_full_fit(model, Xdata, ydata, **kwargs_f)
    return

def do_full_fit(model, Xdata, ydata, savepath="",
        xlabel="",
        ylabel="",
        stepsize=1,
        groupdata=None,
        group_indices=None,
        numericdata=None,
        labeldata=None,
        group_field_name=None,
        numeric_field_name = None,
        label_field_name=None,
        *args,**kwargs):
    """Full fit
    """
    stepsize = float(stepsize)
    
    kwargs=dict()
    kwargs['xlabel'] = xlabel
    kwargs['ylabel'] = ylabel
    kwargs['stepsize'] = stepsize
    kwargs['savepath'] = savepath
    kwargs['guideline'] = 1
    [ypredict, y_abs_err, rmse, mean_error] = do_single_fit(model, Xdata, ydata, **kwargs)

    if numeric_field_name == None: #help identify each point
        raise ValueError("Numeric field name not set.")

    if group_field_name == None:
        headerline = "%s,Measured,Predicted,Absolute error" % numeric_field_name
        myarray = np.array([numericdata, ydata, ypredict, y_abs_err]).transpose()
    else:
        headerline = "%s,%s,Measured,Predicted,Absolute error" % (numeric_field_name, group_field_name)
        myarray = np.array([numericdata, groupdata, ydata, ypredict, y_abs_err]).transpose()
    
    csvname = os.path.join(savepath, "FullFit_data.csv")
    ptools.mixed_array_to_csv(csvname, headerline, myarray)
    
    if group_field_name == None:
        return
    #GET PER-GROUP FITS, and overlay them
    
    
    groups = list(group_indices.keys())
    groups.sort()

   
    xdatalist=list()
    ydatalist=list()
    labellist=list()
    xerrlist=list()
    yerrlist=list()
    group_notelist=list()
    group_notelist.append("RMSE for per-group fitting:")
    for group in groups:
        g_index = group_indices[group]["test_index"]
        g_label = labeldata[g_index[0]]
        kwargs['plotlabel'] = "GroupFit_%s_%s" % (group, g_label)
        [g_ypredict, g_y_abs_err, g_rmse, g_mean_error] = do_single_fit(model, Xdata[g_index], ydata[g_index], **kwargs)
        g_myarray = np.array([labeldata[g_index], groupdata[g_index], ydata[g_index], g_ypredict, g_y_abs_err]).transpose()
        csvname = os.path.join(savepath, "GroupFit_data_%s_%s.csv" % (group, g_label))
        ptools.mixed_array_to_csv(csvname, headerline, g_myarray)
        xdatalist.append(ydata[g_index]) #actual
        ydatalist.append(g_ypredict) #predicted
        labellist.append(g_label)
        xerrlist.append(None)
        yerrlist.append(None)
        group_notelist.append('{:<1}: {:.2f}'.format(g_label, g_rmse))
    kwargs['xdatalist'] = xdatalist
    kwargs['ydatalist'] = ydatalist
    kwargs['stepsize'] = stepsize
    kwargs['xerrlist'] = xerrlist
    kwargs['yerrlist'] = yerrlist
    kwargs['labellist'] = labellist
    kwargs['notelist'] = group_notelist
    kwargs['plotlabel'] = "GroupFit_overlay"
    plotxy.multiple_overlay(**kwargs) 
    
    #EXTRACT EACH GROUP FITS FROM OVERALL FIT, and OVERLAY
   
    xdatalist=list()
    ydatalist=list()
    labellist=list()
    xerrlist=list()
    yerrlist=list()
    group_notelist=list()
    group_notelist.append("RMSE from overall fitting:")
    for group in groups:
        g_index = group_indices[group]["test_index"]
        g_label = labeldata[g_index[0]]
        g_ypredict = ypredict[g_index]
        g_ydata = ydata[g_index]
        g_mean_error = np.mean(g_ypredict - g_ydata)
        g_rmse = np.sqrt(mean_squared_error(g_ypredict, g_ydata))
        xdatalist.append(g_ydata) #actual
        ydatalist.append(g_ypredict) #predicted
        labellist.append(g_label)
        xerrlist.append(None)
        yerrlist.append(None)
        group_notelist.append('{:<1}: {:.2f}'.format(g_label, g_rmse))
    kwargs['xdatalist'] = xdatalist
    kwargs['ydatalist'] = ydatalist
    kwargs['stepsize'] = stepsize
    kwargs['xerrlist'] = xerrlist
    kwargs['yerrlist'] = yerrlist
    kwargs['labellist'] = labellist
    kwargs['notelist'] = group_notelist
    kwargs['plotlabel'] = "OverallFit_overlay"
    plotxy.multiple_overlay(**kwargs) 
    return
