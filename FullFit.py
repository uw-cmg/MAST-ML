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
            trainx=None,
            trainy=None, 
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
    kwargs_f['do_pergroup_fits'] = 1
    kwargs_f['test_numericdata'] = np.asarray(data.get_data(numeric_field_name)).ravel()
    if not(group_field_name == None):
        kwargs_f['test_groupdata'] = np.asarray(data.get_data(group_field_name)).ravel()
        if label_field_name == None:
            label_field_name = group_field_name
        kwargs_f['test_labeldata'] = np.asarray(data.get_data(label_field_name)).ravel()
    kwargs_f['full_xtrain'] = Xdata
    kwargs_f['full_ytrain'] = ydata
    do_full_fit(model, **kwargs_f)
    return

def do_full_fit(model,
        full_xtrain = None,
        full_ytrain = None,
        full_xtest = None,
        full_ytest = None,
        savepath="",
        xlabel="",
        ylabel="",
        stepsize=1,
        do_pergroup_fits = 1,
        test_groupdata=None,
        train_groupdata=None,
        test_numericdata=None,
        test_labeldata=None,
        group_field_name=None,
        numeric_field_name = None,
        label_field_name=None,
        *args,**kwargs):
    """Full fit.
        test_groupdata, test_numericdata, and test_labeldata must correspond to full_ytest.
        If full_ytest is not given, full_ytest is the same as full_ytrain.
    """
    stepsize = float(stepsize)
    if full_xtest is None:
        full_xtest = np.copy(full_xtrain)
    if full_ytest is None:
        full_ytest = np.copy(full_ytrain)
    if train_groupdata is None:
        if test_groupdata is None:
            pass
        else:
            train_groupdata = np.copy(test_groupdata)
    kwargs=dict()
    kwargs['xlabel'] = xlabel
    kwargs['ylabel'] = ylabel
    kwargs['stepsize'] = stepsize
    kwargs['savepath'] = savepath
    kwargs['guideline'] = 1
    kwargs['trainx'] = full_xtrain
    kwargs['trainy'] = full_ytrain
    kwargs['testx'] = full_xtest
    kwargs['testy'] = full_ytest
    [ypredict, y_abs_err, rmse, mean_error] = do_single_fit(model, **kwargs)

    if numeric_field_name == None: #help identify each point
        raise ValueError("Numeric field name not set.")

    if group_field_name == None:
        headerline = "%s,Measured,Predicted,Absolute error" % numeric_field_name
        myarray = np.array([test_numericdata, full_ytest, ypredict, y_abs_err]).transpose()
    else:
        headerline = "%s,%s,Measured,Predicted,Absolute error" % (numeric_field_name, group_field_name)
        myarray = np.array([test_numericdata, test_groupdata, full_ytest, ypredict, y_abs_err]).transpose()
    
    csvname = os.path.join(savepath, "FullFit_data.csv")
    ptools.mixed_array_to_csv(csvname, headerline, myarray)
    
    if group_field_name == None:
        return
    train_group_indices = gttd.get_logo_indices(train_groupdata)
    test_group_indices = gttd.get_logo_indices(test_groupdata)
    
    test_groups = list(test_group_indices.keys())
    test_groups.sort()

    train_groups = list(train_group_indices.keys())
    train_groups.sort()
    
    #EXTRACT EACH GROUP FITS FROM OVERALL FIT, and OVERLAY
    xdatalist=list()
    ydatalist=list()
    labellist=list()
    xerrlist=list()
    yerrlist=list()
    group_notelist=list()
    group_notelist.append("RMSE from overall fitting:")
    rmselist=list()
    for group in test_groups:
        g_index = test_group_indices[group]["test_index"]
        g_label = test_labeldata[g_index[0]]
        g_ypredict = ypredict[g_index]
        g_ydata = full_ytest[g_index]
        g_mean_error = np.mean(g_ypredict - g_ydata)
        g_rmse = np.sqrt(mean_squared_error(g_ypredict, g_ydata))
        xdatalist.append(g_ydata) #actual
        ydatalist.append(g_ypredict) #predicted
        labellist.append(g_label)
        xerrlist.append(None)
        yerrlist.append(None)
        rmselist.append(g_rmse)
        group_notelist.append('{:<1}: {:.2f}'.format(g_label, g_rmse))
    #allow largest RMSE to be plotted if too many lines
    numlines = len(xdatalist)
    if numlines > 6: #too many lines for multiple_overlay
        import heapq
        showmax = 3
        maxrmslist = heapq.nlargest(showmax, rmselist)
        maxidxlist = heapq.nlargest(showmax, range(len(rmselist)), 
                        key=lambda x: rmselist[x])
        temp_xdatalist=list()
        temp_ydatalist=list()
        temp_xerrlist=list()
        temp_yerrlist=list()
        temp_group_notelist=list()
        temp_group_notelist.append("RMSE from overall fitting:")
        temp_labellist=list()
        bigxdata = np.empty((0,1))
        bigydata = np.empty((0,1))
        biglabel = "All others" # Use "_" for no label
        bigxerr = np.empty((0,1))
        bigyerr = np.empty((0,1))
        for nidx in range(0, numlines):
            if not (nidx in maxidxlist):
                bigxdata = np.append(bigxdata, xdatalist[nidx])
                bigydata = np.append(bigydata, ydatalist[nidx])
                bxerr = xerrlist[nidx]
                if bxerr == None:
                    bxerr = np.zeros(len(xdatalist[nidx]))
                bigxerr = np.append(bigxerr, bxerr)
                byerr = yerrlist[nidx]
                if byerr == None:
                    byerr = np.zeros(len(ydatalist[nidx]))
                bigyerr = np.append(bigyerr, byerr)
        temp_xdatalist.append(bigxdata)
        temp_ydatalist.append(bigydata)
        temp_labellist.append(biglabel)
        temp_xerrlist.append(bigxerr)
        temp_yerrlist.append(bigyerr)
        temp_group_notelist.append("Overall: %3.2f" % rmse) #overall rmse
        for nidx in range(0, numlines): #loop repeats because want big array first
            if (nidx in maxidxlist):
                temp_xdatalist.append(xdatalist[nidx])
                temp_ydatalist.append(ydatalist[nidx])
                temp_labellist.append(labellist[nidx])
                temp_xerrlist.append(xerrlist[nidx])
                temp_yerrlist.append(yerrlist[nidx])
                temp_group_notelist.append(group_notelist[nidx+1])
        xdatalist = temp_xdatalist
        ydatalist = temp_ydatalist
        labellist = temp_labellist
        xerrlist = temp_xerrlist
        yerrlist = temp_yerrlist
        group_notelist = temp_group_notelist
    kwargs['xdatalist'] = xdatalist
    kwargs['ydatalist'] = ydatalist
    kwargs['stepsize'] = stepsize
    kwargs['xerrlist'] = xerrlist
    kwargs['yerrlist'] = yerrlist
    kwargs['labellist'] = labellist
    kwargs['notelist'] = group_notelist
    kwargs['plotlabel'] = "OverallFit_overlay"
    plotxy.multiple_overlay(**kwargs) 
    
    #GET PER-GROUP FITS, and overlay them
    if not(do_pergroup_fits == 1):
        return
   
    xdatalist=list()
    ydatalist=list()
    labellist=list()
    xerrlist=list()
    yerrlist=list()
    group_notelist=list()
    group_notelist.append("RMSE for per-group fitting:")
    for test_group in test_groups:
        if not test_group in train_groups: #cannot be trained
            continue
        g_train_index = train_group_indices[test_group]["test_index"]
        g_test_index = test_group_indices[test_group]["test_index"]
        g_label = test_labeldata[g_test_index[0]]
        kwargs['plotlabel'] = "GroupFit_%s_%s" % (test_group, g_label)
        kwargs['trainx'] = full_xtrain[g_train_index]
        kwargs['trainy'] = full_ytrain[g_train_index]
        kwargs['testx'] = full_xtest[g_test_index]
        g_ytest = full_ytest[g_test_index]
        kwargs['testy'] = g_ytest
        [g_ypredict, g_y_abs_err, g_rmse, g_mean_error] = do_single_fit(model, **kwargs)
        g_myarray = np.array([test_labeldata[g_test_index], 
                            test_groupdata[g_test_index], 
                            g_ytest, g_ypredict, g_y_abs_err]).transpose()
        csvname = os.path.join(savepath, "GroupFit_data_%s_%s.csv" % (test_group, g_label))
        ptools.mixed_array_to_csv(csvname, headerline, g_myarray)
        xdatalist.append(g_ytest) #actual
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
    return
