#!/usr/bin/env python
######
# Plot predictions by groups
# Tam Mayeshiba 2017-02-23
######
import matplotlib.pyplot as plt
import matplotlib
import data_parser
import numpy as np
import os
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import data_analysis.printout_tools as ptools
import portion_data.get_test_train_data as gttd

def plot_separate_groups_vs_xfield(fit_data=None, 
                    topred_data=None,
                    std_data=None,
                    topred_Ypredict=None,
                    std_Ypredict=None,
                    group_field_name=None,
                    label_field_name=None,
                    filter_dict=None,
                    xlabel="x",
                    ylabel="y",
                    xfield=None):
    """
        fit_data <data_parser data object>: fitting data
        topred_data <data_parser data object>: data to be predicted
        std_data <data_parser data object>: standard data to be predicted
                        (e.g. if there are not many points in topred_data)
        All data_parser objects should already have x and y features set.
        topred_Ypredict <numpy array>: predictions already made
        std_Ypredict <numpy array>: predictions already made on standard data
        group_field_name <str>: field name for field over which to group
        label_field_name <str>: field name for field containing group labels
        xfield <str>: field name for x field for plotting
    """
    fit_indices = gttd.get_field_logo_indices(fit_data, group_field_name)
    fit_xfield = np.asarray(fit_data.get_data(xfield)).ravel()
    fit_groupdata = np.asarray(fit_data.get_data(group_field_name)).ravel()
    fit_ydata = np.asarray(fit_data.get_y_data()).ravel()

    topred_indices = gttd.get_field_logo_indices(topred_data, group_field_name)
    topred_groupdata = np.asarray(topred_data.get_data(group_field_name)).ravel()
    topred_ydata = np.asarray(topred_data.get_y_data()).ravel()
    topred_xfield = np.asarray(topred_data.get_data(xfield)).ravel()
    if not label_field_name == None:
        labeldata = np.asarray(topred_data.get_data(label_field_name)).ravel()
    else:
        labeldata = np.zeros(len(topred_groupdata))

    if not (std_data == None):
        std_indices = gttd.get_field_logo_indices(std_data, group_field_name)
        std_xfield = np.asarray(std_data.get_data(xfield)).ravel()
        std_groupdata = np.asarray(std_data.get_data(group_field_name)).ravel()
    
    groups = list(topred_indices.keys())
    groups.sort()
    for group in groups:
        print(group)
        test_index = topred_indices[group]["test_index"]
        test_group_val = topred_groupdata[test_index[0]] #left-out group value
        if label_field_name == None:
            test_group_label = "None"
        else:
            test_group_label = labeldata[test_index[0]]
        g_topred_xfield = topred_xfield[test_index]
        g_topred_Ypredict = topred_Ypredict[test_index]
        g_topred_ydata = topred_ydata[test_index]
            
        if std_data == None:
            g_std_xfield = np.copy(g_topred_xfield)
            g_std_Ypredict = np.copy(g_topred_Ypredict)
        else:
            #need to find a better way of matching groups
            for sgroup in std_indices.keys():
                s_test_index = std_indices[sgroup]["test_index"] 
                if std_groupdata[s_test_index[0]] == test_group_val:
                    matchgroup = sgroup
                    continue
            std_test_index = std_indices[matchgroup]["test_index"]
            g_std_xfield = std_xfield[std_test_index]
            g_std_Ypredict = std_Ypredict[std_test_index]

        for fgroup in fit_indices.keys():
            f_test_index = fit_indices[fgroup]["test_index"] 
            if fit_groupdata[f_test_index[0]] == test_group_val:
                fmatchgroup = fgroup
                continue
        fit_test_index = fit_indices[fmatchgroup]["test_index"]
        g_fit_xfield = fit_xfield[fit_test_index]
        g_fit_ydata = fit_ydata[fit_test_index]

        plt.figure()
        plt.hold(True)
        fig, ax = plt.subplots()
        matplotlib.rcParams.update({'font.size':18})
        ax.plot(g_std_xfield, g_std_Ypredict,
                lw=3, color='#ffc04d', label="Prediction")
        ax.scatter(g_fit_xfield, g_fit_ydata,
               lw=0, label="Subset of fitting data", color = 'black')
        ax.scatter(g_topred_xfield, g_topred_ydata,
               lw=0, label="Measured data", color = '#7ec0ee')
        ax.scatter(g_topred_xfield, g_topred_Ypredict,
               lw=0, label="Predicted data", color = 'blue')

        plt.legend(loc = "upper left", fontsize=matplotlib.rcParams['font.size']) #data is sigmoid; 'best' can block data
        plt.title("%s(%s)" % (test_group_val, test_group_label))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig("%s_prediction" % ax.get_title(), dpi=200, bbox_inches='tight')
        plt.close()
        
        headerline = "%s, Measured %s, Predicted %s" % (xlabel, ylabel, ylabel)
        myarray = np.array([g_topred_xfield, g_topred_ydata, g_topred_Ypredict]).transpose()
        ptools.array_to_csv("%s_%s_prediction.csv" % (test_group_val,test_group_label), headerline, myarray)
        if not (std_data == None):
            headerline = "%s, Predicted %s" % (xlabel, ylabel)
            myarray = np.array([g_std_xfield, g_std_Ypredict]).transpose()
            ptools.array_to_csv("%s_%s_std_prediction.csv" % (test_group_val,test_group_label), headerline, myarray)
    return

def plot_overall(fit_data=None, 
                    topred_data=None,
                    std_data=None,
                    topred_Ypredict=None,
                    std_Ypredict=None,
                    group_field_name=None,
                    label_field_name=None,
                    filter_dict=None,
                    xlabel="Measured",
                    xfield=None,
                    ylabel="Predicted",
                    measerrfield=None):
    """
        fit_data <data_parser data object>: fitting data
        topred_data <data_parser data object>: data to be predicted
        std_data <data_parser data object>: standard data to be predicted
        All data_parser objects should already have x and y features set.
        topred_Ypredict <numpy array>: predictions already made
        std_Ypredict <numpy array>: predictions already made on standard data
        xfield <str>: field name for x field for plotting
        measerrfield <str>: field name for measured error field (optional)
        group_field_name <str>: field name for field over which to group
        label_field_name <str>: field name for field containing group labels
    """
    topred_groupdata = np.asarray(topred_data.get_data(group_field_name)).ravel()
    topred_ydata = np.asarray(topred_data.get_y_data()).ravel()
    topred_xfield = np.asarray(topred_data.get_data(xfield)).ravel()
    if not measerrfield == None:
        topred_measerr = np.asarray(topred_data.get_data(measerrfield)).ravel()

    if not label_field_name == None:
        labeldata = np.asarray(topred_data.get_data(label_field_name)).ravel()
    else:
        labeldata = np.zeros(len(topred_groupdata))

    if not (std_data == None):
        std_groupdata = np.asarray(std_data.get_data(group_field_name)).ravel()
        std_xfield = np.asarray(std_data.get_data(xfield)).ravel()
        if not label_field_name == None:
            std_labeldata = np.asarray(std_data.get_data(label_field_name)).ravel()
        else:
            std_labeldata = np.zeros(len(std_groupdata))
    
    matplotlib.rcParams.update({'font.size':18})
    plt.figure()
    plt.hold(True)
    if measerrfield == None:
        plt.scatter(topred_ydata, topred_Ypredict,
                   lw=0, label="prediction points", color = 'blue')
    else:
       plt.errorbar(topred_ydata, topred_Ypredict, xerr=topred_measerr, 
            linewidth=1,
            linestyle = "None", color="red",
            markerfacecolor='red' , marker='o',
            markersize=10, markeredgecolor="None") 
    plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim(), ls="--", c=".3")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig("overall_prediction", dpi=200, bbox_inches='tight')
    plt.close()

    headerline = "Group value, Group label, %s, %s, %s" % (xfield, xlabel, ylabel)
    myarray = np.array([topred_groupdata, labeldata, topred_xfield, topred_ydata, topred_Ypredict]).transpose()
    ptools.mixed_array_to_csv("overall_prediction.csv", headerline, myarray)
    if not (std_data == None):
        headerline = "Group value, Group label,%s,%s" % (xfield, ylabel)
        myarray = np.array([std_groupdata, std_labeldata, std_xfield, std_Ypredict]).transpose()
        ptools.mixed_array_to_csv("overall_std_prediction.csv", headerline, myarray)
    return
