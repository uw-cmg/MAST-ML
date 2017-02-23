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


'''
Plot CD predictions of ∆sigma for data and topredict_data as well as model's output
model is trained to data, which is in the domain of IVAR, IVAR+/ATR1 (assorted fluxes and fluences)
topredict_data is data in the domain of LWR conditions (low flux, high fluence)
'''

def plot_by_groups(fit_data=None, 
                    topred_data=None,
                    std_data=None,
                    topred_Ypredict=None,
                    std_Ypredict=None,
                    group_field_name=None,
                    label_field_name=None,
                    filter_dict=None,
                    xlabel="x",
                    ylabel="y",
                    xfield=None,
                    overall_xlabel="Measured",
                    overall_ylabel="Predicted"):
    """
        fit_data <data_parser data object>: fitting data
        topred_data <data_parser data object>: data to be predicted
        std_data <data_parser data object>: standard data to be predicted
        All data_parser objects should already have x and y features set.
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
        plt.xlabel("$%s$" % xlabel)
        plt.ylabel("$%s$" % ylabel)
        plt.savefig("%s_prediction" % ax.get_title(), dpi=200, bbox_inches='tight')
        plt.close()
        
        headerline = "%s, Measured %s, Predicted %s" % (xlabel, ylabel, ylabel)
        myarray = np.array([g_topred_xfield, g_topred_ydata, g_topred_Ypredict]).transpose()
        ptools.array_to_csv("%s_%s_pred.csv" % (test_group_val,test_group_label), headerline, myarray)
        if not (std_data == None):
            headerline = "%s, Predicted %s" % (xlabel, ylabel)
            myarray = np.array([g_std_xfield, g_std_Ypredict]).transpose()
            ptools.array_to_csv("%s_%s_std_pred.csv" % (test_group_val,test_group_label), headerline, myarray)

    plt.figure()
    plt.scatter(topred_ydata, topred_Ypredict,
               lw=0, label="prediction points", color = 'blue')
    plt.xlabel(overall_xlabel)
    plt.ylabel(overall_ylabel)
    plt.savefig(savepath.format("overall_prediction"), dpi=200, bbox_inches='tight')
    plt.close()

    headerline = "Group val, Group label, %s, Measured %s, Predicted %s" % (xlabel, ylabel, ylabel)
    myarray = np.array([topred_groupdata, labeldata, topred_xfield, topred_ydata, topred_YPredict]).transpose()
    ptools.array_to_csv("overall_prediction.csv", headerline, myarray)
    if not (std_data == None):
        headerline = "%s, Predicted %s" % (xlabel, ylabel)
        myarray = np.array([std_xfield, std_Ypredict]).transpose()
        ptools.array_to_csv("overall_std_prediction.csv", headerline, myarray)
    return
