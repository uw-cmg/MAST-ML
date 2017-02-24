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
from mean_error import mean_error
import data_analysis.printout_tools as ptools
import portion_data.get_test_train_data as gttd

def plot_separate_groups_vs_xfield(fit_data=None, 
                    topred_data=None,
                    std_data=None,
                    topred_Ypredict=None,
                    std_Ypredict=None,
                    group_field_name=None,
                    label_field_name=None,
                    xlabel="x",
                    ylabel="y",
                    xfield=None,
                    plot_filter_out=""):
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
        plot_filter_out <str>: semicolon-delimited list of
                        field name, operator, value triplets for filtering
                        out data
    """
    predfield = "__predicted"
    topred_data.add_feature(predfield,topred_Ypredict)
    std_data.add_feature(predfield,std_Ypredict)
    if len(plot_filter_out) > 0:
        ftriplets = plot_filter_out.split(";")
        for ftriplet in ftriplets:
            fpcs = ftriplet.split(",")
            ffield = fpcs[0].strip()
            foperator = fpcs[1].strip()
            fval = fpcs[2].strip()
            try:
                fval = float(fval)
            except (ValueError, TypeError):
                pass
            fit_data.add_exclusive_filter(ffield, foperator, fval)
            topred_data.add_exclusive_filter(ffield, foperator, fval)
            if not (std_data == None):
                std_data.add_exclusive_filter(ffield, foperator, fval)

    fit_indices = gttd.get_field_logo_indices(fit_data, group_field_name)
    fit_xfield = np.asarray(fit_data.get_data(xfield)).ravel()
    fit_groupdata = np.asarray(fit_data.get_data(group_field_name)).ravel()
    fit_ydata = np.asarray(fit_data.get_y_data()).ravel()

    topred_indices = gttd.get_field_logo_indices(topred_data, group_field_name)
    topred_groupdata = np.asarray(topred_data.get_data(group_field_name)).ravel()
    topred_ydata = np.asarray(topred_data.get_y_data()).ravel()
    topred_predicted = np.asarray(topred_data.get_data(predfield)).ravel()

    topred_xfield = np.asarray(topred_data.get_data(xfield)).ravel()
    if not label_field_name == None:
        labeldata = np.asarray(topred_data.get_data(label_field_name)).ravel()
    else:
        labeldata = np.zeros(len(topred_groupdata))

    if not (std_data == None):
        std_indices = gttd.get_field_logo_indices(std_data, group_field_name)
        std_xfield = np.asarray(std_data.get_data(xfield)).ravel()
        std_groupdata = np.asarray(std_data.get_data(group_field_name)).ravel()
        std_predicted = np.asarray(std_data.get_data(predfield)).ravel()
    
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
        g_topred_predicted = topred_predicted[test_index]
        g_topred_ydata = topred_ydata[test_index]
        smatchgroup=-1
        if std_data == None:
            smatchgroup=0
            g_std_xfield = np.copy(g_topred_xfield)
            g_std_predicted = np.copy(g_topred_predicted)
        else:
            #need to find a better way of matching groups
            for sgroup in std_indices.keys():
                s_test_index = std_indices[sgroup]["test_index"] 
                if std_groupdata[s_test_index[0]] == test_group_val:
                    smatchgroup = sgroup
                    continue
        if smatchgroup > -1:
            std_test_index = std_indices[smatchgroup]["test_index"]
            g_std_xfield = std_xfield[std_test_index]
            g_std_predicted = std_predicted[std_test_index]

        fmatchgroup = -1
        for fgroup in fit_indices.keys():
            f_test_index = fit_indices[fgroup]["test_index"] 
            if fit_groupdata[f_test_index[0]] == test_group_val:
                fmatchgroup = fgroup
                continue
        if fmatchgroup > -1:
            fit_test_index = fit_indices[fmatchgroup]["test_index"]
            g_fit_xfield = fit_xfield[fit_test_index]
            g_fit_ydata = fit_ydata[fit_test_index]

        fig = plt.figure()
        plt.hold(True)
        matplotlib.rcParams.update({'font.size':18})
        if fmatchgroup > -1:
            plt.plot(g_fit_xfield, g_fit_ydata,
                    markersize=10, marker='o', markeredgecolor='black',
                    markerfacecolor = 'black', markeredgewidth=3,
                    linestyle='None',
                    label="Subset of fitting data")
        if smatchgroup > -1:
            plt.plot(g_std_xfield, g_std_predicted,
                    #marker='o', markersize=10,
                    lw=3, color='blue', label="Prediction")
        plt.plot(g_topred_xfield, g_topred_predicted,
                    markersize=20, marker=(8,2,0), markeredgecolor='blue',
                    markerfacecolor='None', markeredgewidth=3,
                    linestyle='None',
                    label="Predicted data")
        plt.plot(g_topred_xfield, g_topred_ydata,
                    markersize=20, marker='o', markeredgecolor='red',
                    markerfacecolor='None', markeredgewidth=3,
                    linestyle='None',
                    label='Measured data')
        lgd=plt.legend(loc = "upper left", 
                        fontsize=matplotlib.rcParams['font.size'], 
                        numpoints=1,
                        fancybox=True) 
        lgd.get_frame().set_alpha(0.5) #translucent legend!
        titlestr = "%s(%s)" % (test_group_val, test_group_label)
        plt.title(titlestr)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig("%s_prediction" % titlestr, dpi=200, bbox_inches='tight')
        plt.hold(False)
        plt.close(fig)
        
        headerline = "%s, Measured %s, Predicted %s" % (xlabel, ylabel, ylabel)
        myarray = np.array([g_topred_xfield, g_topred_ydata, g_topred_predicted]).transpose()
        ptools.array_to_csv("%s_%s_prediction.csv" % (test_group_val,test_group_label), headerline, myarray)
        if not (std_data == None):
            headerline = "%s, Predicted %s" % (xlabel, ylabel)
            myarray = np.array([g_std_xfield, g_std_predicted]).transpose()
            ptools.array_to_csv("%s_%s_std_prediction.csv" % (test_group_val,test_group_label), headerline, myarray)

    fit_data.remove_all_filters()
    std_data.remove_all_filters()
    topred_data.remove_all_filters()
    return

def plot_overall(fit_data=None, 
                    topred_data=None,
                    std_data=None,
                    topred_Ypredict=None,
                    std_Ypredict=None,
                    group_field_name=None,
                    label_field_name=None,
                    xlabel="Measured",
                    xfield=None,
                    ylabel="Predicted",
                    measerrfield=None,
                    plot_filter_out="",
                    only_fit_matches=0,
                    ):
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
        plot_filter_out <str>: semicolon-delimited list of
                        field name, operator, value triplets for filtering
                        out data
        only_fit_matches <int or str>: 0 - plot all predicted data, even if the
                                group was not present in the fitting data
                                1 - only plot predicted data whose group was
                                present in the fitting data
    """
    predfield = "__predicted"
    topred_data.add_feature(predfield,topred_Ypredict)
    std_data.add_feature(predfield,std_Ypredict)
    if len(plot_filter_out) > 0:
        ftriplets = plot_filter_out.split(";")
        for ftriplet in ftriplets:
            fpcs = ftriplet.split(",")
            ffield = fpcs[0].strip()
            foperator = fpcs[1].strip()
            fval = fpcs[2].strip()
            try:
                fval = float(fval)
            except (ValueError, TypeError):
                pass
            fit_data.add_exclusive_filter(ffield, foperator, fval)
            topred_data.add_exclusive_filter(ffield, foperator, fval)
            if not (std_data == None):
                std_data.add_exclusive_filter(ffield, foperator, fval)
    
    if int(only_fit_matches) == 1:
        initial_fit_groupdata = np.asarray(fit_data.get_data(group_field_name)).ravel()
        initial_topred_groupdata = np.asarray(topred_data.get_data(group_field_name)).ravel()
        initial_std_groupdata = np.asarray(std_data.get_data(group_field_name)).ravel()
        unique_fit_groups = np.unique(initial_fit_groupdata)
        unique_topred_groups = np.unique(initial_topred_groupdata)
        unique_std_groups = np.unique(initial_std_groupdata)
        topred_nonmatch_groups = np.setdiff1d(unique_topred_groups, unique_fit_groups)
        std_nonmatch_groups = np.setdiff1d(unique_std_groups, unique_fit_groups)
        for nonmatch in topred_nonmatch_groups:
            topred_data.add_exclusive_filter(group_field_name,'=',nonmatch)
        for nonmatch in std_nonmatch_groups:
            std_data.add_exclusive_filter(group_field_name,'=',nonmatch)
    elif int(only_fit_matches) == 0:
        pass
    else:
        raise ValueError("%s is not a valid value for only_fit_matches")

    topred_groupdata = np.asarray(topred_data.get_data(group_field_name)).ravel()
    topred_ydata = np.asarray(topred_data.get_y_data()).ravel()
    topred_xfield = np.asarray(topred_data.get_data(xfield)).ravel()
    topred_predicted = np.asarray(topred_data.get_data(predfield)).ravel()
    if not measerrfield == None:
        topred_measerr = np.asarray(topred_data.get_data(measerrfield)).ravel()

    if not label_field_name == None:
        labeldata = np.asarray(topred_data.get_data(label_field_name)).ravel()
    else:
        labeldata = np.zeros(len(topred_groupdata))

    if not (std_data == None):
        std_groupdata = np.asarray(std_data.get_data(group_field_name)).ravel()
        std_xfield = np.asarray(std_data.get_data(xfield)).ravel()
        std_predicted = np.asarray(std_data.get_data(predfield)).ravel()
        if not label_field_name == None:
            std_labeldata = np.asarray(std_data.get_data(label_field_name)).ravel()
        else:
            std_labeldata = np.zeros(len(std_groupdata))
   
    #annotations
    rmse = np.sqrt(mean_squared_error(topred_ydata, topred_predicted))
    rmsestr = "RMS error: %3.2f" % rmse
    print(rmsestr)
    meanerr = mean_error(topred_predicted, topred_ydata) #ordering matters!
    meanstr = "Mean error: %3.2f" % meanerr
    print(meanstr)
    #
    matplotlib.rcParams.update({'font.size':18})
    smallfont = 0.90*matplotlib.rcParams['font.size']
    plt.figure()
    plt.hold(True)
    if measerrfield == None:
        plt.plot(topred_ydata, topred_predicted,
                    linestyle='None',
                    marker='o', markeredgewidth=2,
                    markeredgecolor='blue', markerfacecolor='None',
                    markersize=15)
    else:
        darkred="#8B0000"
        (_, caps, _) = plt.errorbar(topred_ydata, topred_predicted, 
            xerr=topred_measerr, 
            linewidth=2,
            linestyle = "None", color=darkred,
            markeredgewidth=2, markeredgecolor=darkred,
            markerfacecolor='red' , marker='o',
            markersize=15)
        for cap in caps:
            cap.set_color(darkred)
            cap.set_markeredgewidth(2)
        #http://stackoverflow.com/questions/7601334/how-to-set-the-line-width-of-error-bar-caps-in-matplotlib
    #print dashed dividing line, but do not overextend axes
    (ymin, ymax) = plt.gca().get_ylim()
    (xmin, xmax) = plt.gca().get_xlim()
    minmin = min(xmin, ymin)
    maxmax = max(xmax, ymax)
    plt.plot((minmin+1, maxmax-1), (minmin+1, maxmax-1), ls="--", c=".3")
    plt.annotate(rmsestr, xy=(0.05, 0.90), xycoords="axes fraction", fontsize=smallfont)
    plt.annotate(meanstr, xy=(0.05, 0.83), xycoords="axes fraction", fontsize=smallfont)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    savestr = "overall_prediction"
    if int(only_fit_matches) > 0:
        savestr = savestr + "_only_fit_matches"
    plt.savefig(savestr, dpi=200, bbox_inches='tight')
    plt.close()

    headerline = "Group value, Group label, %s, %s, %s" % (xfield, xlabel, ylabel)
    myarray = np.array([topred_groupdata, labeldata, topred_xfield, topred_ydata, topred_predicted]).transpose()
    ptools.mixed_array_to_csv("%s.csv" % savestr, headerline, myarray)
    if not (std_data == None):
        headerline = "Group value, Group label,%s,%s" % (xfield, ylabel)
        myarray = np.array([std_groupdata, std_labeldata, std_xfield, std_predicted]).transpose()
        ptools.mixed_array_to_csv("%s_std.csv" % savestr, headerline, myarray)
    fit_data.remove_all_filters()
    std_data.remove_all_filters()
    topred_data.remove_all_filters()
    return
