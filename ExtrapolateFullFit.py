import matplotlib.pyplot as plt
import matplotlib
import data_parser
import numpy as np
import os
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import data_analysis.printout_tools as ptools
import portion_data.get_test_train_data as gttd
import FullFit

def execute(model, data, savepath, 
        test_csv = None,
        standard_csv = None,
        group_field_name = None,
        label_field_name = None,
        numeric_field_name = None,
        split_xfield = None,
        xlabel="Measured",
        ylabel="Predicted",
        split_xlabel = "X",
        split_ylabel = "Prediction",
        stepsize=1,
        measerrfield = None,
        plot_filter_out = "",
        fit_only_on_matched_groups = 0,
        *args,**kwargs):
    """Do full fit for extrapolation.
    """
    stepsize=float(stepsize)
    #get datasets
    ##training
    train_xdata = np.asarray(data.get_x_data())
    train_ydata = np.asarray(data.get_y_data()).ravel()
    ##testing
    if test_csv == None:
        raise ValueError("test_csv is None")
    test_data = data_parser.parse(test_csv)
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
            test_data.add_exclusive_filter(ffield, foperator, fval)
    test_data.set_y_feature(data.y_feature) # same feature as fitting data
    test_data.set_x_features(data.x_features) # same features as fitting data
    test_xdata = np.asarray(test_data.get_x_data())
    test_ydata = np.asarray(test_data.get_y_data()).ravel()
    if numeric_field_name == None: #help identify each point
        numeric_field_name = data.x_features[0]
    #
    kwargs_f = dict()
    kwargs_f['xlabel'] = xlabel
    kwargs_f['ylabel'] = ylabel
    kwargs_f['stepsize'] = stepsize
    kwargs_f['numeric_field_name'] = numeric_field_name
    kwargs_f['group_field_name'] = group_field_name
    kwargs_f['savepath'] = os.path.join(savepath,'test')
    if not (os.path.isdir(kwargs_f['savepath'])):
        os.mkdir(kwargs_f['savepath'])
    kwargs_f['do_pergroup_fits'] = 0 # NO per-group fitting
    kwargs_f['test_numericdata'] = np.asarray(test_data.get_data(numeric_field_name)).ravel()
    if not(group_field_name == None):
        kwargs_f['test_groupdata'] = np.asarray(test_data.get_data(group_field_name)).ravel()
        kwargs_f['train_groupdata'] = np.asarray(data.get_data(group_field_name)).ravel()
        if label_field_name == None:
            label_field_name = group_field_name
        kwargs_f['label_field_name'] = label_field_name
        kwargs_f['test_labeldata'] = np.asarray(test_data.get_data(label_field_name)).ravel()
    kwargs_f['full_xtrain'] = train_xdata
    kwargs_f['full_ytrain'] = train_ydata
    kwargs_f['full_xtest'] = test_xdata
    kwargs_f['full_ytest'] = test_ydata
    test_data_array = FullFit.do_full_fit(model, **kwargs_f)
    
    ##standard has no y
    if not(standard_csv == None):
        std_data = data_parser.parse(standard_csv)
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
                std_data.add_exclusive_filter(ffield, foperator, fval)
        std_data.set_y_feature(data.x_features[0]) # dummy y feature
        std_data.set_x_features(data.x_features) # same features as fitting data
        std_xdata = np.asarray(std_data.get_x_data())
        std_ydata = np.asarray(std_data.get_y_data()).ravel()
        kwargs_s = dict()
        kwargs_s['savepath'] = savepath
        kwargs_s['savepath'] = os.path.join(savepath,'standard')
        if not (os.path.isdir(kwargs_s['savepath'])):
            os.mkdir(kwargs_s['savepath'])
        kwargs_s['fitlabel'] = "standard"
        kwargs_s['do_pergroup_fits'] = 0 # NO per-group fitting
        kwargs_s['test_numericdata'] = np.asarray(std_data.get_data(numeric_field_name)).ravel()
        if not(group_field_name == None):
            kwargs_s['test_groupdata'] = np.asarray(std_data.get_data(group_field_name)).ravel()
            kwargs_s['train_groupdata'] = np.asarray(std.get_data(group_field_name)).ravel()
            if label_field_name == None:
                label_field_name = group_field_name
            kwargs_s['label_field_name'] = label_field_name
            kwargs_s['test_labeldata'] = np.asarray(std_data.get_data(label_field_name)).ravel()
        kwargs_s['full_xtrain'] = train_xdata
        kwargs_s['full_ytrain'] = train_ydata
        kwargs_s['full_xtest'] = std_xdata
        kwargs_s['full_ytest'] = std_ydata
        std_data_array = FullFit.do_full_fit(model, **kwargs_s)
    return

def execute_old(model, data, savepath, lwr_data, 
            topredict_data_csv = None, 
            standard_conditions_csv = None,
            group_field_name = None,
            label_field_name = None,
            xfield = None,
            xlabel = "X",
            ylabel = "Prediction",
            overall_xlabel = "Measured",
            overall_ylabel = "Predicted",
            measerrfield = None,
            plot_filter_out = "",
            fit_only_on_matched_groups = 0,
            *args, **kwargs):
    """
        Fits on data and extrapolates prediction on 
        topredict_data and standard_conditions.
        Args:
            model: model to use
            data <data_parser data object>: data for fitting
            topredict_data_csv <str>: csv name for dataset for predictions
                            (to replace "lwr_data")
            standard_conditions_csv <str>: csv name for dataset of 
                                standard conditions, especially to be used 
                                if there are not enough points
                                in topredict_data_csv
            group_field_name <str>: field name for grouping total fit results
            label_field_name <str>: field name for labels of grouping
            xfield <str>: field name for x value to plot fit results against
            xlabel <str>: x-axis label for individual plots
            ylabel <str>: y-axis label for individual plots
            overall_xlabel <str>: x-axis label for overall plot
            overall_ylabel <str>: y-axis label for overall plot
            measerrfield <str>: field name for measured error in topredict_data,
    """
    fit_Xdata = np.asarray(data.get_x_data())
    fit_ydata = np.asarray(data.get_y_data()).ravel()

    pd_path = os.path.abspath(topredict_data_csv) 
    topredict_data = data_parser.parse(pd_path)
    topredict_data.set_y_feature(data.y_feature) #same feature as fitting data 
    topredict_data.set_x_features(data.x_features)#same features as fitting data
    topredict_Xdata = np.asarray(topredict_data.get_x_data())
    topredict_ydata = np.asarray(topredict_data.get_y_data()).ravel()

    if not (standard_conditions_csv== None):
        std_path = os.path.abspath(standard_conditions_csv)
        std_data = data_parser.parse(std_path)
        std_data.set_y_feature(xfield) #dummy y feature
        std_data.set_x_features(data.x_features)
        std_Xdata = np.asarray(std_data.get_x_data())
    else:
        std_Ypredict = None
        std_data = None
    
    if int(fit_only_on_matched_groups) == 1:
        fit_index=list()
        fit_indices = gttd.get_field_logo_indices(data, group_field_name)
        fit_groups = list(fit_indices.keys())
        topredict_groupdata = np.asarray(topredict_data.get_data(group_field_name)).ravel()
        topredict_groups = np.unique(topredict_groupdata)
        match_groups = np.intersect1d(fit_groups, topredict_groups)
        for mgroup in match_groups:
            fit_index.extend(fit_indices[mgroup]["test_index"])
        fit_Xdata = fit_Xdata[fit_index]
        fit_ydata = fit_ydata[fit_index]

    model.fit(fit_Xdata, fit_ydata)
    #note that plot_filter_out only affects the plotting, not the model fitting

    Ypredict = model.predict(topredict_Xdata)
    
    if not (std_data == None):
        std_Ypredict = model.predict(std_Xdata)

    from plot_data import plot_by_groups as pbg
    pbg.plot_overall(fit_data=data, 
                    topred_data=topredict_data,
                    std_data=std_data,
                    topred_Ypredict=Ypredict,
                    std_Ypredict=std_Ypredict,
                    group_field_name=group_field_name,
                    label_field_name=label_field_name,
                    xlabel=overall_xlabel,
                    ylabel=overall_ylabel,
                    xfield=xfield,
                    measerrfield=measerrfield,
                    plot_filter_out=plot_filter_out,
                    only_fit_matches=0)
    
    pbg.plot_overall(fit_data=data, 
                    topred_data=topredict_data,
                    std_data=std_data,
                    topred_Ypredict=Ypredict,
                    std_Ypredict=std_Ypredict,
                    group_field_name=group_field_name,
                    label_field_name=label_field_name,
                    xlabel=overall_xlabel,
                    ylabel=overall_ylabel,
                    xfield=xfield,
                    measerrfield=measerrfield,
                    plot_filter_out=plot_filter_out,
                    only_fit_matches=1)
    
    pbg.plot_overall(fit_data=data, 
                    topred_data=topredict_data,
                    std_data=std_data,
                    topred_Ypredict=Ypredict,
                    std_Ypredict=std_Ypredict,
                    group_field_name=group_field_name,
                    label_field_name=label_field_name,
                    xlabel=overall_xlabel,
                    ylabel=overall_ylabel,
                    xfield=xfield,
                    measerrfield=measerrfield,
                    plot_filter_out=plot_filter_out,
                    only_fit_matches=2)


    pbg.plot_separate_groups_vs_xfield(fit_data=data, 
                    topred_data=topredict_data,
                    std_data=std_data,
                    topred_Ypredict=Ypredict,
                    std_Ypredict=std_Ypredict,
                    group_field_name=group_field_name,
                    label_field_name=label_field_name,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    xfield=xfield,
                    plot_filter_out=plot_filter_out)
    return
