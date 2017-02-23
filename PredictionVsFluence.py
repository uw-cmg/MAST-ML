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

def execute(model, data, savepath, lwr_data, *args, **kwargs):
    """
        topredict_data_csv: csv with dataset for predictions
                        (to replace "lwr_data")
        standard_conditions_csv: csv with dataset of standard conditions,
                            especially to be used if there are not enough points
                            in topredict_data
    """
    dargs = dict()
    dargs["plot_filter"] = None #not implemented
    dargs["topredict_data_csv"] = "Raise error" 
    dargs["group_field_name"] = "Raise error"
    dargs["label_field_name"] = None
    dargs["standard_conditions_csv"] = None #dataset of standard conditions
    dargs["xlabel"] = "x"
    dargs["xfield"] = "Raise error"
    dargs["ylabel"] = "y"
    dargs["overall_xlabel"] = "Measured"
    dargs["overall_ylabel"] = "Predicted"
    for keyword in dargs.keys():
        if keyword in kwargs.keys():
            dargs[keyword] = kwargs[keyword]
        if dargs[keyword] == "Raise error":
            raise ValueError("Keyword %s has no value set. Raise error." % keyword)
    plot_filter = dargs["plot_filter"]
    topredict_data_csv = dargs["topredict_data_csv"]
    group_field_name = dargs["group_field_name"]
    label_field_name = dargs["label_field_name"]
    standard_conditions_csv = dargs["standard_conditions_csv"]
    xlabel = dargs["xlabel"]
    xfield = dargs["xfield"]
    ylabel = dargs["ylabel"]
    overall_xlabel = dargs["overall_xlabel"]
    overall_ylabel = dargs["overall_ylabel"]

    fit_Xdata = np.asarray(data.get_x_data())
    fit_ydata = np.asarray(data.get_y_data()).ravel()

    model.fit(fit_Xdata, fit_ydata)
    #note that plot_filter only affects the plotting, not the model fitting

    fit_indices = gttd.get_field_logo_indices(data, group_field_name)
    fit_xfield = np.asarray(data.get_data(xfield)).ravel()
    fit_groupdata = np.asarray(data.get_data(group_field_name)).ravel()

    pd_path = os.path.abspath(topredict_data_csv) 
    topredict_data = data_parser.parse(pd_path)
    topredict_data.set_y_feature(data.y_feature) #same feature as fitting data 
    topredict_data.set_x_features(data.x_features)#same features as fitting data
    indices = gttd.get_field_logo_indices(topredict_data, group_field_name)
    groupdata = np.asarray(topredict_data.get_data(group_field_name)).ravel()
    topredict_xfield = np.asarray(topredict_data.get_data(xfield)).ravel()
    if not label_field_name == None:
        labeldata = np.asarray(topredict_data.get_data(label_field_name)).ravel()
    
    topredict_Xdata = np.asarray(topredict_data.get_x_data())
    topredict_ydata = np.asarray(topredict_data.get_y_data()).ravel()

    if not (standard_conditions_csv== None):
        std_path = os.path.abspath(standard_conditions_csv)
        std_data = data_parser.parse(std_path)
        std_data.set_y_feature(xfield) #dummy y feature
        std_data.set_x_features(data.x_features)
        std_Xdata = np.asarray(std_data.get_x_data())
        std_xfield = np.asarray(std_data.get_data(xfield)).ravel()
        std_groupdata = np.asarray(std_data.get_data(group_field_name)).ravel()
        std_indices = gttd.get_field_logo_indices(std_data, group_field_name)
    
    Ypredict_overall=np.zeros(len(topredict_ydata))
    groups = list(indices.keys())
    groups.sort()
    for group in groups:
        print(group)
        train_index = indices[group]["train_index"]
        test_index = indices[group]["test_index"]
        test_group_val = groupdata[test_index[0]] #left-out group value
        if label_field_name == None:
            test_group_label = "None"
        else:
            test_group_label = labeldata[test_index[0]]
        Ypredict = model.predict(topredict_Xdata[test_index])
        Ypredict_overall[test_index] = Ypredict

        if standard_conditions_csv== None:
            std_x = np.copy(topredict_xfield[test_index])
            std_Ypredict = np.copy(Ypredict)
        else:
            #need to find a better way of matching groups
            for sgroup in std_indices.keys():
                s_test_index = std_indices[sgroup]["test_index"] 
                if std_groupdata[s_test_index[0]] == test_group_val:
                    matchgroup = sgroup
                    continue
            std_test_index = std_indices[matchgroup]["test_index"]
            std_x = std_xfield[std_test_index]
            std_Ypredict = model.predict(std_Xdata[std_test_index])

        for fgroup in fit_indices.keys():
            f_test_index = fit_indices[fgroup]["test_index"] 
            if fit_groupdata[f_test_index[0]] == test_group_val:
                fmatchgroup = fgroup
                continue
        fit_test_index = fit_indices[fmatchgroup]["test_index"]

        plt.figure()
        plt.hold(True)
        fig, ax = plt.subplots()
        matplotlib.rcParams.update({'font.size':18})
        ax.plot(std_x, std_Ypredict,
                lw=3, color='#ffc04d', label="Prediction")
        ax.scatter(fit_xfield[fit_test_index], fit_ydata[fit_test_index],
               lw=0, label="Fitting data", color = 'black')
        ax.scatter(topredict_xfield[test_index], topredict_ydata[test_index],
               lw=0, label="Measured data", color = '#7ec0ee')
        ax.scatter(topredict_xfield[test_index], Ypredict,
               lw=0, label="Prediction points", color = 'blue')

        plt.legend(loc = "upper left", fontsize=matplotlib.rcParams['font.size']) #data is sigmoid; 'best' can block data
        plt.title("%s(%s)" % (test_group_val, test_group_label))
        plt.xlabel("$%s$" % xlabel)
        plt.ylabel("$%s$" % ylabel)
        plt.savefig(savepath.format("%s_prediction" % ax.get_title()), dpi=200, bbox_inches='tight')
        plt.close()
        
        headerline = "%s, Measured %s, Predicted %s" % (xlabel, ylabel, ylabel)
        myarray = np.array([topredict_xfield[test_index], groupdata[test_index], Ypredict]).transpose()
        ptools.array_to_csv("%s_%s_pred.csv" % (test_group_val,test_group_label), headerline, myarray)
        if not (standard_conditions_csv== None):
            headerline = "%s, Predicted %s" % (xlabel, ylabel)
            myarray = np.array([std_x, std_Ypredict]).transpose()
            ptools.array_to_csv("%s_%s_std_pred.csv" % (test_group_val,test_group_label), headerline, myarray)

    plt.figure()
    plt.scatter(topredict_ydata, Ypredict_overall,
               lw=0, label="prediction points", color = 'blue')
    plt.xlabel(overall_xlabel)
    plt.ylabel(overall_ylabel)
    plt.savefig(savepath.format("overall_prediction"), dpi=200, bbox_inches='tight')
    plt.close()
    return
