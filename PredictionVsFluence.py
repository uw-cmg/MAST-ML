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
    
    Ypredict = model.predict(topredict_Xdata)
    std_Ypredict = model.predict(std_Xdata)

    from plot_data import plot_by_groups as pbg
    pbg.plot_by_groups(fit_data=data, 
                    topred_data=topredict_data,
                    std_data=std_data,
                    topred_Ypredict=Ypredict,
                    std_Ypredict=std_Ypredict,
                    group_field_name=group_field_name,
                    label_field_name=label_field_name,
                    filter_dict=None,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    xfield=xfield,
                    overall_xlabel=overall_xlabel,
                    overall_ylabel=overall_ylabel)
    return
