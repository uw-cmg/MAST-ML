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
Plot CD predictions of ∆sigma for data and lwr_data as well as model's output
model is trained to data, which is in the domain of IVAR, IVAR+/ATR1 (assorted fluxes and fluences)
lwr_data is data in the domain of LWR conditions (low flux, high fluence)
'''

def execute(model, data, savepath, lwr_data, *args, **kwargs):
    if "temp_filter" in kwargs.keys():
        temp_filter = int(kwargs["temp_filter"]) #int matches data
        raise ValueError("currently not working")
    else:
        temp_filter = None
    if "field_name" in kwargs.keys():
        field_name = kwargs["field_name"]
    else:
        field_name = "No field set"
        raise ValueError("field_name not in kwargs. %s" % field_name)
    if "label_field_name" in kwargs.keys():
        label_field_name = kwargs["label_field_name"]
    else:
        label_field_name = None
    if "standard_conditions" in kwargs.keys():
        standard_conditions = kwargs["standard_conditions"]
    else:
        standard_conditions = None

    Xdata_fit = np.asarray(data.get_x_data())
    ydata_fit = np.asarray(data.get_y_data()).ravel()

    model.fit(Xdata_fit, ydata_fit)
    #note that temp_filter only affects the plotting, not the model fitting

    fluence_str = "log(fluence_n_cm2)"
    flux_str = "log(flux_n_cm2_sec)"
    eff_str = "log(eff fl 100p=26)"
    temp_str = "temperature_C"

    indices = gttd.get_field_logo_indices(lwr_data, field_name)
    fielddata = np.asarray(lwr_data.get_data(field_name)).ravel()
    eff_fluence_data = np.asarray(lwr_data.get_data(eff_str)).ravel()
    if not label_field_name == None:
        labeldata = np.asarray(lwr_data.get_data(label_field_name)).ravel()
    
    Xdata = np.asarray(lwr_data.get_x_data())
    ydata = np.asarray(lwr_data.get_y_data()).ravel()

    if not (standard_conditions == None):
        std_path = os.path.abspath(standard_conditions)
        std_data = data_parser.parse(std_path)
        std_data.set_y_feature(eff_str) #dummy y feature
        std_data.set_x_features(data.x_features)
        std_Xdata = np.asarray(std_data.get_x_data())
        eff_fluence_std = np.asarray(std_data.get_data(eff_str)).ravel()
        std_fielddata = np.asarray(std_data.get_data(field_name)).ravel()
        std_indices = gttd.get_field_logo_indices(std_data, field_name)

    groups = list(indices.keys())
    groups.sort()
    for group in groups:
        print(group)
        train_index = indices[group]["train_index"]
        test_index = indices[group]["test_index"]
        test_group_val = fielddata[test_index[0]] #left-out group value
        if label_field_name == None:
            test_group_label = "None"
        else:
            test_group_label = labeldata[test_index[0]]
        model.fit(Xdata[train_index], ydata[train_index])
        Ypredict = model.predict(Xdata[test_index])

        if standard_conditions == None:
            line_data = np.copy(eff_fluence_data[test_index])
            Ypredictline = np.copy(Ypredict)
        else:
            #need to find a better way of matching groups
            for sgroup in std_indices.keys():
                s_test_index = std_indices[sgroup]["test_index"] 
                if std_fielddata[s_test_index[0]] == test_group_val:
                    matchgroup = sgroup
                    continue
            std_test_index = std_indices[matchgroup]["test_index"]
            line_data = eff_fluence_std[std_test_index]
            Ypredictline = model.predict(std_Xdata[std_test_index])

        plt.figure()
        plt.hold(True)
        fig, ax = plt.subplots()
        matplotlib.rcParams.update({'font.size':18})
        ax.plot(line_data, Ypredictline,
                lw=3, color='#ffc04d', label="LWR prediction")
        #also want to plot TRAIN data; get groupings separately?
        #            lw=0, label='IVAR data',
        #           color='black')   
        ax.scatter(eff_fluence_data[test_index], fielddata[test_index],
               lw=0, label="CD LWR data", color = '#7ec0ee')
        ax.scatter(eff_fluence_data[test_index], Ypredict,
               lw=0, label="LWR prediction points", color = 'blue')

        plt.legend(loc = "upper left", fontsize=matplotlib.rcParams['font.size']) #data is sigmoid; 'best' can block data
        plt.title("%s(%s)" % (test_group_val, test_group_label))
        plt.xlabel("log(Eff Fluence(n/cm$^{2}$))")
        plt.ylabel("$\Delta\sigma_{y}$ (MPa)")
        plt.savefig(savepath.format("%s_LWR" % ax.get_title()), dpi=200, bbox_inches='tight')
        plt.close()
        
        headerline = "logEffFluence LWR, Points LWR, Predicted LWR"
        myarray = np.array([eff_fluence_data[test_index], fielddata[test_index], Ypredict]).transpose()
        ptools.array_to_csv("%s_%s_LWR.csv" % (test_group_val,test_group_label), headerline, myarray)
        if not (standard_conditions == None):
            headerline = "logEffFluence LWR, Predicted LWR"
            myarray = np.array([line_data, Ypredictline]).transpose()
            ptools.array_to_csv("%s_%s_LWRpred.csv" % (test_group_val,test_group_label), headerline, myarray)
    return
