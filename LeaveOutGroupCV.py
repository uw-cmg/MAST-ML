import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from mean_error import mean_error
import csv
import matplotlib
import portion_data.get_test_train_data as gttd
import data_analysis.printout_tools as ptools
import plot_data.plot_rmse as plotrmse

def execute(model, data, savepath, 
            field_name=None,
            label_field_name=None,
            xlabel="Group number",
            ylabel="RMSE",
            title="",
            *args, **kwargs):
    if field_name == None:
        raise ValueError("No field name set.")

    rms_list = list()
    group_value_list = list()
    group_label_list=list()
    me_list = list()
    group_list = list()

    Xdata = np.asarray(data.get_x_data())
    ydata = np.asarray(data.get_y_data()).ravel()
    fielddata = np.asarray(data.get_data(field_name)).ravel()
    if not label_field_name == None:
        labeldata = np.asarray(data.get_data(label_field_name)).ravel()

    indices = gttd.get_field_logo_indices(data, field_name)
    
    groups = list(indices.keys())
    groups.sort()
    for group in groups:
        train_index = indices[group]["train_index"]
        test_index = indices[group]["test_index"]
        test_group_val = fielddata[test_index[0]] #left-out group value
        if label_field_name == None:
            test_group_label = "None"
        else:
            test_group_label = labeldata[test_index[0]]
        model.fit(Xdata[train_index], ydata[train_index])
        Ypredict = model.predict(Xdata[test_index])

        rms = np.sqrt(mean_squared_error(Ypredict, ydata[test_index]))
        me = mean_error(Ypredict, ydata[test_index])
        rms_list.append(rms)
        group_value_list.append(test_group_val)
        group_label_list.append(test_group_label)
        me_list.append(me)
        group_list.append(group) #should be able to just use groups

    print('Mean RMSE: ', np.mean(rms_list))
    print('Mean Mean Error: ', np.mean(me_list))

    notelist=list()
    notelist.append("Mean RMSE: {:.2f}".format(np.mean(rms_list)))
    
    kwargs=dict()
    kwargs['xlabel'] = xlabel
    kwargs['ylabel'] = ylabel
    kwargs['title'] = title
    kwargs['notelist'] = notelist
    kwargs['savepath'] = savepath
    if label_field_name == None:
        kwargs['group_label_list'] = None
    else:
        kwargs['group_label_list'] = group_label_list

    plotrmse.vs_leftoutgroup(group_value_list, rms_list, **kwargs)

    csvname = "leavegroupout.csv"
    headerline = "Group,Group value,Group label,RMSE,Mean error"
    save_array = np.asarray([group_list, group_value_list, group_label_list, rms_list, me_list]).transpose()
    ptools.mixed_array_to_csv(csvname, headerline, save_array) 
    return
