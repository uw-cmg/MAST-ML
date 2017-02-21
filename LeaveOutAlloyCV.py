import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from mean_error import mean_error
import csv
import matplotlib
import portion_data.get_test_train_data as gttd
import data_analysis.printout_tools as ptools
def execute(model, data, savepath, *args, **kwargs):
    if "test_name" in kwargs.keys():
        test_name = kwargs["test_name"]
    else:
        test_name = "Leave One Group Out"
    if "field_name" in kwargs.keys():
        field_name = kwargs["field_name"]
    else:
        field_name = "No field set"
    if "xlabel" in kwargs.keys():
        xlabel = kwargs["xlabel"]
    else:
        xlabel = "Group number"
    if "ylabel" in kwargs.keys():
        ylabel = kwargs["ylabel"]
    else:
        ylabel = "RMSE"
    if "title" in kwargs.keys():
        title = kwargs["title"]
    else:
        title = ""
    rms_list = list()
    group_value_list = list()
    me_list = list()
    group_list = list()

    Xdata = np.asarray(data.get_x_data())
    ydata = np.asarray(data.get_y_data()).ravel()
    fielddata = np.asarray(data.get_data(field_name)).ravel()

    indices = gttd.get_field_logo_indices(data, field_name)
    groups = list(indices.keys())
    groups.sort()
    for group in groups:
        train_index = indices[group]["train_index"]
        test_index = indices[group]["test_index"]
        test_group_val = fielddata[test_index[0]] #left-out group value
        model.fit(Xdata[train_index], ydata[train_index])
        Ypredict = model.predict(Xdata[test_index],ydata[test_index])

        rms = np.sqrt(mean_squared_error(Ypredict, np.array(data.get_y_data()).ravel()))
        me = mean_error(Ypredict, np.array(data.get_y_data()).ravel())
        rms_list.append(rms)
        group_value_list.append(test_group_val)
        me_list.append(me)
        group_list.append(group) #should be able to just use groups

    print('Mean RMSE: ', np.mean(rms_list))
    print('Mean Mean Error: ', np.mean(me_list))

    # graph rmse vs alloy 
    matplotlib.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.xticks(np.arange(0, max(group_list) + 1, 5)) #group_list is numeric
    ax.scatter(group_list, rms_list, color='black', s=10)
    #plot zero line
    ax.plot((0, max(group_list)), (0, 0), ls="--", c=".3") 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(title) > 0:
        ax.set_title(title)
    ax.text(.5, .90, 'Mean RMSE: {:.2f} MPa'.format(np.mean(rms_list)), transform=ax.transAxes, fontsize=0.85*matplotlib.rcParams['font.size'])
    for x in np.argsort(rms_list)[-5:]:
        ax.annotate(s = alloy_list[x],xy = (alloy_list[x], rms_list[x]))
    fig.savefig(test_name.replace(" ","_"), dpi=200, bbox_inches='tight')
    fig.clf()
    plt.close()

    csvname = "%s.csv" % test_name.replace(" ","_")
    headerline = "Group,Group value,RMSE"
    save_array = np.array([group_list, group_value_list, rms_list]).transpose()
    ptools.array_to_csv(csvname, headerline, save_array, fmtstr=None) 
    return
