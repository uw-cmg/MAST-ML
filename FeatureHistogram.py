import matplotlib
import matplotlib.pyplot as plt
import data_parser
import numpy as np
import data_analysis.printout_tools as ptools
import plot_data.plot_histogram as plothist
import plot_data.plot_xy as plotxy
import os
import portion_data.get_test_train_data as gttd

######
# Create a feature histogram
# Tam Mayeshiba 2017-03-24
######

def execute(model, data, savepath, 
        feature_field_name = None,
        group_field_name=None,
        label_field_name=None,
        xlabel=None,
        num_bins = 50,
        verbose = 0,
        mean_xlabel="",
        mean_ylabel="",
        *args, **kwargs):
    """Simple plot
        Args:
            feature_field_name <str>: Field name for numeric feature to 
                                        histogram.
            group_field_name <str>: Field name for numeric field for grouping 
                                    plots.
                                    If None, all data will be plotted.
            label_field_name <str>: Field name for string label field that
                                    goes with group field name. Leave as
                                    None if group field name is None. 
            xlabel <str>: x label
            num_bins <int>: Number of bins for histogram
    """
    num_bins = int(num_bins)
    if xlabel == None:
        xlabel = feature_field_name

    Xdata = np.asarray(data.get_x_data())
    Ydata = np.asarray(data.get_y_data()).ravel()
    xfeatures = data.x_features
    yfeature = data.y_feature

    feature_data = np.asarray(data.get_data(feature_field_name)).ravel()
    
    kwargs=dict()
    kwargs['num_bins'] = num_bins
    kwargs['savepath'] = savepath

    if group_field_name == None:
        kwargs['xlabel'] = xlabel
        plothist.simple_histogram(feature_data, **kwargs)
        headerline = "%s" % feature_field_name
        myarray = np.array([feature_data]).transpose()
        
        csvname = os.path.join(savepath,"%s.csv" % xlabel.replace(" ","_"))
        ptools.array_to_csv(csvname, headerline, myarray)
        return
    group_indices = gttd.get_field_logo_indices(data, group_field_name)
    if not (label_field_name == None):
        labeldata = np.asarray(data.get_data(label_field_name)).ravel()
    groups = list(group_indices.keys())
    means = list()
    stds = list()
    labels=list()
    okaygroups=list()
    for group in groups:
        fdata = feature_data[group_indices[group]["test_index"]]
        fdata = np.array(fdata,'float') #convert None's to numpy NaN
        if (verbose > 0):
            print("Group: %i" % group)
            print(fdata)
        if np.count_nonzero(np.isnan(fdata)) == len(fdata): #all NaN
            print("Group %i has no data. Skipping." % group)
            continue
        okaygroups.append(group)
        if not (label_field_name == None):
            labels.append(labeldata[group_indices[group]["test_index"][0]])
        else:
            labels.append("No label")
        gmean = np.nanmean(fdata) #ignore NaN in mean calculation
        gstd = np.nanstd(fdata)   #ignore NaN in std calculation
        means.append(gmean)
        stds.append(gstd)
    means = np.array(means)
    stds = np.array(stds)
    kwargs['xlabel'] = "Mean %s" % xlabel
    plothist.simple_histogram(means, **kwargs)
    kwargs['xlabel'] = "Std Devs %s" % xlabel
    plothist.simple_histogram(stds, **kwargs)

    kwargs2=dict()
    if mean_xlabel == "":
        mean_xlabel = "%s" % group_field_name
    if mean_ylabel == "":
        mean_ylabel = "Mean %s" % xlabel
    kwargs2['xlabel'] = mean_xlabel
    kwargs2['ylabel'] = mean_ylabel
    kwargs2['savepath'] = savepath
    kwargs2['yerr'] = stds
    kwargs2['plottype'] = "scatter"
    plotxy.single(okaygroups, means, **kwargs2)

    headerline = "Group,Label,Mean,StdDev"
    myarray = np.array([okaygroups,labels,means,stds]).transpose()
    csvname = os.path.join(savepath,"Means_%s.csv" % xlabel.replace(" ","_"))
    ptools.mixed_array_to_csv(csvname, headerline, myarray)
    return
