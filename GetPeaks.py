import matplotlib
import matplotlib.pyplot as plt
import data_parser
import numpy as np
import data_analysis.printout_tools as ptools
import plot_data.plot_xy as plotxy
import os
import portion_data.get_test_train_data as gttd
import peakutils
from peakutils.plot import plot as pplot
####peakutils from https://bitbucket.org/lucashnegri/peakutils

#######
# Get peaks
# Tam Mayeshiba 2017-03-24
#######

def get_indexes(ydata, threshold, distance, peaks, valleys):
    """See execute for arguments
    """
    if peaks == 1:
        peak_indexes = peakutils.indexes(ydata,thres=threshold,min_dist=distance)
    if valleys == 1:
        valley_indexes = peakutils.indexes(-1.0*ydata,thres=threshold, min_dist=distance)
    if peaks == 0:
        indexes = valley_indexes
    else:
        indexes = peak_indexes
        if valleys == 1:
            indexes = np.append(indexes, valley_indexes)
    return indexes

def execute(model, data, savepath, 
        feature_field_name=None,
        x_field_name = None,
        group_field_name=None,
        label_field_name=None,
        threshold= 1.0,
        distance = 1.0,
        peaks = 1,
        valleys = 0,
        *args, **kwargs):
    """Get peaks in data
        Args:
            feature_field_name <str>: Field name for data to evaluate for peaks
            x_field_name <str>: X axis field
            group_field_name <str>: Field name for numeric field for grouping 
                                    plots.
                                    If None, all data will be plotted.
            label_field_name <str>: Field name for string label field that
                                    goes with group field name. Leave as
                                    None if group field name is None. 
            threshold <float>: Threshold for peaks
            distance <float>: Minimum distance apart peaks should be
            peaks <int>: 1 - Find peaks (default)
                        0 - do not find peaks
            valleys <int>: 1 - Find valleys 
                            0 - do not find valleys (default)
    """
    threshold=float(threshold)
    distance = float(distance)
    peaks = int(peaks)
    valleys = int(valleys)

    feature_data = np.asarray(data.get_data(feature_field_name)).ravel()
    x_data = np.asarray(data.get_data(x_field_name)).ravel()

    if (peaks == 0) and (valleys == 0):
        print("Nothing to evaluate! peaks and valleys input are both 0")
        print("Exiting.")
        return

    if not (group_field_name == None):    
        group_indices = gttd.get_field_logo_indices(data, group_field_name)
        groups = list(group_indices.keys())
    else:
        groups = list([0])
        group_indices=dict()
        group_indices[0]=dict()
        group_indices[0]["test_index"] = range(0, len(feature_data))
    if not (label_field_name == None):
        labeldata = np.asarray(data.get_data(label_field_name)).ravel()
    else:
        labeldata = np.zeros(len(feature_data))

    labels=list()

    for group in groups:
        gx_data = x_data[group_indices[group]["test_index"]]
        gfeature_data = feature_data[group_indices[group]["test_index"]]
        kwargs=dict()
        label = labeldata[group_indices[group]["test_index"][0]]
        labels.append(label)
        groupstr = "%i".zfill(3) % group
        if group_field_name == None:
            grouppath = savepath
        else:
            groupstr = groupstr + "_%s" % label
            grouppath = os.path.join(savepath, groupstr)
        if not os.path.isdir(grouppath):
            os.mkdir(grouppath)
        kwargs['savepath'] = grouppath
        gindexes = get_indexes(gfeature_data, threshold, distance, 
                                peaks, valleys)
        plt.figure()
        pplot(gx_data, gfeature_data, gindexes)
        plt.tight_layout()
        plt.savefig(os.path.join(grouppath,"peaks"))
        plt.close()
        headerline = "xcoord,peakval" 
        myarray = np.array([gx_data[gindexes],gfeature_data[gindexes]]).transpose()
        csvname = os.path.join(grouppath,"peak_data.csv")
        ptools.array_to_csv(csvname, headerline, myarray)
    return
