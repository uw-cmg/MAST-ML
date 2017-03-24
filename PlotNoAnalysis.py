import matplotlib
import matplotlib.pyplot as plt
import data_parser
import numpy as np
import data_analysis.printout_tools as ptools
import plot_data.plot_xy as plotxy
import os
import portion_data.get_test_train_data as gttd

def execute(model, data, savepath, 
        plottype="scatter",
        group_field_name=None,
        label_field_name=None,
        flipaxes=0,
        do_fft=0,
        *args, **kwargs):
    """Simple plot
        Args:
            plottype <str>: "line" or "scatter"
            group_field_name <str>: Field name for numeric field for grouping 
                                    plots.
                                    If None, all data will be plotted.
            label_field_name <str>: Field name for string label field that
                                    goes with group field name. Leave as
                                    None if group field name is None. 
            flipaxes <int>: 0, plot y features against multiple X features
                                (default)
                           1, plot each given "X" feature as the y against
                                a single "y" feature as the x
            do_fft <int>: 0, no fast Fourier transform (default)
                        1, fast Fourier transform as well
    """
    flipaxes = int(flipaxes)
    do_fft = int(do_fft)
    
    Xdata = np.asarray(data.get_x_data())
    Ydata = np.asarray(data.get_y_data()).ravel()
    xfeatures = data.x_features
    yfeature = data.y_feature

    xshape = Xdata.shape
    if len(xshape) > 1:
        numplots = xshape[1]
    else:
        numplots = 1

    if not (group_field_name == None):    
        group_indices = gttd.get_field_logo_indices(data, group_field_name)
        if not (label_field_name == None):
            labeldata = np.asarray(data.get_data(label_field_name)).ravel()
        groups = list(group_indices.keys())
    else:
        groups = list([0])
        group_indices=dict()
        group_indices[0]=dict()
        group_indices[0]["test_index"] = range(0, len(Ydata))

    for numplot in range(0, numplots):
        for group in groups:
            xdata = Xdata[group_indices[group]["test_index"],numplot]
            ydata = Ydata[group_indices[group]["test_index"]]
            kwargs=dict()
            groupstr = "%i".zfill(3) % group
            if group_field_name == None:
                grouppath = savepath
            else:
                if not (label_field_name == None):
                    label = labeldata[group_indices[group]["test_index"][0]]
                    groupstr = groupstr + "_%s" % label
                grouppath = os.path.join(savepath, groupstr)
            if not os.path.isdir(grouppath):
                os.mkdir(grouppath)
            kwargs['savepath'] = grouppath
            kwargs['plottype'] = plottype
            if flipaxes == 0:
                kwargs['xlabel'] = xfeatures[numplot]
                kwargs['ylabel'] = yfeature
                plotxy.single(xdata, ydata, **kwargs)
                if do_fft == 1:
                    kwargs['ylabel'] = yfeature + "_fft"
                    plotxy.single(xdata, fft(ydata), **kwargs)
            else:
                kwargs['xlabel'] = yfeature
                kwargs['ylabel'] = xfeatures[numplot]
                plotxy.single(ydata, xdata, **kwargs)
                if do_fft == 1:
                    kwargs['ylabel'] = xfeatures[numplot] + "_fft"
                    plotxy.single(ydata, fft(xdata), **kwargs)
    return
