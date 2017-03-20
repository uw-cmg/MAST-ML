import matplotlib
import matplotlib.pyplot as plt
import data_parser
import numpy as np
import data_analysis.printout_tools as ptools
import plot_data.plot_xy as plotxy
import portion_data.get_test_train_data as gttd
import os
def execute(model, data, savepath, 
        plottype="scatter",
        group_field_name=None,
        label_field_name=None,
        flipaxes=0,
        *args, **kwargs):
    """Simple plot
        Args:
            flipaxes <int>: 0, plot y features against multiple X features
                                (default)
                           1, plot each given "X" feature as the y against
                                a single "y" feature as the x
    """
    flipaxes = int(flipaxes)
    
    Xdata = np.asarray(data.get_x_data())
    Ydata = np.asarray(data.get_y_data()).ravel()
    xfeatures = data.x_features
    yfeature = data.y_feature

    xshape = Xdata.shape
    if len(xshape) > 1:
        numplots = xshape[1]
    else:
        numplots = 1
    
    group_indices = gttd.get_field_logo_indices(data, group_field_name)
    if not (label_field_name == None):
        labeldata = np.asarray(data.get_data(label_field_name)).ravel()
    groups = list(group_indices.keys())

    for numplot in range(0, numplots):
        for group in groups:
            xdata = Xdata[group_indices[group]["test_index"],numplot]
            ydata = Ydata[group_indices[group]["test_index"]]
            kwargs=dict()
            groupstr = "%i".zfill(3) % group
            grouppath = savepath + "_" + groupstr
            if not (label_field_name == None):
                label = labeldata[group_indices[group]["test_index"][0]]
                grouppath = grouppath + "_%s" % label
            if not os.path.isdir(grouppath):
                os.mkdir(grouppath)
            kwargs['savepath'] = grouppath
            kwargs['plottype'] = plottype
            if flipaxes == 0:
                kwargs['xlabel'] = xfeatures[numplot]
                kwargs['ylabel'] = yfeature
                plotxy.single(xdata, ydata, **kwargs)
            else:
                kwargs['xlabel'] = yfeature
                kwargs['ylabel'] = xfeatures[numplot]
                plotxy.single(ydata, xdata, **kwargs)
    return
