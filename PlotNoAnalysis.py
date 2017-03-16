import matplotlib
import matplotlib.pyplot as plt
import data_parser
import numpy as np
import data_analysis.printout_tools as ptools
import plot_data.plot_xy as plotxy

def execute(model, data, savepath, 
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
    
    Xdata = data.get_x_data()
    ydata = np.asarray(data.get_y_data()).ravel()
    xfeatures = data.x_features
    yfeature = data.y_feature

    xshape = Xdata.shape
    if len(xshape) > 1:
        numplots = xshape[1]
    else:
        numplots = 1

    for numplot in range(0, numplots):
        xdata = Xdata[:,numplot]
        ydata = ydata
        kwargs=dict()
        kwargs['savepath'] = savepath
        if flipaxes == 0:
            kwargs['xlabel'] = xfeatures[numplot]
            kwargs['ylabel'] = yfeature
            plotxy.single(xdata, ydata, **kwargs)
        else:
            kwargs['xlabel'] = yfeature
            kwargs['ylabel'] = xfeatures[numplot]
            plotxy.single(ydata, xdata, **kwargs)
    return
