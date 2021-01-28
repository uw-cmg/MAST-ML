"""
This module contains a collection of functions which make plots (saved as png files) using matplotlib, generated from
some model fits and cross-validation evaluation within a MAST-ML run.

This module also contains a method to create python notebooks containing plotted data and the relevant source code from
this module, to enable the user to make their own modifications to the created plots in a straightforward way (useful for
tweaking plots for a presentation or publication).
"""

import math
import os
import pandas as pd
import numpy as np
from collections import Iterable
from math import log, floor, ceil

from mastml.metrics import Metrics

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure, figaspect
from matplotlib.font_manager import FontProperties

matplotlib.rc('font', size=18, family='sans-serif') # set all font to bigger
matplotlib.rc('figure', autolayout=True) # turn on autolayout

# adding dpi as a constant global so it can be changed later
DPI = 250

class Scatter():
    """
    Class to generate scatter plots, such as parity plots showing true vs. predicted data values

    Args:
        None

    Methods:

        plot_predicted_vs_true: method to plot a parity plot

            Args:

                y_true: (pd.Series), series of true y data

                y_pred: (pd.Series), series of predicted y data

                savepath: (str), string denoting the save path for the figure image

                file_name: (str), string denoting the character of the file name, e.g. train vs. test

                x_label: (str), string denoting the true and predicted property name

                metrics_list: (list), list of strings of metric names to evaluate and include on the figure

                groups: (pd.Series), series of group designations

                show_figure: (bool), whether or not to show the figure output (e.g. when using Jupyter notebook)

            Returns:
                None
    """
    @classmethod
    def plot_predicted_vs_true(cls, y_true, y_pred, savepath, file_name, x_label, metrics_list=None, groups=None, show_figure=False):
        # Make the dataframe/array 1D if it isn't
        y_true = check_dimensions(y_true)
        y_pred = check_dimensions(y_pred)

        # Set image aspect ratio:
        fig, ax = make_fig_ax()

        # gather max and min
        max1 = max(np.nanmax(y_true), np.nanmax(y_pred))
        min1 = min(np.nanmin(y_true), np.nanmin(y_pred))

        maxx = max(y_true)
        minn = min(y_true)
        maxx = round(float(maxx), rounder(maxx - minn))
        minn = round(float(minn), rounder(maxx - minn))
        _set_tick_labels(ax, maxx, minn)

        if groups is None:
            ax.scatter(y_true, y_pred, c='b', edgecolor='darkblue', zorder=2, s=100, alpha=0.7)
        else:
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'black']
            markers = ['o', 'v', '^', 's', 'p', 'h', 'D', '*', 'X', '<', '>', 'P']
            colorcount = markercount = 0
            for groupcount, group in enumerate(np.unique(groups)):
                mask = groups == group
                ax.scatter(y_true[mask], y_pred[mask], label=group, color=colors[colorcount],
                           marker=markers[markercount], s=100, alpha=0.7)
                ax.legend(loc='lower right', fontsize=12)
                colorcount += 1
                if colorcount % len(colors) == 0:
                    markercount += 1
                    colorcount = 0

        # draw dashed horizontal line
        ax.plot([min1, max1], [min1, max1], 'k--', lw=2, zorder=1)

        ax.set_xlabel('True '+ x_label, fontsize=14)
        ax.set_ylabel('Predicted ' + x_label, fontsize=14)

        if metrics_list is None:
            # Use some default metric set
            metrics_list = ['r2_score', 'mean_absolute_error', 'root_mean_squared_error', 'rmse_over_stdev']
        stats_dict = Metrics(metrics_list=metrics_list).evaluate(y_true=y_true, y_pred=y_pred)

        plot_stats(fig, stats_dict, x_align=0.65, y_align=0.90, fontsize=12)

        # Save data to excel file and image
        df = pd.DataFrame().from_dict(data={"y_true": y_true, "y_pred": y_pred})
        df.to_excel(os.path.join(savepath, file_name + '.xlsx'))
        fig.savefig(os.path.join(savepath, file_name + '.png'), dpi=DPI, bbox_inches='tight')
        if show_figure == True:
            plt.show()
        else:
            plt.close()
        return

class Histogram():
    """
    Class to generate histogram plots, such as histograms of residual values

    Args:
        None

    Methods:

        plot_histogram: method to plot a basic histogram of supplied data

            Args:

                df: (pd.DataFrame), dataframe or series of data to plot as a histogram

                savepath: (str), string denoting the save path for the figure image

                file_name: (str), string denoting the character of the file name, e.g. train vs. test

                x_label: (str), string denoting the  property name

                show_figure: (bool), whether or not to show the figure output (e.g. when using Jupyter notebook)

            Returns:
                None

        plot_residuals_histogram: method to plot a histogram of residual values

            Args:

                y_true: (pd.Series), series of true y data

                y_pred: (pd.Series), series of predicted y data

                savepath: (str), string denoting the save path for the figure image

                file_name: (str), string denoting the character of the file name, e.g. train vs. test

                show_figure: (bool), whether or not to show the figure output (e.g. when using Jupyter notebook)

            Returns:
                None

        _get_histogram_bins: Method to obtain the number of bins to use when plotting a histogram

            Args:

                df: (pandas Series or numpy array), array of y data used to construct histogram

            Returns:

                num_bins: (int), the number of bins to use when plotting a histogram
    """
    @classmethod
    def plot_histogram(cls, df, savepath, file_name, x_label, show_figure=False):
        # Make the dataframe 1D if it isn't
        df = check_dimensions(df)

        # make fig and ax, use x_align when placing text so things don't overlap
        x_align = 0.70
        fig, ax = make_fig_ax(aspect_ratio=0.5, x_align=x_align)

        #Get num_bins using smarter method
        num_bins = cls._get_histogram_bins(df=df)

        # do the actual plotting
        ax.hist(df, bins=num_bins, color='b', edgecolor='k')

        # normal text stuff
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel('Number of occurrences', fontsize=14)

        plot_stats(fig, dict(df.describe()), x_align=x_align, y_align=0.90, fontsize=12)

        # Save data to excel file and image
        df.to_excel(os.path.join(savepath, file_name + '.xlsx'))
        df.describe().to_excel(os.path.join(savepath, file_name + '_statistics.xlsx'))
        fig.savefig(os.path.join(savepath, file_name + '.png'), dpi=DPI, bbox_inches='tight')
        if show_figure == True:
            plt.show()
        else:
            plt.close()
        return

    @classmethod
    def plot_residuals_histogram(cls, y_true, y_pred, savepath, show_figure=False, file_name='residual_histogram'):
        y_true = check_dimensions(y_true)
        y_pred = check_dimensions(y_pred)
        residuals = y_pred-y_true
        cls.plot_histogram(df=residuals,
                            savepath=savepath,
                            file_name=file_name,
                            x_label='Residuals',
                            show_figure=show_figure)
        return

    @classmethod
    def _get_histogram_bins(cls, df):

        bin_dividers = np.linspace(df.shape[0], 0.05*df.shape[0], df.shape[0])
        bin_list = list()
        try:
            for divider in bin_dividers:
                if divider == 0:
                    continue
                bins = int((df.shape[0])/divider)
                if bins < df.shape[0]/2:
                    bin_list.append(bins)
        except:
            num_bins = 10
        if len(bin_list) > 0:
            num_bins = max(bin_list)
        else:
            num_bins = 10
        return num_bins


### Helpers:

def check_dimensions(y):
    """
    Method to check the dimensions of supplied data. Plotters need data to be 1D and often data is passed in as 2D

    Args:

        y: (numpy array or pd.DataFrame), array or dataframe of data used for plotting

    Returns:

        y: (pd.Series), series that is now 1D

    """
    if len(y.shape) > 1:
        if type(y) == pd.core.frame.DataFrame:
            y = pd.DataFrame.squeeze(y)
        elif type(y) == np.ndarray:
            y = pd.DataFrame(y.ravel()).squeeze()
            #y = y.ravel()
    else:
        if type(y) == np.ndarray:
            y = pd.DataFrame(y).squeeze()
    return y

def reset_index(y):
    return pd.DataFrame(np.array(y))

def trim_array(arr_list):
    """
    Method used to trim a set of arrays to make all arrays the same shape

    Args:

        arr_list: (list), list of numpy arrays, where arrays are different sizes

    Returns:

        arr_list: (), list of trimmed numpy arrays, where arrays are same size

    """

    # TODO: a better way to handle arrays with very different shapes? Otherwise average only uses # of points of smallest array
    # Need to make arrays all same shapes if they aren't
    sizes = [arr.shape[0] for arr in arr_list]
    size_min = min(sizes)
    arr_list_ = list()
    for i, arr in enumerate(arr_list):
        if arr.shape[0] > size_min:
            while arr.shape[0] > size_min:
                arr = np.delete(arr, -1)
        arr_list_.append(arr)
    arr_list = arr_list_
    return arr_list

def rounder(delta):
    """
    Method to obtain number of decimal places to report on plots

    Args:

        delta: (float), a float representing the change in two y values on a plot, used to obtain the plot axis spacing size

    Return:

        (int), an integer denoting the number of decimal places to use

    """
    if 0.001 <= delta < 0.01:
        return 3
    elif 0.01 <= delta < 0.1:
        return 2
    elif 0.1 <= delta < 1:
        return 1
    elif 1 <= delta < 100000:
        return 0
    else:
        return 0

def stat_to_string(name, value, nice_names):
    """
    Method that converts a metric object into a string for displaying on a plot

    Args:

        name: (str), long name of a stat metric or quantity

        value: (float), value of the metric or quantity

    Return:

        (str), a string of the metric name, adjusted to look nicer for inclusion on a plot

    """

    " Stringifies the name value pair for display within a plot "
    if name in nice_names:
        name = nice_names[name]
    else:
        name = name.replace('_', ' ')

    # has a name only
    if not value:
        return name
    # has a mean and std
    if isinstance(value, tuple):
        mean, std = value
        return f'{name}:' + '\n\t' + f'{mean:.3f}' + r'$\pm$' + f'{std:.3f}'
    # has a name and value only
    if isinstance(value, int) or (isinstance(value, float) and value%1 == 0):
        return f'{name}: {int(value)}'
    if isinstance(value, float):
        return f'{name}: {value:.3f}'
    return f'{name}: {value}' # probably a string

def plot_stats(fig, stats, x_align=0.65, y_align=0.90, font_dict=dict(), fontsize=14):
    """
    Method that prints stats onto the plot. Goes off screen if they are too long or too many in number.

    Args:

        fig: (matplotlib figure object), a matplotlib figure object

        stats: (dict), dict of statistics to be included with a plot

        x_align: (float), float denoting x position of where to align display of stats on a plot

        y_align: (float), float denoting y position of where to align display of stats on a plot

        font_dict: (dict), dict of matplotlib font options to alter display of stats on plot

        fontsize: (int), the fontsize of stats to display on plot

    Returns:

        None

    """

    stat_str = '\n'.join(stat_to_string(name, value, nice_names=nice_names())
                           for name,value in stats.items())

    fig.text(x_align, y_align, stat_str,
             verticalalignment='top', wrap=True, fontdict=font_dict, fontproperties=FontProperties(size=fontsize))

def make_fig_ax(aspect_ratio=0.5, x_align=0.65, left=0.10):
    """
    Method to make matplotlib figure and axes objects. Using Object Oriented interface from https://matplotlib.org/gallery/api/agg_oo_sgskip.html

    Args:

        aspect_ratio: (float), aspect ratio for figure and axes creation

        x_align: (float), x position to draw edge of figure. Needed so can display stats alongside plot

        left: (float), the leftmost position to draw edge of figure

    Returns:

        fig: (matplotlib fig object), a matplotlib figure object with the specified aspect ratio

        ax: (matplotlib ax object), a matplotlib axes object with the specified aspect ratio

    """
    # Set image aspect ratio:
    w, h = figaspect(aspect_ratio)
    fig = plt.figure(figsize=(w,h))
    #fig = Figure(figsize=(w, h))
    FigureCanvas(fig)

    # Set custom positioning, see this guide for more details:
    # https://python4astronomers.github.io/plotting/advanced.html
    #left   = 0.10
    bottom = 0.15
    right  = 0.01
    top    = 0.05
    width = x_align - left - right
    height = 1 - bottom - top
    ax = fig.add_axes((left, bottom, width, height), frameon=True)
    fig.set_tight_layout(False)
    
    return fig, ax

def make_fig_ax_square(aspect='equal', aspect_ratio=1):
    """
    Method to make square shaped matplotlib figure and axes objects. Using Object Oriented interface from

    https://matplotlib.org/gallery/api/agg_oo_sgskip.html

    Args:

        aspect: (str), 'equal' denotes x and y aspect will be equal (i.e. square)

        aspect_ratio: (float), aspect ratio for figure and axes creation

    Returns:

        fig: (matplotlib fig object), a matplotlib figure object with the specified aspect ratio

        ax: (matplotlib ax object), a matplotlib axes object with the specified aspect ratio

    """
    # Set image aspect ratio:
    w, h = figaspect(aspect_ratio)
    fig = Figure(figsize=(w,h))
    FigureCanvas(fig)
    ax = fig.add_subplot(111, aspect=aspect)

    return fig, ax

def make_axis_same(ax, max1, min1):
    """
    Method to make the x and y ticks for each axis the same. Useful for parity plots

    Args:

        ax: (matplotlib axis object), a matplotlib axes object

        max1: (float), the maximum value of a particular axis

        min1: (float), the minimum value of a particular axis

    Returns:

        None

    """
    if max1 - min1 > 5:
        step = (int(max1) - int(min1)) // 3
        ticks = range(int(min1), int(max1)+step, step)
    else:
        ticks = np.linspace(min1, max1, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

def nice_mean(ls):
    """
    Method to return mean of a list or equivalent array with NaN values

    Args:

        ls: (list), list of values

    Returns:

        (numpy array), array containing mean of list of values or NaN if list has no values

    """

    if len(ls) > 0:
        return np.mean(ls)
    return np.nan

def nice_std(ls):
    """
    Method to return standard deviation of a list or equivalent array with NaN values

    Args:

        ls: (list), list of values

    Returns:

        (numpy array), array containing standard deviation of list of values or NaN if list has no values

    """
    if len(ls) > 0:
        return np.std(ls)
    return np.nan

def round_down(num, divisor):
    """
    Method to return a rounded down number

    Args:

        num: (float), a number to round down

        divisor: (int), divisor to denote how to round down

    Returns:

        (float), the rounded-down number

    """

    return num - (num%divisor)

def round_up(num, divisor):
    """
    Method to return a rounded up number

    Args:

        num: (float), a number to round up

        divisor: (int), divisor to denote how to round up

    Returns:

        (float), the rounded-up number

    """
    return float(math.ceil(num / divisor)) * divisor

def get_divisor(high, low):
    """
    Method to obtain a sensible divisor based on range of two values

    Args:

        high: (float), a max data value

        low: (float), a min data value

    Returns:

        divisor: (float), a number used to make sensible axis ticks

    """

    delta = high-low
    divisor = 10
    if delta > 1000:
        divisor = 100
    if delta < 1000:
        if delta > 100:
            divisor = 10
        if delta < 100:
            if delta > 10:
                divisor = 1
            if delta < 10:
                if delta > 1:
                    divisor = 0.1
                if delta < 1:
                    if delta > 0.01:
                        divisor = 0.001
                else:
                    divisor = 0.001
    return divisor

def recursive_max(arr):
    """
    Method to recursively find the max value of an array of iterables.

    Credit: https://www.linkedin.com/pulse/ask-recursion-during-coding-interviews-identify-good-talent-veteanu/

    Args:

        arr: (numpy array), an array of values or iterables

    Returns:

        (float), max value in arr

    """
    return max(
        recursive_max(e) if isinstance(e, Iterable) else e
        for e in arr
    )

def recursive_min(arr):
    """
    Method to recursively find the min value of an array of iterables.

    Credit: https://www.linkedin.com/pulse/ask-recursion-during-coding-interviews-identify-good-talent-veteanu/

    Args:

        arr: (numpy array), an array of values or iterables

    Returns:

        (float), min value in arr

    """

    return min(
        recursive_min(e) if isinstance(e, Iterable) else e
        for e in arr
    )

def recursive_max_and_min(arr):
    """
    Method to recursively return max and min of values or iterables in array

    Args:

        arr: (numpy array), an array of values or iterables

    Returns:

        (tuple), tuple containing max and min of arr

    """
    return recursive_max(arr), recursive_min(arr)

def _set_tick_labels(ax, maxx, minn):
    """
    Method that sets the x and y ticks to be in the same range

    Args:

        ax: (matplotlib axes object), a matplotlib axes object

        maxx: (float), a maximum value

        minn: (float), a minimum value

    Returns:

        None

    """
    _set_tick_labels_different(ax, maxx, minn, maxx, minn) # I love it when this happens

def _set_tick_labels_different(ax, max_tick_x, min_tick_x, max_tick_y, min_tick_y):
    """
    Method that sets the x and y ticks, when the axes have different ranges

    Args:

        ax: (matplotlib axes object), a matplotlib axes object

        max_tick_x: (float), the maximum tick value for the x axis

        min_tick_x: (float), the minimum tick value for the x axis

        max_tick_y: (float), the maximum tick value for the y axis

        min_tick_y: (float), the minimum tick value for the y axis

    Returns:

        None

    """

    tickvals_x = nice_range(min_tick_x, max_tick_x)
    tickvals_y = nice_range(min_tick_y, max_tick_y)

    if tickvals_x[-1]-tickvals_x[len(tickvals_x)-2] < tickvals_x[len(tickvals_x)-3]-tickvals_x[len(tickvals_x)-4]:
        tickvals_x = tickvals_x[:-1]
    if tickvals_y[-1]-tickvals_y[len(tickvals_y)-2] < tickvals_y[len(tickvals_y)-3]-tickvals_y[len(tickvals_y)-4]:
        tickvals_y = tickvals_y[:-1]
    #tickvals_x = _clean_tick_labels(tickvals=tickvals_x, delta=max_tick_x-min_tick_x)
    #tickvals_y = _clean_tick_labels(tickvals=tickvals_y, delta=max_tick_y - min_tick_y)

    ax.set_xticks(ticks=tickvals_x)
    ax.set_yticks(ticks=tickvals_y)

    ticklabels_x = [str(tick) for tick in tickvals_x]
    ticklabels_y = [str(tick) for tick in tickvals_y]

    rotation = 0
    # Look at length of x tick labels to see if may be possibly crowded. If so, rotate labels
    tick_length = len(str(tickvals_x[1]))
    if tick_length >= 4:
        rotation = 45
    ax.set_xticklabels(labels=ticklabels_x, fontsize=14, rotation=rotation)
    ax.set_yticklabels(labels=ticklabels_y, fontsize=14)

def _clean_tick_labels(tickvals, delta):
    """
    Method to attempt to clean up axis tick values so they don't overlap from being too dense

    Args:

        tickvals: (list), a list containing the initial axis tick values

        delta: (float), number representing the numerical difference of two ticks

    Returns:

        tickvals_clean: (list), a list containing the updated axis tick values

    """
    tickvals_clean = list()
    if delta >= 100:
        for i, val in enumerate(tickvals):
            if i <= len(tickvals)-1:
                if tickvals[i]-tickvals[i-1] >= 100:
                    tickvals_clean.append(val)
    else:
        tickvals_clean = tickvals
    return tickvals_clean

## Math utilities to aid plot_helper to make ranges

def nice_range(lower, upper):
    """
    Method to create a range of values, including the specified start and end points, with nicely spaced intervals

    Args:

        lower: (float or int), lower bound of range to create

        upper: (float or int), upper bound of range to create

    Returns:

        (list), list of numerical values in established range

    """

    flipped = 1 # set to -1 for inverted

    # Case for validation where nan is passed in
    if np.isnan(lower):
        lower = 0
    if np.isnan(upper):
        upper = 0.1

    if upper < lower:
        upper, lower = lower, upper
        flipped = -1
    return [_int_if_int(x) for x in _nice_range_helper(lower, upper)][::flipped]

def _nice_range_helper(lower, upper):
    """
    Method to help make a better range of axis ticks

    Args:

        lower: (float), lower value of axis ticks

        upper: (float), upper value of axis ticks

    Returns:

        upper: (float), modified upper tick value fixed based on set of axis ticks

    """
    steps = 8
    diff = abs(lower - upper)

    # special case where lower and upper are the same
    if diff == 0:
        return [lower,]

    # the exact step needed
    step = diff / steps

    # a rough estimate of best step
    step = _nearest_pow_ten(step) # whole decimal increments

    # tune in one the best step size
    factors = [0.1, 0.2, 0.5, 1, 2, 5, 10]

    # use this to minimize how far we are from ideal step size
    def best_one(steps_factor):
        steps_count, factor = steps_factor
        return abs(steps_count - steps)
    n_steps, best_factor = min([(diff / (step * f), f) for f in factors], key = best_one)

    #print('should see n steps', ceil(n_steps + 2))
    # multiply in the optimal factor for getting as close to ten steps as we can
    step = step * best_factor

    # make the bounds look nice
    lower = _three_sigfigs(lower)
    upper = _three_sigfigs(upper)

    start = _round_up(lower, step)

    # prepare for iteration
    x = start # pointless init
    i = 0

    # itereate until we reach upper
    while x < upper - step:
        x = start + i * step
        yield _three_sigfigs(x) # using sigfigs because of floating point error
        i += 1

    # finish off with ending bound
    yield upper

def _three_sigfigs(x):
    """
    Method invoking special case of _n_sigfigs to return 3 sig figs

    Args:

        x: (float), an axis tick number

    Returns:

        (float), number of sig figs (always 3)

    """
    return _n_sigfigs(x, 3)

def _n_sigfigs(x, n):
    """
    Method to return number of sig figs to use for axis ticks

    Args:

        x: (float), an axis tick number

    Returns:

        (float), number of sig figs

    """
    sign = 1
    if x == 0:
        return 0
    if x < 0: # case for negatives
        x = -x
        sign = -1
    if x < 1:
        base = n - round(log(x, 10))
    else:
        base = (n-1) - round(log(x, 10))
    return sign * round(x, base)

def _nearest_pow_ten(x):
    """
    Method to return the nearest power of ten for an axis tick value

    Args:

        x: (float), an axis tick number

    Returns:

        (float), nearest power of ten of x

    """
    sign = 1
    if x == 0:
        return 0
    if x < 0: # case for negatives
        x = -x
        sign = -1
    return sign*10**ceil(log(x, 10))

def _int_if_int(x):
    """
    Method to return integer mapped value of x

    Args:

        x: (float or int), a number

    Returns:

        x: (float), value of x mapped as integer

    """
    if int(x) == x:
        return int(x)
    return x

def _round_up(x, inc):
    """
    Method to round up the value of x

    Args:

        x: (float or int), a number

        inc: (float), an increment for axis ticks

    Returns:

        (float), value of x rounded up

    """
    sign = 1
    if x < 0: # case for negative
        x = -x
        sign = -1

    return sign * inc * ceil(x / inc)

def nice_names():
    nice_names = {
    # classification:
    'accuracy': 'Accuracy',
    'f1_binary': '$F_1$',
    'f1_macro': 'f1_macro',
    'f1_micro': 'f1_micro',
    'f1_samples': 'f1_samples',
    'f1_weighted': 'f1_weighted',
    'log_loss': 'log_loss',
    'precision_binary': 'Precision',
    'precision_macro': 'prec_macro',
    'precision_micro': 'prec_micro',
    'precision_samples': 'prec_samples',
    'precision_weighted': 'prec_weighted',
    'recall_binary': 'Recall',
    'recall_macro': 'rcl_macro',
    'recall_micro': 'rcl_micro',
    'recall_samples': 'rcl_samples',
    'recall_weighted': 'rcl_weighted',
    'roc_auc': 'ROC_AUC',
    # regression:
    'explained_variance': 'expl_var',
    'mean_absolute_error': 'MAE',
    'mean_squared_error': 'MSE',
    'mean_squared_log_error': 'MSLE',
    'median_absolute_error': 'MedAE',
    'root_mean_squared_error': 'RMSE',
    'rmse_over_stdev': r'RMSE/$\sigma_y$',
    'r2_score': '$R^2$',
    'r2_score_noint': '$R^2_{noint}$',
    'r2_score_adjusted': '$R^2_{adjusted}$',
    'r2_score_fitted': '$R^2_{fitted}$'
    }
    return nice_names
