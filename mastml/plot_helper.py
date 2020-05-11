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
import itertools
import warnings
import logging
from collections import Iterable
from os.path import join
from collections import OrderedDict
from math import log, floor, ceil

# Ignore the harmless warning about the gelsd driver on mac.
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")
# Ignore matplotlib deprecation warning (set as all warnings for now)
warnings.filterwarnings(action="ignore")

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure, figaspect
from matplotlib.animation import FuncAnimation
from matplotlib.font_manager import FontProperties
from scipy.stats import gaussian_kde, norm
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

# Needed imports for ipynb_maker
#from mastml.utils import nice_range
#from mastml.metrics import nice_names

import inspect
import textwrap
from pandas import DataFrame, Series

import nbformat

from functools import wraps

import forestci as fci

matplotlib.rc('font', size=18, family='sans-serif') # set all font to bigger
matplotlib.rc('figure', autolayout=True) # turn on autolayout

# adding dpi as a constant global so it can be changed later
DPI = 250

#logger = logging.getLogger() # only used inside ipynb_maker I guess

# HEADERENDER don't delete this line, it's used by ipynb maker

logger = logging.getLogger('mastml') # the real logger

def ipynb_maker(plot_func):
    """
    This method creates Jupyter Notebooks so user can modify and regenerate the plots produced by MAST-ML.

    Args:

        plot_func: (plot_helper method), a plotting method contained in plot_helper.py which contains the

        @ipynb_maker decorator

    Returns:

        (plot_helper method), the same plot_func as used as input, but after having written the Jupyter notebook with source code to create plot

    """

    from mastml import plot_helper # Strange self-import but it works, as had cyclic import issues with ipynb_maker as its own module

    @wraps(plot_func)
    def wrapper(*args, **kwargs):
        # convert everything to kwargs for easier display
        # from geniuses at https://stackoverflow.com/a/831164
        #kwargs.update(dict(zip(plot_func.func_code.co_varnames, args)))
        sig = inspect.signature(plot_func)
        binding = sig.bind(*args, **kwargs)
        all_args = binding.arguments

        # if this is an outdir style function, fill in savepath and delete outdir
        if 'savepath' in all_args:
            ipynb_savepath = all_args['savepath']
            knows_savepath = True
            basename = os.path.basename(ipynb_savepath) # fix absolute path problem
        elif 'outdir' in all_args:
            knows_savepath = False
            basename = plot_func.__name__
            ipynb_savepath = os.path.join(all_args['outdir'], basename)
        else:
            raise Exception('you must have an "outdir" or "savepath" argument to use ipynb_maker')

        readme = textwrap.dedent(f"""\
            This notebook was automatically generated from your MAST-ML run so you can recreate the
            plots. Some things are a bit different from the usual way of creating plots - we are
            using the [object oriented
            interface](https://matplotlib.org/tutorials/introductory/lifecycle.html) instead of
            pyplot to create the `fig` and `ax` instances.
        """)

        # get source of the top of plot_helper.py
        header = ""
        with open(plot_helper.__file__) as f:
            for line in f.readlines():
                if 'HEADERENDER' in line:
                    break
                header += line

        core_funcs = [plot_helper.stat_to_string, plot_helper.plot_stats, plot_helper.make_fig_ax,
                      plot_helper.get_histogram_bins, plot_helper.nice_names, plot_helper.nice_range,
                      plot_helper.nice_mean, plot_helper.nice_std, plot_helper.rounder, plot_helper._set_tick_labels,
                      plot_helper._set_tick_labels_different, plot_helper._nice_range_helper, plot_helper._nearest_pow_ten,
                      plot_helper._three_sigfigs, plot_helper._n_sigfigs, plot_helper._int_if_int, plot_helper._round_up,
                      plot_helper.prediction_intervals]
        func_strings = '\n\n'.join(inspect.getsource(func) for func in core_funcs)

        plot_func_string = inspect.getsource(plot_func)
        # remove first line that has this decorator on it (!!!)
        plot_func_string = '\n'.join(plot_func_string.split('\n')[1:])

        # put the arguments and their values in the code
        arg_assignments = []
        arg_names = []
        for key, var in all_args.items():
            if isinstance(var, DataFrame):
                # this is amazing
                arg_assignments.append(f"{key} = pd.read_csv(StringIO('''\n{var.to_csv(index=False)}'''))")
            elif isinstance(var, Series):
                arg_assignments.append(f"{key} = pd.Series(pd.read_csv(StringIO('''\n{var.to_csv(index=False)}''')).iloc[:,0])")
            else:
                arg_assignments.append(f'{key} = {repr(var)}')
            arg_names.append(key)
        args_block = ("from numpy import array\n" +
                      "from collections import OrderedDict\n" +
                      "from io import StringIO\n" +
                    "from sklearn.gaussian_process import GaussianProcessRegressor  # Need for error plots\n" +
                    "from sklearn.gaussian_process.kernels import *  # Need for error plots\n" +
                    "from sklearn.ensemble import RandomForestRegressor  # Need for error plots\n" +
                      '\n'.join(arg_assignments))
        arg_names = ', '.join(arg_names)

        if knows_savepath:
            if '.png' not in basename:
                basename += '.png'
            main = textwrap.dedent(f"""\
                import pandas as pd
                from IPython.display import Image, display

                {plot_func.__name__}({arg_names})
                display(Image(filename='{basename}'))
            """)
        else:
            main = textwrap.dedent(f"""\
                import pandas as pd
                from IPython.display import Image, display

                plot_paths = plot_predicted_vs_true(train_quad, test_quad, outdir, label)
                for plot_path in plot_paths:
                    display(Image(filename=plot_path))
            """)

        nb = nbformat.v4.new_notebook()
        readme_cell = nbformat.v4.new_markdown_cell(readme)
        text_cells = [header, func_strings, plot_func_string, args_block, main]
        cells = [readme_cell] + [nbformat.v4.new_code_cell(cell_text) for cell_text in text_cells]
        nb['cells'] = cells
        nbformat.write(nb, ipynb_savepath + '.ipynb')

        return plot_func(*args, **kwargs)
    return wrapper

def make_train_test_plots(run, path, is_classification, label, model, train_X, test_X, groups=None):
    """
    General plotting method used to execute sequence of specific plots of train-test data analysis

    Args:

        run: (dict), a particular split_result from masml_driver

        path: (str), path to save the generated plots and analysis of split_result designated in 'run'

        is_classification: (bool), whether or not the analysis is a classification task

        label: (str), name of the y data variable being fit

        model: (scikit-learn model object), a scikit-learn model/estimator

        train_X: (numpy array), array of X features used in training

        test_X: (numpy array), array of X features used in testing

        groups: (numpy array), array of group names

    Returns:

        None

    """
    y_train_true, y_train_pred, y_test_true = \
        run['y_train_true'], run['y_train_pred'], run['y_test_true']
    y_test_pred, train_metrics, test_metrics = \
        run['y_test_pred'], run['train_metrics'], run['test_metrics']
    train_groups, test_groups = run['train_groups'], run['test_groups']

    if is_classification:
        # Need these class prediction probabilities for ROC curve analysis
        y_train_pred_proba = run['y_train_pred_proba']
        y_test_pred_proba = run['y_test_pred_proba']

        title = 'train_confusion_matrix'
        plot_confusion_matrix(y_train_true, y_train_pred,
                              join(path, title+'.png'), train_metrics,
                              title=title)
        title = 'test_confusion_matrix'
        plot_confusion_matrix(y_test_true, y_test_pred,
                              join(path, title+'.png'), test_metrics,
                              title=title)
        title = 'train_roc_curve'
        plot_roc_curve(y_train_true, y_train_pred_proba, join(path, title+'png'))
        title = 'test_roc_curve'
        plot_roc_curve(y_test_true, y_test_pred_proba, join(path, title+'png'))
        title = 'train_precision_recall_curve'
        plot_precision_recall_curve(y_train_true, y_train_pred_proba, join(path, title+'png'))
        title = 'test_precision_recall_curve'
        plot_precision_recall_curve(y_test_true, y_test_pred_proba, join(path, title + 'png'))
    else: # is_regression
        plot_predicted_vs_true((y_train_true, y_train_pred, train_metrics, train_groups),
                          (y_test_true,  y_test_pred,  test_metrics, test_groups), 
                          path, label=label)

        title = 'train_residuals_histogram'
        plot_residuals_histogram(y_train_true, y_train_pred,
                                 join(path, title+'.png'), train_metrics,
                                 title=title, label=label)
        title = 'test_residuals_histogram'
        plot_residuals_histogram(y_test_true,  y_test_pred,
                                 join(path, title+'.png'), test_metrics,
                                 title=title, label=label)

def make_error_plots(run, path, is_classification, label, model, train_X, test_X, rf_error_method, rf_error_percentile,
                     is_validation, validation_column_name, validation_X, groups=None):

    y_train_true, y_train_pred, y_test_true = \
        run['y_train_true'], run['y_train_pred'], run['y_test_true']
    y_test_pred, train_metrics, test_metrics = \
        run['y_test_pred'], run['train_metrics'], run['test_metrics']
    train_groups, test_groups = run['train_groups'], run['test_groups']
    if is_validation:
        y_validation_pred, y_validation_true, prediction_metrics = \
            run['y_validation_pred'+'_'+str(validation_column_name)], \
            run['y_validation_true'+'_'+str(validation_column_name)], \
            run['prediction_metrics']

    if is_classification:
        logger.debug('There is no error distribution plotting for classification problems, just passing through...')
    else: # is_regression

        #title = 'train_normalized_error'
        #plot_normalized_error(y_train_true, y_train_pred, join(path, title+'.png'), model, error_method, percentile,
        # X=train_X, Xtrain=train_X, Xtest=test_X)

        title = 'test_normalized_error'
        plot_normalized_error(y_test_true, y_test_pred, join(path, title+'.png'), model, rf_error_method,
                              rf_error_percentile, X=test_X, Xtrain=train_X, Xtest=test_X)

        #title = 'train_cumulative_normalized_error'
        #plot_cumulative_normalized_error(y_train_true, y_train_pred, join(path, title+'.png'), model, error_method,
        # percentile, X=train_X, Xtrain=train_X, Xtest=test_X)

        title = 'test_cumulative_normalized_error'
        plot_cumulative_normalized_error(y_test_true, y_test_pred, join(path, title+'.png'), model, rf_error_method,
                                         rf_error_percentile, X=test_X, Xtrain=train_X, Xtest=test_X)

        # HERE, add your RMS residual vs. error plot function

        if is_validation:
            title = 'validation_cumulative_normalized_error'
            plot_cumulative_normalized_error(y_validation_true, y_validation_pred, join(path, title+'.png'), model, rf_error_method,
                                             rf_error_percentile, X=validation_X, Xtrain=train_X, Xtest=test_X)
            title = 'validation_normalized_error'
            plot_normalized_error(y_validation_true, y_validation_pred, join(path, title + '.png'), model, rf_error_method,
                                  rf_error_percentile, X=validation_X, Xtrain=train_X, Xtest=test_X)

@ipynb_maker
def plot_confusion_matrix(y_true, y_pred, savepath, stats, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Method used to generate a confusion matrix for a classification run. Additional information can be found
    at: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    Args:

        y_true: (numpy array), array containing the true y data values

        y_pred: (numpy array), array containing the predicted y data values

        savepath: (str), path to save the plotted confusion matrix

        stats: (dict), dict of training or testing statistics for a particular run

        normalize: (bool), whether or not to normalize data output as truncated float vs. double

        title: (str), title of the confusion matrix plot

        cmap: (matplotlib colormap), the color map to use for confusion matrix plotting

    Returns:

        None

    """

    # calculate confusion matrix and lables in correct order
    cm = confusion_matrix(y_true, y_pred)
    #classes = sorted(list(set(y_true).intersection(set(y_pred))))
    classes = sorted(list(set(y_true).union(set(y_pred))))

    x_align = 0.64
    fig, ax = make_fig_ax(x_align=x_align)

    #ax.set_title(title)

    # create the colorbar, not really needed but everyones got 'em
    mappable = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #fig.colorbar(mappable)

    # set x and y ticks to labels
    tick_marks = range(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation='horizontal', fontsize=18)

    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, rotation='horizontal', fontsize=18)

    # draw number in the boxes
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    # plots the stats
    plot_stats(fig, stats, x_align=0.60, y_align=0.90)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')

@ipynb_maker
def plot_roc_curve(y_true, y_pred, savepath):
    """
    Method to calculate and plot the receiver-operator characteristic curve for classification model results

    Args:

        y_true: (numpy array), array of true y data values

        y_pred: (numpy array), array of predicted y data values

        savepath: (str), path to save the plotted ROC curve

    Returns:

        None

    """

    #TODO: have work when probability=False in model params. Suggest user set probability=True!!
    #classes = sorted(list(set(y_true).union(set(y_pred))))
    #n_classes = y_pred.shape[1]

    classes = list(np.unique(y_true))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    x_align = 0.95
    fig, ax = make_fig_ax(aspect_ratio=0.66, x_align=x_align, left=0.15)
    colors = ['blue', 'red']
    #for i in range(len(classes)):
    #    ax.plot(fpr[i], tpr[i], color=colors[i], lw=2, label='ROC curve class '+str(i)+' (area = %0.2f)' % roc_auc[i])
    ax.plot(fpr[1], tpr[1], color=colors[0], lw=2, label='ROC curve' + ' (area = %0.2f)' % roc_auc[1])
    ax.plot([0, 1], [0, 1], color='black', label='Random guess', lw=2, linestyle='--')
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_yticks(np.linspace(0, 1, 5))
    _set_tick_labels(ax, maxx=1, minn=0)
    ax.set_xlabel('False Positive Rate', fontsize='16')
    ax.set_ylabel('True Positive Rate', fontsize='16')
    ax.legend(loc="lower right", fontsize=12)
    #plot_stats(fig, stats, x_align=0.60, y_align=0.90)
    fig.savefig(savepath, dpi=DPI, bbox_to_inches='tight')

@ipynb_maker
def plot_precision_recall_curve(y_true, y_pred, savepath):
    """
    Method to calculate and plot the precision-recall curve for classification model results

    Args:

        y_true: (numpy array), array of true y data values

        y_pred: (numpy array), array of predicted y data values

        savepath: (str), path to save the plotted precision-recall curve

    Returns:

        None

    """

    # Note this only works with probability predictions of the classifier labels.
    classes = list(np.unique(y_true))

    precision = dict()
    recall = dict()
    #roc_auc = dict()

    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(y_true, y_pred[:, i])

    x_align = 0.95
    fig, ax = make_fig_ax(aspect_ratio=0.66, x_align=x_align, left=0.15)
    colors = ['blue', 'red']
    #for i in range(len(classes)):
    #    ax.plot(fpr[i], tpr[i], color=colors[i], lw=2, label='ROC curve class '+str(i)+' (area = %0.2f)' % roc_auc[i])
    ax.step(recall[1], precision[1], color=colors[0], lw=2, label='Precision-recall curve')
    #ax.fill_between(recall[1], precision[1], alpha=0.4, color=colors[0])
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_yticks(np.linspace(0, 1, 5))
    _set_tick_labels(ax, maxx=1, minn=0)
    ax.set_xlabel('Recall', fontsize='16')
    ax.set_ylabel('Precision', fontsize='16')
    ax.legend(loc="upper right", fontsize=12)
    #plot_stats(fig, stats, x_align=0.60, y_align=0.90)
    fig.savefig(savepath, dpi=DPI, bbox_to_inches='tight')
    return

@ipynb_maker
def plot_residuals_histogram(y_true, y_pred, savepath,
                             stats, title='residuals histogram', label='residuals'):
    """
    Method to calculate and plot the histogram of residuals from regression model

    Args:

        y_true: (numpy array), array of true y data values

        y_pred: (numpy array), array of predicted y data values

        savepath: (str), path to save the plotted precision-recall curve

        stats: (dict), dict of training or testing statistics for a particular run

        title: (str), title of residuals histogram

        label: (str), label used for axis labeling

    Returns:

        None

    """

    # make fig and ax, use x_align when placing text so things don't overlap
    x_align = 0.64
    fig, ax = make_fig_ax(x_align=x_align)

    #ax.set_title(title)
    # do the actual plotting
    residuals = y_true - y_pred

    #Output residuals data and stats to spreadsheet
    path = os.path.dirname(savepath)
    pd.DataFrame(residuals).describe().to_csv(os.path.join(path,'residual_statistics.csv'))
    pd.DataFrame(residuals).to_csv(path+'/'+'residuals.csv')

    #Get num_bins using smarter method
    num_bins = get_histogram_bins(y_df=residuals)
    ax.hist(residuals, bins=num_bins, color='b', edgecolor='k')

    # normal text stuff
    ax.set_xlabel('Value of '+label, fontsize=16)
    ax.set_ylabel('Number of occurences', fontsize=16)

    # make y axis ints, because it is discrete
    #ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plot_stats(fig, stats, x_align=x_align, y_align=0.90)
    plot_stats(fig, pd.DataFrame(residuals).describe().to_dict()[0], x_align=x_align, y_align=0.60)

    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')

@ipynb_maker
def plot_target_histogram(y_df, savepath, title='target histogram', label='target values'):
    """
    Method to plot the histogram of true y values

    Args:

        y_df: (pandas dataframe), dataframe of true y data values

        savepath: (str), path to save the plotted precision-recall curve

        title: (str), title of residuals histogram

        label: (str), label used for axis labeling

    Returns:

        None

    """

    # make fig and ax, use x_align when placing text so things don't overlap
    x_align = 0.70
    fig, ax = make_fig_ax(aspect_ratio=0.5, x_align=x_align)

    #ax.set_title(title)

    #Get num_bins using smarter method
    num_bins = get_histogram_bins(y_df=y_df)

    # do the actual plotting
    try:
        ax.hist(y_df, bins=num_bins, color='b', edgecolor='k')#, histtype='stepfilled')
    except:
        print('Could not plot target histgram')
        return
    # normal text stuff
    ax.set_xlabel('Value of '+label, fontsize=16)
    ax.set_ylabel('Number of occurences', fontsize=16)

    # make y axis ints, because it is discrete
    #ax.yaxis.set_major_locator(MaxNLocator(integer=True))


    plot_stats(fig, dict(y_df.describe()), x_align=x_align, y_align=0.90, fontsize=14)
    # Save input data stats to csv
    savepath_parse = savepath.split('target_histogram.png')[0]
    y_df.describe().to_csv(savepath_parse+'/''input_data_statistics.csv')

    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')

@ipynb_maker
def plot_predicted_vs_true(train_quad, test_quad, outdir, label):
    """
    Method to create a parity plot (predicted vs. true values)

    Args:

        train_quad: (tuple), tuple containing 4 numpy arrays: true training y data, predicted training y data,

        training metric data, and groups used in training

        test_quad: (tuple), tuple containing 4 numpy arrays: true test y data, predicted test y data,

        testing metric data, and groups used in testing

        outdir: (str), path to save plots to

        label: (str), label used for axis labeling

    Returns:

        None

    """
    filenames = list()
    y_train_true, y_train_pred, train_metrics, train_groups = train_quad
    y_test_true, y_test_pred, test_metrics, test_groups = test_quad

    # make diagonal line from absolute min to absolute max of any data point
    # using round because Ryan did - but won't that ruin small numbers??? TODO this
    #max1 = max(y_train_true.max(), y_train_pred.max(),
    #           y_test_true.max(), y_test_pred.max())
    max1 = max(y_train_true.max(), y_test_true.max())
    #min1 = min(y_train_true.min(), y_train_pred.min(),
    #           y_test_true.min(), y_test_pred.min())
    min1 = min(y_train_true.min(), y_test_true.min())
    max1 = round(float(max1), rounder(max1-min1))
    min1 = round(float(min1), rounder(max1-min1))
    for y_true, y_pred, stats, groups, title_addon in \
            (train_quad+('train',), test_quad+('test',)):
        # make fig and ax, use x_align when placing text so things don't overlap
        x_align=0.64
        fig, ax = make_fig_ax(x_align=x_align)

        # set tick labels
        # notice that we use the same max and min for all three. Don't
        # calculate those inside the loop, because all the should be on the same scale and axis
        _set_tick_labels(ax, max1, min1)

        # plot diagonal line
        ax.plot([min1, max1], [min1, max1], 'k--', lw=2, zorder=1)

        # do the actual plotting
        if groups is None:
            ax.scatter(y_true, y_pred, color='blue', edgecolors='black', s=100, zorder=2, alpha=0.7)
        else:
            handles = dict()
            unique_groups = np.unique(np.concatenate((train_groups, test_groups), axis=0))
            unique_groups_train = np.unique(train_groups)
            unique_groups_test = np.unique(test_groups)
            #logger.debug(' '*12 + 'unique groups: ' +str(list(unique_groups)))
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'black']
            markers = ['o', 'v', '^', 's', 'p', 'h', 'D', '*', 'X', '<', '>', 'P']
            colorcount = markercount = 0
            for groupcount, group in enumerate(unique_groups):
                mask = groups == group
                #logger.debug(' '*12 + f'{group} group_percent = {np.count_nonzero(mask) / len(groups)}')
                handles[group] = ax.scatter(y_true[mask], y_pred[mask], label=group, color=colors[colorcount],
                                            marker=markers[markercount], s=100, alpha=0.7)
                colorcount += 1
                if colorcount % len(colors) == 0:
                    markercount += 1
                    colorcount = 0
            if title_addon == 'train':
                to_delete = [k for k in handles.keys() if k not in unique_groups_train]
                for k in to_delete:
                    del handles[k]
            elif title_addon == 'test':
                to_delete = [k for k in handles.keys() if k not in unique_groups_test]
                for k in to_delete:
                    del handles[k]
            ax.legend(handles.values(), handles.keys(), loc='lower right', fontsize=12)

        # set axis labels
        ax.set_xlabel('True '+label, fontsize=16)
        ax.set_ylabel('Predicted '+label, fontsize=16)

        plot_stats(fig, stats, x_align=x_align, y_align=0.90)

        filename = 'predicted_vs_true_'+ title_addon + '.png'
        filenames.append(filename)
        fig.savefig(join(outdir, filename), dpi=DPI, bbox_inches='tight')
        df = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true})
        df.to_csv(join(outdir, 'predicted_vs_true_' + title_addon + '.csv'))

    return filenames

def plot_scatter(x, y, savepath, groups=None, xlabel='x', label='target data'):
    """
    Method to create a general scatter plot

    Args:

        x: (numpy array), array of x data

        y: (numpy array), array of y data

        savepath: (str), path to save plots to

        groups: (list), list of group labels

        xlabel: (str), label used for x-axis labeling

        label: (str), label used for y-axis labeling

    Returns:

        None

    """

    # Set image aspect ratio:
    fig, ax = make_fig_ax()

    # set tick labels
    max_tick_x = max(x)
    min_tick_x = min(x)
    max_tick_y = max(y)
    min_tick_y = min(y)
    max_tick_x = round(float(max_tick_x), rounder(max_tick_x-min_tick_x))
    min_tick_x = round(float(min_tick_x), rounder(max_tick_x-min_tick_x))
    max_tick_y = round(float(max_tick_y), rounder(max_tick_y-min_tick_y))
    min_tick_y = round(float(min_tick_y), rounder(max_tick_y-min_tick_y))
    #divisor_y = get_divisor(max(y), min(y))
    #max_tick_y = round_up(max(y), divisor_y)
    #min_tick_y = round_down(min(y), divisor_y)

    _set_tick_labels_different(ax, max_tick_x, min_tick_x, max_tick_y, min_tick_y)

    if groups is None:
        ax.scatter(x, y, c='b', edgecolor='darkblue', zorder=2, s=100, alpha=0.7)
    else:
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'black']
        markers = ['o', 'v', '^', 's', 'p', 'h', 'D', '*', 'X', '<', '>', 'P']
        colorcount = markercount = 0
        for groupcount, group in enumerate(np.unique(groups)):
            mask = groups == group
            ax.scatter(x[mask], y[mask], label=group, color=colors[colorcount], marker=markers[markercount], s=100, alpha=0.7)
            ax.legend(loc='lower right', fontsize=12)
            colorcount += 1
            if colorcount % len(colors) == 0:
                markercount += 1
                colorcount = 0

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel('Value of '+label, fontsize=16)
    #ax.set_xticklabels(rotation=45)
    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')

def plot_keras_history(model_history, savepath, plot_type):
    # Set image aspect ratio:
    fig, ax = make_fig_ax()
    keys = model_history.history.keys()
    for k in keys:
        if 'loss' not in k and 'val' not in k:
            metric = k
    accuracy = model_history.history[str(metric)]
    loss = model_history.history['loss']

    if plot_type == 'accuracy':
        ax.plot(accuracy, label='training '+str(metric))
        ax.set_ylabel(str(metric)+' (Accuracy)', fontsize=16)
        try:
            validation_accuracy = model_history.history['val_'+str(metric)]
            ax.plot(validation_accuracy, label='validation '+str(metric))
        except:
            pass
    if plot_type == 'loss':
        ax.plot(loss, label='training loss')
        ax.set_ylabel(str(metric)+' (Loss)', fontsize=16)
        try:
            validation_loss = model_history.history['val_loss']
            ax.plot(validation_loss, label='validation loss')
        except:
            pass
    ax.legend(loc='upper right', fontsize=12)

    #_set_tick_labels_different(ax, max_tick_x, min_tick_x, max_tick_y, min_tick_y)
    ax.set_xlabel('Epochs', fontsize=16)

    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')
    return

@ipynb_maker
def plot_best_worst_split(y_true, best_run, worst_run, savepath,
                          title='Best Worst Overlay', label='target_value'):
    """
    Method to create a parity plot (predicted vs. true values) of just the best scoring and worst scoring CV splits

    Args:

        y_true: (numpy array), array of true y data

        best_run: (dict), the best scoring split_result from mastml_driver

        worst_run: (dict), the worst scoring split_result from mastml_driver

        savepath: (str), path to save plots to

        title: (str), title of the best_worst_split plot

        label: (str), label used for axis labeling

    Returns:

        None

    """
    # make fig and ax, use x_align when placing text so things don't overlap
    x_align = 0.64
    fig, ax = make_fig_ax(x_align=x_align)

    maxx = max(y_true) # TODO is round the right thing here?
    minn = min(y_true)
    maxx = round(float(maxx), rounder(maxx-minn))
    minn = round(float(minn), rounder(maxx-minn))
    ax.plot([minn, maxx], [minn, maxx], 'k--', lw=2, zorder=1)

    # set tick labels
    _set_tick_labels(ax, maxx, minn)

    # do the actual plotting
    ax.scatter(best_run['y_test_true'],  best_run['y_test_pred'],  c='red',
               alpha=0.7, label='best test',  edgecolor='darkred',  zorder=2, s=100)
    ax.scatter(worst_run['y_test_true'], worst_run['y_test_pred'], c='blue',
               alpha=0.7, label='worst test', edgecolor='darkblue', zorder=3, s=60)
    ax.legend(loc='lower right', fontsize=12)

    # set axis labels
    ax.set_xlabel('True '+label, fontsize=16)
    ax.set_ylabel('Predicted '+label, fontsize=16)

    #font_dict = {'size'   : 10, 'family' : 'sans-serif'}

    # Duplicate the stats dicts with an additional label
    best_stats = OrderedDict([('Best Run', None)])
    best_stats.update(best_run['test_metrics'])
    worst_stats = OrderedDict([('worst Run', None)])
    worst_stats.update(worst_run['test_metrics'])

    plot_stats(fig, best_stats, x_align=x_align, y_align=0.90)
    plot_stats(fig, worst_stats, x_align=x_align, y_align=0.60)

    fig.savefig(savepath + '.png', dpi=DPI, bbox_inches='tight')

    df_best = pd.DataFrame({'best run pred': best_run['y_test_pred'], 'best run true': best_run['y_test_true']})
    df_worst = pd.DataFrame({'worst run pred': worst_run['y_test_pred'], 'worst run true': worst_run['y_test_true']})

    df_best.to_csv(savepath + '_best.csv')
    df_worst.to_csv(savepath + '_worst.csv')

@ipynb_maker
def plot_best_worst_per_point(y_true, y_pred_list, savepath, metrics_dict,
                              avg_stats, title='best worst per point', label='target_value'):
    """
    Method to create a parity plot (predicted vs. true values) of the set of best and worst CV scores for each

    individual data point.

    Args:

        y_true: (numpy array), array of true y data

        y_pred_list: (list), list of numpy arrays containing predicted y data for each CV split

        savepath: (str), path to save plots to

        metrics_dict: (dict), dict of scikit-learn metric objects to calculate score of predicted vs. true values

        avg_stats: (dict), dict of calculated average metrics over all CV splits

        title: (str), title of the best_worst_per_point plot

        label: (str), label used for axis labeling

    Returns:

        None

    """

    worsts = []
    bests = []
    new_y_true = []
    for yt, y_pred in zip(y_true, y_pred_list):
        if len(y_pred) == 0 or np.nan in y_pred_list or yt == np.nan:
            continue
        worsts.append(max(y_pred, key=lambda yp: abs(yp-yt)))
        bests.append( min(y_pred, key=lambda yp: abs(yp-yt)))
        new_y_true.append(yt)

    worst_stats = OrderedDict([('Worst combined:', None)])
    best_stats = OrderedDict([('Best combined:', None)])
    for name, (_, func) in metrics_dict.items():
        worst_stats[name] = func(new_y_true, worsts)
        best_stats[name] = func(new_y_true, bests)

    # make fig and ax, use x_align when placing text so things don't overlap
    x_align = 15.5/24 #mmm yum
    fig, ax = make_fig_ax(x_align=x_align)

    # gather max and min
    #all_vals = [val for val in worsts+bests if val is not None]
    max1 = max(y_true)
    min1 = min(y_true)

    # draw dashed horizontal line
    ax.plot([min1, max1], [min1, max1], 'k--', lw=2, zorder=1)

    # set axis labels
    ax.set_xlabel('True '+label, fontsize=16)
    ax.set_ylabel('Predicted '+label, fontsize=16)

    # set tick labels
    #maxx = max((max(bests), max(worsts), max(new_y_true)))
    #minn = min((min(bests), min(worsts), min(new_y_true)))
    #maxx, minn = recursive_max_and_min([bests, worsts, new_y_true])
    maxx = round(float(max1), rounder(max1-min1))
    minn = round(float(min1), rounder(max1-min1))
    _set_tick_labels(ax, maxx, minn)

    ax.scatter(new_y_true, bests,  c='red',  alpha=0.7, label='best test',
               edgecolor='darkred',  zorder=2, s=100)
    ax.scatter(new_y_true, worsts, c='blue', alpha=0.7, label='worst test',
               edgecolor='darkblue', zorder=3, s=60)
    ax.legend(loc='lower right', fontsize=12)

    plot_stats(fig, avg_stats, x_align=x_align, y_align=0.51, fontsize=10)
    plot_stats(fig, worst_stats, x_align=x_align, y_align=0.73, fontsize=10)
    plot_stats(fig, best_stats, x_align=x_align, y_align=0.95, fontsize=10)

    fig.savefig(savepath + '.png', dpi=DPI, bbox_inches='tight')

    df = pd.DataFrame({'y true': new_y_true,
                       'best per point': bests,
                       'worst per point': worsts})
    df.to_csv(savepath + '.csv')

@ipynb_maker
def plot_predicted_vs_true_bars(y_true, y_pred_list, avg_stats,
                                savepath, title='best worst with bars', label='target_value', groups=None):
    """
    Method to calculate parity plot (predicted vs. true) of average predictions, averaged over all CV splits, with error

    bars on each point corresponding to the standard deviation of the predicted values over all CV splits.

    Args:

        y_true: (numpy array), array of true y data

        y_pred_list: (list), list of numpy arrays containing predicted y data for each CV split

        avg_stats: (dict), dict of calculated average metrics over all CV splits

        savepath: (str), path to save plots to

        title: (str), title of the best_worst_per_point plot

        label: (str), label used for axis labeling

    Returns:

        None

    """
    means = [nice_mean(y_pred) for y_pred in y_pred_list]
    standard_error_means = [nice_std(y_pred)/np.sqrt(len(y_pred))
                            for y_pred in y_pred_list]
    standard_errors = [nice_std(y_pred) for y_pred in y_pred_list]
    # make fig and ax, use x_align when placing text so things don't overlap
    x_align = 0.64
    fig, ax = make_fig_ax(x_align=x_align)

    # gather max and min
    max1 = max(np.nanmax(y_true), np.nanmax(means))
    min1 = min(np.nanmin(y_true), np.nanmin(means))

    # draw dashed horizontal line
    ax.plot([min1, max1], [min1, max1], 'k--', lw=2, zorder=1)

    # set axis labels
    ax.set_xlabel('True '+label, fontsize=16)
    ax.set_ylabel('Predicted '+label, fontsize=16)

    # set tick labels
    #maxx, minn = recursive_max_and_min([means, y_true])
    maxx = max(y_true)
    minn = min(y_true)
    maxx = round(float(maxx), rounder(maxx-minn))
    minn = round(float(minn), rounder(maxx-minn))
    #print(maxx, minn, rounder(maxx - minn))
    _set_tick_labels(ax, maxx, minn)

    if groups is None:
        ax.errorbar(y_true, means, yerr=standard_errors, fmt='o', markerfacecolor='blue', markeredgecolor='black', markersize=10,
                alpha=0.7, capsize=3)
    else:
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'black']
        markers = ['o', 'v', '^', 's', 'p', 'h', 'D', '*', 'X', '<', '>', 'P']
        colorcount = markercount = 0
        handles = dict()
        unique_groups = np.unique(groups)
        for groupcount, group in enumerate(unique_groups):
            mask = groups == group
            # logger.debug(' '*12 + f'{group} group_percent = {np.count_nonzero(mask) / len(groups)}')
            handles[group] = ax.errorbar(y_true[mask], np.array(means)[mask], yerr=np.array(standard_errors)[mask],
                                         marker=markers[markercount], markerfacecolor=colors[colorcount],
                                         markeredgecolor=colors[colorcount], ecolor=colors[colorcount],
                                         markersize=10, alpha=0.7, capsize=3, fmt='o')
            colorcount += 1
            if colorcount % len(colors) == 0:
                markercount += 1
                colorcount = 0
        ax.legend(handles.values(), handles.keys(), loc='lower right', fontsize=10)

    plot_stats(fig, avg_stats, x_align=x_align, y_align=0.90)

    fig.savefig(savepath + '.png', dpi=DPI, bbox_inches='tight')

    df = pd.DataFrame({'y true': y_true,
                       'average predicted values': means,
                       'error bar values': standard_errors})
    df.to_csv(savepath + '.csv')

@ipynb_maker
def plot_metric_vs_group(metric, groups, stats, avg_stats, savepath):
    """
    Method to plot the value of a particular calculated metric (e.g. RMSE, R^2, etc) for each data group

    Args:

        metric: (str), name of a calculation metric

        groups: (numpy array), array of group names

        stats: (dict), dict of training or testing statistics for a particular run

        avg_stats: (dict), dict of calculated average metrics over all CV splits

        savepath: (str), path to save plots to

    Returns:

        None

    """

    # make fig and ax, use x_align when placing text so things don't overlap
    x_align = 0.64
    fig, ax = make_fig_ax(x_align=x_align)

    # do the actual plotting
    ax.scatter(groups,  stats,  c='blue', alpha=0.7, edgecolor='darkblue',  zorder=2, s=100)

    # set axis labels
    ax.set_xlabel('Group', fontsize=16)
    ax.set_ylabel(metric, fontsize=16)
    ax.set_xticklabels(labels=groups, fontsize=14)
    plot_stats(fig, avg_stats, x_align=x_align, y_align=0.90)

    # Save data stats to csv
    savepath_parse = savepath.split(str(metric)+'_vs_group.png')[0]
    pd.DataFrame(groups, stats).to_csv(os.path.join(savepath_parse, str(metric)+'_vs_group.csv'))

    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')
    return

@ipynb_maker
def plot_metric_vs_group_size(metric, groups, stats, avg_stats, savepath):
    """
    Method to plot the value of a particular calculated metric (e.g. RMSE, R^2, etc) as a function of the size of each group.

    Args:

        metric: (str), name of a calculation metric

        groups: (numpy array), array of group names

        stats: (dict), dict of training or testing statistics for a particular run

        avg_stats: (dict), dict of calculated average metrics over all CV splits

        savepath: (str), path to save plots to

    Returns:

        None

    """

    # make fig and ax, use x_align when placing text so things don't overlap
    x_align = 0.64
    fig, ax = make_fig_ax(x_align=x_align)

    # Get unique groups from full group array
    unique_groups = np.unique(groups)

    # Get the size of each group
    group_lengths = list()
    for group in unique_groups:
        group_lengths.append(len(np.concatenate(np.where(groups==group)).tolist()))

    # do the actual plotting
    ax.scatter(group_lengths,  stats,  c='blue', alpha=0.7, edgecolor='darkblue',  zorder=2, s=100)

    # set axis labels
    ax.set_xlabel('Group size', fontsize=16)
    ax.set_ylabel(metric, fontsize=16)
    #ax.set_xticklabels(labels=group_lengths, fontsize=14)
    plot_stats(fig, avg_stats, x_align=x_align, y_align=0.90)

    # Save data stats to csv
    savepath_parse = savepath.split(str(metric)+'_vs_group_size.png')[0]
    pd.DataFrame(group_lengths, stats).to_csv(os.path.join(savepath_parse, str(metric)+'_vs_group_size.csv'))

    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')
    return

def prediction_intervals(model, X, rf_error_method, rf_error_percentile, Xtrain, Xtest):
    """
    Method to calculate prediction intervals when using Random Forest and Gaussian Process regression models.

    Prediction intervals for random forest adapted from https://blog.datadive.net/prediction-intervals-for-random-forests/

    Args:

        model: (scikit-learn model/estimator object), a scikit-learn model object

        X: (numpy array), array of X features

        method: (str), type of error bar to formulate (e.g. "stdev" is standard deviation of predicted errors, "confint"
        is error bar as confidence interval

        percentile: (int), percentile for which to form error bars

    Returns:

        err_up: (list), list of upper bounds of error bars for each data point

        err_down: (list), list of lower bounds of error bars for each data point

    """
    err_down = list()
    err_up = list()
    X_aslist = X.values.tolist()
    if model.__class__.__name__ in ['RandomForestRegressor', 'GradientBoostingRegressor', 'ExtraTreesRegressor']:

        #if rf_error_method == 'jackknife':
        #    #new method based on http://contrib.scikit-learn.org/forest-confidence-interval/auto_examples/plot_mpg.html#sphx-glr-auto-examples-plot-mpg-py
        #    #inbag = fci.calc_inbag(Xtrain.shape[0], model)
        #    random_forest_variances = fci.random_forest_error(model, X_train=Xtrain, X_test=Xtest, calibrate=False)
        #    # remove NaN values from array
        #    random_forest_stdevs = np.sqrt(random_forest_variances)
        #    indices_to_ignore = np.argwhere(np.isnan((random_forest_stdevs)))
        #    random_forest_stdevs = random_forest_stdevs[~np.isnan(random_forest_stdevs)]
        #    err_up = random_forest_stdevs
        #    err_down = random_forest_stdevs
        if rf_error_method == 'jackknife':
            rf_variances = fci.random_forest_error(model, X_train=Xtrain, X_test=Xtest, calibrate=False)
            rf_stdevs = np.sqrt(rf_variances)
            indices_to_ignore = np.argwhere(np.isnan((rf_stdevs)))
            rf_stdevs = rf_stdevs[~np.isnan(rf_stdevs)]
            err_up = err_down = rf_stdevs

        else:
            for x in range(len(X_aslist)):
                preds = list()
                if model.__class__.__name__ == 'RandomForestRegressor':
                    for pred in model.estimators_:
                        preds.append(pred.predict(np.array(X_aslist[x]).reshape(1,-1))[0])
                elif model.__class__.__name__ == 'GradientBoostingRegressor':
                    for pred in model.estimators_.tolist():
                        preds.append(pred[0].predict(np.array(X_aslist[x]).reshape(1,-1))[0])
                if rf_error_method == 'confint':
                    e_down = np.percentile(preds, (100 - int(rf_error_percentile)) / 2.)
                    e_up = np.percentile(preds, 100 - (100 - int(rf_error_percentile)) / 2.)
                elif rf_error_method == 'stdev':
                    e_down = np.std(preds)
                    e_up = np.std(preds)
                elif rf_error_method == 'False' or rf_error_method is False:
                    # basically default to stdev
                    e_down = np.std(preds)
                    e_up = np.std(preds)
                if e_up == 0.0:
                    e_up = 10 ** 10
                if e_down == 0.0:
                    e_down = 10 ** 10
                err_down.append(e_down)
                err_up.append(e_up)

    if model.__class__.__name__=='GaussianProcessRegressor':
        preds = model.predict(X, return_std=True)[1] # Get the stdev model error from the predictions of GPR
        err_up = preds
        err_down = preds

    #if model.__class__.__name__=='ModelImport':
    #    if model.model.__class__.__name__=='GaussianProcessRegressor':
    #        from sklearn.gaussian_process import GaussianProcessRegressor
    #        kernel = model.model.kernel_
    #        gprmodel = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
    #        preds = gprmodel.predict(X, return_std=True)[1] # Get the stdev model error from the predictions of GPR
    #        err_up = preds
    #        err_down = preds
    return err_down, err_up

@ipynb_maker
def plot_normalized_error(y_true, y_pred, savepath, model, rf_error_method, rf_error_percentile, X=None, Xtrain=None,
                          Xtest=None):
    """
    Method to plot the normalized residual errors of a model prediction

    Args:

        y_true: (numpy array), array containing the true y data values

        y_pred: (numpy array), array containing the predicted y data values

        savepath: (str), path to save the plotted normalized error plot

        model: (scikit-learn model/estimator object), a scikit-learn model object

        X: (numpy array), array of X features

        avg_stats: (dict), dict of calculated average metrics over all CV splits

    Returns:

        None

    """

    path = os.path.dirname(savepath)
    # Here: if model is random forest or Gaussian process, get real error bars. Else, just residuals
    model_name = model.__class__.__name__
    # TODO: also add support for Gradient Boosted Regressor
    models_with_error_predictions = ['RandomForestRegressor', 'GaussianProcessRegressor', 'GradientBoostingRegressor']
    has_model_errors = False
    if model_name in models_with_error_predictions:
        has_model_errors = True
        err_down, err_up = prediction_intervals(model, X, rf_error_method=rf_error_method,
                                                rf_error_percentile=rf_error_percentile, Xtrain=Xtrain, Xtest=Xtest)

    y_pred_ = y_pred
    y_true_ = y_true

    x_align = 0.64
    fig, ax = make_fig_ax(x_align=x_align)
    mu = 0
    sigma = 1
    residuals = (y_true_ - y_pred_)
    normalized_residuals = (y_true_-y_pred_)/np.std(y_true_-y_pred_)
    density_residuals = gaussian_kde(normalized_residuals)
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, y_true_.shape[0])
    ax.plot(x, norm.pdf(x, mu, sigma), linewidth=4, color='blue', label="Analytical Gaussian")
    ax.plot(x, density_residuals(x), linewidth=4, color='green', label="Model Residuals")
    maxx = 5
    minn = -5

    if has_model_errors:
        err_avg = [(abs(e1)+abs(e2))/2 for e1, e2 in zip(err_up, err_down)]
        model_errors = (y_true_-y_pred_)/err_avg
        density_errors = gaussian_kde(model_errors)
        maxy = max(max(density_residuals(x)), max(norm.pdf(x, mu, sigma)), max(density_errors(x)))
        miny = min(min(density_residuals(x)), min(norm.pdf(x, mu, sigma)), max(density_errors(x)))
        ax.plot(x, density_errors(x), linewidth=4, color='purple', label="Model Errors")
        # Save data to csv file
        data_dict = {"Y True": y_true_, "Y Pred": y_pred_, "Plotted x values": x, "error_bars_up": err_up,
                     "error_bars_down": err_down, "error_avg": err_avg,
                     "analytical gaussian (plotted y blue values)": norm.pdf(x, mu, sigma),
                     "model residuals": residuals,
                     "model normalized residuals (plotted y green values)": density_residuals(x),
                     "model errors (plotted y purple values)": density_errors(x)}
        pd.DataFrame(data_dict).to_csv(savepath.split('.png')[0]+'.csv')
    else:
        # Save data to csv file
        data_dict = {"Y True": y_true, "Y Pred": y_pred, "x values": x, "analytical gaussian": norm.pdf(x, mu, sigma),
                    "model residuals": density_residuals(x)}
        pd.DataFrame(data_dict).to_csv(savepath.split('.png')[0]+'.csv')
        maxy = max(max(density_residuals(x)), max(norm.pdf(x, mu, sigma)))
        miny = min(min(density_residuals(x)), min(norm.pdf(x, mu, sigma)))

    ax.legend(loc=0, fontsize=12, frameon=False)
    ax.set_xlabel(r"$\mathrm{x}/\mathit{\sigma}$", fontsize=18)
    ax.set_ylabel("Probability density", fontsize=18)
    _set_tick_labels_different(ax, maxx, minn, maxy, miny)
    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')
    return

@ipynb_maker
def plot_cumulative_normalized_error(y_true, y_pred, savepath, model, rf_error_method, rf_error_percentile, X=None,
                                     Xtrain=None, Xtest=None):
    """
    Method to plot the cumulative normalized residual errors of a model prediction

    Args:

        y_true: (numpy array), array containing the true y data values

        y_pred: (numpy array), array containing the predicted y data values

        savepath: (str), path to save the plotted cumulative normalized error plot

        model: (scikit-learn model/estimator object), a scikit-learn model object

        X: (numpy array), array of X features

        avg_stats: (dict), dict of calculated average metrics over all CV splits

    Returns:

        None

    """

    # Here: if model is random forest or Gaussian process, get real error bars. Else, just residuals
    model_name = model.__class__.__name__
    models_with_error_predictions = ['RandomForestRegressor', 'GaussianProcessRegressor', 'GradientBoostingRegressor']
    has_model_errors = False
    if model_name in models_with_error_predictions:
        has_model_errors = True
        err_down, err_up = prediction_intervals(model, X, rf_error_method=rf_error_method,
                                                    rf_error_percentile=rf_error_percentile,  Xtrain=Xtrain, Xtest=Xtest)

    y_pred_ = y_pred
    y_true_ = y_true

    #Need to remove NaN's before plotting. These will be present when doing validation runs. Note NaN's only show up in y_pred_
    y_true_ = y_true_[~np.isnan(y_pred_)]
    y_pred_ = y_pred_[~np.isnan(y_pred_)]

    # Need to amend which y_true and y_pred we are considering by removing values from indices_to_ignore to make sure
    # length of these arrays and err_avg matches
    #y_true_ = np.delete(y_true_, indices_to_ignore)
    #y_pred_ = np.delete(y_pred_, indices_to_ignore)

    x_align = 0.64
    fig, ax = make_fig_ax(x_align=x_align)

    analytic_gau = np.random.normal(0, 1, 10000)
    analytic_gau = abs(analytic_gau)
    n_analytic = np.arange(1, len(analytic_gau) + 1) / np.float(len(analytic_gau))
    X_analytic = np.sort(analytic_gau)
    residuals = y_true_-y_pred_
    normalized_residuals = abs((y_true_-y_pred_)/np.std(y_true_-y_pred_))
    n_residuals = np.arange(1, len(normalized_residuals) + 1) / np.float(len(normalized_residuals))
    X_residuals = np.sort(normalized_residuals) #r"$\mathrm{Predicted \/ Value}, \mathit{eV}$"
    ax.set_xlabel(r"$\mathrm{x}/\mathit{\sigma}$", fontsize=18)
    ax.set_ylabel("Fraction", fontsize=18)
    ax.step(X_residuals, n_residuals, linewidth=3, color='green', label="Model Residuals")
    ax.step(X_analytic, n_analytic, linewidth=3, color='blue', label="Analytical Gaussian")
    ax.set_xlim([0, 5])

    if has_model_errors:
        err_avg = [(abs(e1)+abs(e2))/2 for e1, e2 in zip(err_up, err_down)]
        model_errors = abs((y_true_-y_pred_)/err_avg)
        n_errors = np.arange(1, len(model_errors) + 1) / np.float(len(model_errors))
        X_errors = np.sort(model_errors)
        ax.step(X_errors, n_errors, linewidth=3, color='purple', label="Model Errors")
        # Save data to csv file
        data_dict = {"Y True": y_true, "Y Pred": y_pred, "Analytical Gaussian values": analytic_gau, "Analytical Gaussian (sorted, blue data)": X_analytic,
                     "model residuals": residuals,
                     "Model normalized residuals": normalized_residuals, "Model Residuals (sorted, green data)": X_residuals,
                     "error_bars_up": err_up, "error_bars_down": err_down,
                     "Model error values (r value: (ytrue-ypred)/(model error avg))": model_errors,
                     "Model errors (sorted, purple values)": X_errors}
        # Save this way to avoid issue with different array sizes in data_dict
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df = df.transpose()
        df.to_csv(savepath.split('.png')[0]+'.csv', index=False)
    else:
        # Save data to csv file
        data_dict = {"Y True": y_true, "Y Pred": y_pred, "x analytical": X_analytic, "analytical gaussian": n_analytic, "x residuals": X_residuals,
                     "model residuals": n_residuals}
        # Save this way to avoid issue with different array sizes in data_dict
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df = df.transpose()
        df.to_csv(savepath.split('.png')[0]+'.csv', index=False)

    ax.legend(loc=0, fontsize=14, frameon=False)
    xlabels = np.linspace(2, 3, 3)
    ylabels = np.linspace(0.9, 1, 2)
    axin = zoomed_inset_axes(ax, 2.5, loc=7)
    axin.step(X_residuals, n_residuals, linewidth=3, color='green', label="Model Residuals")
    axin.step(X_analytic, n_analytic, linewidth=3, color='blue', label="Analytical Gaussian")
    if has_model_errors:
        axin.step(X_errors, n_errors, linewidth=3, color='purple', label="Model Errors")
    axin.set_xticklabels(xlabels, fontsize=8, rotation=90)
    axin.set_yticklabels(ylabels, fontsize=8)
    axin.set_xlim([2, 3])
    axin.set_ylim([0.9, 1])

    maxx = 5
    minn = 0
    maxy = 1.1
    miny = 0
    _set_tick_labels_different(ax, maxx, minn, maxy, miny)

    mark_inset(ax, axin, loc1=1, loc2=2)
    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')
    return

@ipynb_maker
def plot_average_cumulative_normalized_error(y_true, y_pred, savepath, has_model_errors, err_avg=None):
    """
    Method to plot the cumulative normalized residual errors of a model prediction

    Args:

        y_true: (numpy array), array containing the true y data values

        y_pred: (numpy array), array containing the predicted y data values

        savepath: (str), path to save the plotted cumulative normalized error plot

        model: (scikit-learn model/estimator object), a scikit-learn model object

        X: (numpy array), array of X features

        avg_stats: (dict), dict of calculated average metrics over all CV splits

    Returns:

        None

    """

    x_align = 0.64
    fig, ax = make_fig_ax(x_align=x_align)

    analytic_gau = np.random.normal(0, 1, 10000)
    analytic_gau = abs(analytic_gau)
    n_analytic = np.arange(1, len(analytic_gau) + 1) / np.float(len(analytic_gau))
    X_analytic = np.sort(analytic_gau)
    residuals = y_true-y_pred
    normalized_residuals = abs((y_true-y_pred)/np.std(y_true-y_pred))
    n_residuals = np.arange(1, len(normalized_residuals) + 1) / np.float(len(normalized_residuals))
    X_residuals = np.sort(normalized_residuals) #r"$\mathrm{Predicted \/ Value}, \mathit{eV}$"
    ax.set_xlabel(r"$\mathrm{x}/\mathit{\sigma}$", fontsize=18)
    ax.set_ylabel("Fraction", fontsize=18)
    ax.step(X_residuals, n_residuals, linewidth=3, color='green', label="Model Residuals")
    ax.step(X_analytic, n_analytic, linewidth=3, color='blue', label="Analytical Gaussian")
    ax.set_xlim([0, 5])

    if has_model_errors:
        model_errors = abs((y_true-y_pred)/err_avg)
        n_errors = np.arange(1, len(model_errors) + 1) / np.float(len(model_errors))
        X_errors = np.sort(model_errors)
        ax.step(X_errors, n_errors, linewidth=3, color='purple', label="Model Errors")
        # Save data to csv file
        data_dict = {"Y True": y_true, "Y Pred": y_pred, "Analytical Gaussian values": analytic_gau,
                     "Analytical Gaussian (sorted, blue data)": X_analytic,
                     "model residuals": residuals,
                     "Model normalized residuals": normalized_residuals, "Model Residuals (sorted, green data)": X_residuals,
                     "Model error values (r value: (ytrue-ypred)/(model error avg))": model_errors,
                     "Model errors (sorted, purple values)": X_errors}
        # Save this way to avoid issue with different array sizes in data_dict
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df = df.transpose()
        df.to_csv(savepath.split('.png')[0]+'.csv', index=False)
    else:
        # Save data to csv file
        data_dict = {"Y True": y_true, "Y Pred": y_pred, "x analytical": X_analytic, "analytical gaussian": n_analytic,
                     "x residuals": X_residuals, "model residuals": n_residuals}
        # Save this way to avoid issue with different array sizes in data_dict
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df = df.transpose()
        df.to_csv(savepath.split('.png')[0]+'.csv', index=False)

    ax.legend(loc=0, fontsize=14, frameon=False)
    xlabels = np.linspace(2, 3, 3)
    ylabels = np.linspace(0.9, 1, 2)
    axin = zoomed_inset_axes(ax, 2.5, loc=7)
    axin.step(X_residuals, n_residuals, linewidth=3, color='green', label="Model Residuals")
    axin.step(X_analytic, n_analytic, linewidth=3, color='blue', label="Analytical Gaussian")
    if has_model_errors:
        axin.step(X_errors, n_errors, linewidth=3, color='purple', label="Model Errors")
    axin.set_xticklabels(xlabels, fontsize=8, rotation=90)
    axin.set_yticklabels(ylabels, fontsize=8)
    axin.set_xlim([2, 3])
    axin.set_ylim([0.9, 1])

    maxx = 5
    minn = 0
    maxy = 1.1
    miny = 0
    _set_tick_labels_different(ax, maxx, minn, maxy, miny)

    mark_inset(ax, axin, loc1=1, loc2=2)
    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')
    return

@ipynb_maker
def plot_average_normalized_error(y_true, y_pred, savepath, has_model_errors, err_avg=None):
    """
    Method to plot the normalized residual errors of a model prediction

    Args:

        y_true: (numpy array), array containing the true y data values

        y_pred: (numpy array), array containing the predicted y data values

        savepath: (str), path to save the plotted normalized error plot

        model: (scikit-learn model/estimator object), a scikit-learn model object

        X: (numpy array), array of X features

        avg_stats: (dict), dict of calculated average metrics over all CV splits

    Returns:

        None

    """

    x_align = 0.64
    fig, ax = make_fig_ax(x_align=x_align)
    mu = 0
    sigma = 1
    residuals = y_true - y_pred
    normalized_residuals = (y_true-y_pred)/np.std(y_true-y_pred)
    density_residuals = gaussian_kde(normalized_residuals)
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, y_true.shape[0])
    ax.plot(x, norm.pdf(x, mu, sigma), linewidth=4, color='blue', label="Analytical Gaussian")
    ax.plot(x, density_residuals(x), linewidth=4, color='green', label="Model Residuals")
    maxx = 5
    minn = -5

    if has_model_errors:
        model_errors = (y_true-y_pred)/err_avg
        density_errors = gaussian_kde(model_errors)
        maxy = max(max(density_residuals(x)), max(norm.pdf(x, mu, sigma)), max(density_errors(x)))
        miny = min(min(density_residuals(x)), min(norm.pdf(x, mu, sigma)), min(density_errors(x)))
        ax.plot(x, density_errors(x), linewidth=4, color='purple', label="Model Errors")
        # Save data to csv file
        data_dict = {"Y True": y_true, "Y Pred": y_pred, "Plotted x values": x, "Model errors": err_avg,
                     "analytical gaussian (plotted y blue values)": norm.pdf(x, mu, sigma),
                     "model residuals": residuals,
                     "model normalized residuals (plotted y green values)": density_residuals(x),
                     "model errors (plotted y purple values)": density_errors(x)}
        pd.DataFrame(data_dict).to_csv(savepath.split('.png')[0]+'.csv')
    else:
        # Save data to csv file
        data_dict = {"Y True": y_true, "Y Pred": y_pred, "x values": x, "analytical gaussian": norm.pdf(x, mu, sigma),
                    "model residuals": density_residuals(x)}
        pd.DataFrame(data_dict).to_csv(savepath.split('.png')[0]+'.csv')
        maxy = max(max(density_residuals(x)), max(norm.pdf(x, mu, sigma)))
        miny = min(min(density_residuals(x)), min(norm.pdf(x, mu, sigma)))

    ax.legend(loc=0, fontsize=12, frameon=False)
    ax.set_xlabel(r"$\mathrm{x}/\mathit{\sigma}$", fontsize=18)
    ax.set_ylabel("Probability density", fontsize=18)
    _set_tick_labels_different(ax, maxx, minn, maxy, miny)
    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')
    return

def plot_1d_heatmap(xs, heats, savepath, xlabel='x', heatlabel='heats'):
    """
    Method to plot a heatmap for values of a single variable; used for plotting GridSearch results in hyperparameter optimization.

    Args:

        xs: (numpy array), array of first variable values to plot heatmap against

        heats: (numpy array), array of heat values to plot

        savepath: (str), path to save the 1D heatmap to

        xlabel: (str), the x-axis label

        heatlabel: (str), the heat value axis label

    """

    #TODO have more general solution
    try:
        fig, ax = make_fig_ax()
        ax.bar(xs, heats)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(heatlabel)

        fig.savefig(savepath, dpi=DPI, bbox_inches='tight')
    # Escape from error of passing tuples when optimizing neural net
    except TypeError:
        pass


def plot_2d_heatmap(xs, ys, heats, savepath,
                    xlabel='x', ylabel='y', heatlabel='heat'):
    """
    Method to plot a heatmap for values of two variables; used for plotting GridSearch results in hyperparameter optimization.

    Args:

        xs: (numpy array), array of first variable values to plot heatmap against

        ys: (numpy array), array of second variable values to plot heatmap against

        heats: (numpy array), array of heat values to plot

        savepath: (str), path to save the 2D heatmap to

        xlabel: (str), the x-axis label

        ylabel: (str), the y-axis label

        heatlabel: (str), the heat value axis label

    """
    #TODO have more general solution
    try:
        fig, ax = make_fig_ax()
        scat = ax.scatter(xs, ys, c=heats) # marker='o', lw=0, s=20, cmap=cm.plasma
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cb = fig.colorbar(scat)
        cb.set_label(heatlabel)

        fig.savefig(savepath, dpi=DPI, bbox_inches='tight')
    # Escape from error of passing tuples when optimizing neural net
    except TypeError:
        pass

def plot_3d_heatmap(xs, ys, zs, heats, savepath,
                    xlabel='x', ylabel='y', zlabel='z', heatlabel='heat'):
    """
    Method to plot a heatmap for values of three variables; used for plotting GridSearch results in hyperparameter optimization.

    Args:

        xs: (numpy array), array of first variable values to plot heatmap against

        ys: (numpy array), array of second variable values to plot heatmap against

        zs: (numpy array), array of third variable values to plot heatmap against

        heats: (numpy array), array of heat values to plot

        savepath: (str), path to save the 2D heatmap to

        xlabel: (str), the x-axis label

        ylabel: (str), the y-axis label

        zlabel: (str), the z-axis label

        heatlabel: (str), the heat value axis label

    """
    # Escape from error of passing tuples when optimzing neural net
    # TODO have more general solution
    try:
        # this import has side effects, needed for 3d plots:
        from mpl_toolkits.mplot3d import Axes3D
        # Set image aspect ratio:
        # (eeds to be wide enough or plot will shrink really skinny)
        w, h = figaspect(0.6)
        fig = Figure(figsize=(w,h))
        FigureCanvas(fig) # modifies fig in place
        ax = fig.add_subplot(111, projection='3d')

        scat = ax.scatter(xs, ys, zs, c=heats)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        cb = fig.colorbar(scat)
        cb.set_label(heatlabel)

        fig.savefig(savepath, dpi=DPI, bbox_inches='tight')
    except TypeError:
        pass

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return [fig]
    anim = FuncAnimation(fig, animate, frames=range(0,90,5), blit=True)
    #anim.save(savepath+'.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
    anim.save(savepath+'.gif', fps=5, dpi=80, writer='imagemagick')

@ipynb_maker
def plot_learning_curve(train_sizes, train_mean, test_mean, train_stdev, test_stdev, score_name, learning_curve_type, savepath='data_learning_curve'):
    """
    Method used to plot both data and feature learning curves

    Args:

        train_sizes: (numpy array), array of x-axis values, such as fraction of data used or number of features

        train_mean: (numpy array), array of training data mean values, averaged over some type/number of CV splits

        test_mean: (numpy array), array of test data mean values, averaged over some type/number of CV splits

        train_stdev: (numpy array), array of training data standard deviation values, from some type/number of CV splits

        test_stdev: (numpy array), array of test data standard deviation values, from some type/number of CV splits

        score_name: (str), type of score metric for learning curve plotting; used in y-axis label

        learning_curve_type: (str), type of learning curve employed: 'sample_learning_curve' or 'feature_learning_curve'

        savepath: (str), path to save the plotted learning curve to

    Returns:

        None

    """
    # Set image aspect ratio (do custom for learning curve):
    w, h = figaspect(0.75)
    fig = Figure(figsize=(w,h))
    FigureCanvas(fig)
    gs = plt.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0:, 0:])

    max_x = max(train_sizes)
    min_x = min(train_sizes)

    max_y, min_y = recursive_max_and_min([
        train_mean,
        train_mean + train_stdev,
        train_mean - train_stdev,
        test_mean,
        test_mean + test_stdev,
        test_mean - test_stdev,
        ])

    max_x = round(float(max_x), rounder(max_x-min_x))
    min_x = round(float(min_x), rounder(max_x-min_x))
    max_y = round(float(max_y), rounder(max_y-min_y))
    min_y = round(float(min_y), rounder(max_y-min_y))
    _set_tick_labels_different(ax, max_x, min_x, max_y, min_y)

    # plot and collect handles h1 and h2 for making legend
    h1 = ax.plot(train_sizes, train_mean, '-o', color='blue', markersize=10, alpha=0.7)[0]
    ax.fill_between(train_sizes, train_mean-train_stdev, train_mean+train_stdev,
                     alpha=0.1, color='blue')
    h2 = ax.plot(train_sizes, test_mean, '-o', color='red', markersize=10, alpha=0.7)[0]
    ax.fill_between(train_sizes, test_mean-test_stdev, test_mean+test_stdev,
                     alpha=0.1, color='red')
    ax.legend([h1, h2], ['train score', 'validation score'], loc='center right', fontsize=12)
    if learning_curve_type == 'sample_learning_curve':
        ax.set_xlabel('Number of training data points', fontsize=16)
    elif learning_curve_type == 'feature_learning_curve':
        ax.set_xlabel('Number of features selected', fontsize=16)
    else:
        raise ValueError('The param "learning_curve_type" must be either "sample_learning_curve" or "feature_learning_curve"')
    ax.set_ylabel(score_name, fontsize=16)
    fig.savefig(savepath+'.png', dpi=DPI, bbox_inches='tight')

    # Save output data to spreadsheet
    df_concat = pd.concat([pd.DataFrame(train_sizes), pd.DataFrame(train_mean), pd.DataFrame(train_stdev),
                           pd.DataFrame(test_mean), pd.DataFrame(test_stdev)], 1)
    df_concat.columns = ['train_sizes', 'train_mean', 'train_stdev', 'test_mean', 'test_stdev']
    df_concat.to_csv(savepath+'.csv')
    try:
        plot_learning_curve_convergence(train_sizes, test_mean, score_name, learning_curve_type, savepath)
    except IndexError:
        logger.error('MASTML encountered an error while trying to plot the learning curve convergences plots, likely due to '
                  'insufficient data')

@ipynb_maker
def plot_learning_curve_convergence(train_sizes, test_mean, score_name, learning_curve_type, savepath):
    """
    Method used to plot both the convergence of data and feature learning curves as a function of amount of data or features

    used, respectively.

    Args:

        train_sizes: (numpy array), array of x-axis values, such as fraction of data used or number of features

        test_mean: (numpy array), array of test data mean values, averaged over some type/number of CV splits

        score_name: (str), type of score metric for learning curve plotting; used in y-axis label

        learning_curve_type: (str), type of learning curve employed: 'sample_learning_curve' or 'feature_learning_curve'

        savepath: (str), path to save the plotted convergence learning curve to

    Returns:

        None

    """

    # Function to examine the minimization of error in learning curve CV scores as function of amount of data or number
    # of features used.
    steps = [x for x in range(len(train_sizes))]
    test_mean = test_mean.tolist()
    slopes = list()
    for step, val in zip(range(len(steps)), range(len(test_mean))):
        if step+1 < len(steps):
            slopes.append((test_mean[val + 1] - test_mean[val]) / (steps[step + 1] - steps[step]))

    # Remove first entry to steps for plotting
    del steps[0]

    # Get moving average of slopes to generate smoother curve
    window_size = round(len(test_mean)/3)
    steps_moving_average = pd.DataFrame(steps).rolling(window=window_size).mean()
    slopes_moving_average = pd.DataFrame(slopes).rolling(window=window_size).mean()

    #### Plotting
    # Set image aspect ratio (do custom for learning curve):
    w, h = figaspect(0.75)
    fig = Figure(figsize=(w,h))
    FigureCanvas(fig)
    gs = plt.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0:, 0:])

    max_x = max(steps)
    min_x = min(steps)
    max_y = max(slopes)
    min_y = min(slopes)
    _set_tick_labels_different(ax, max_x, min_x, max_y, min_y)

    ax.plot(steps, slopes, '-o', color='blue', markersize=10, alpha=0.7)
    ax.plot(steps_moving_average, slopes_moving_average, '-o', color='green', markersize=10, alpha=0.7)
    ax.legend(['score slope', 'smoothed score slope'], loc='lower right', fontsize=12)
    ax.set_xlabel('Learning curve step', fontsize=16)
    ax.set_ylabel('Change in '+score_name, fontsize=16)
    fig.savefig(savepath+'_convergence'+'.png', dpi=DPI, bbox_inches='tight')

    datadict = {"Steps": np.array(steps), "Slopes": np.array(slopes),
                "Steps moving average": np.squeeze(np.array(steps_moving_average)),
                "Slopes moving average": np.squeeze(np.array(slopes_moving_average))}

    pd.DataFrame().from_dict(data=datadict).to_csv(savepath+'_convergence.csv')

    if learning_curve_type == 'feature_learning_curve':
        # First, set number optimal features to all features in case stopping criteria not met
        n_features_optimal = len(test_mean)
        # Determine what the optimal number of features are to use
        # TODO: have user set stopping criteria. Have default by 0.005 (0.5%)
        stopping_criteria = 0.005
        slopes_moving_average_list = slopes_moving_average.values.reshape(-1,).tolist()
        for i, slope in enumerate(slopes_moving_average_list):
            try:
                delta = abs(-1*(slopes_moving_average_list[i+1] - slopes_moving_average_list[i])/slopes_moving_average_list[i])
                if delta < stopping_criteria:
                    n_features_optimal = steps[i]
            except IndexError:
                break

        # Write optimal number of features to text file
        f = open(savepath+'_optimalfeaturenumber.txt', 'w')
        f.write(str(n_features_optimal)+'\n')
        f.close()

### Helpers:

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

def get_histogram_bins(y_df):
    """
    Method to obtain the number of bins to use when plotting a histogram

    Args:

        y_df: (pandas Series or numpy array), array of y data used to construct histogram

    Returns:

        num_bins: (int), the number of bins to use when plotting a histogram

    """

    bin_dividers = np.linspace(y_df.shape[0], 0.05*y_df.shape[0], y_df.shape[0])
    bin_list = list()
    try:
        for divider in bin_dividers:
            if divider == 0:
                continue
            bins = int((y_df.shape[0])/divider)
            if bins < y_df.shape[0]/2:
                bin_list.append(bins)
    except:
        num_bins = 10
    if len(bin_list) > 0:
        num_bins = max(bin_list)
    else:
        num_bins = 10
    return num_bins

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
    fig = Figure(figsize=(w,h))
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
    'R2': '$R^2$',
    'R2_noint': '$R^2_{noint}$',
    'R2_adjusted': '$R^2_{adjusted}$',
    'R2_fitted': '$R^2_{fitted}$'
    }
    return nice_names