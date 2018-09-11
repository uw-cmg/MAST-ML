"""
A collection of functions which make plots (png files) using matplotlib.
Most of these plots take in (data, other_data, ..., savepath, stats, title).
Where the data args are numpy arrays, savepath is a string, and stats is an
ordered dictionary which maps names to either values, or mean-stdev pairs.

A plot can also take an "outdir" instead of a savepath. If this is the case,
it must return a list of filenames where it saved the figures.
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
import matplotlib.mlab as mlab
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

from .utils import nice_range # TODO include this in ipynb_helper

matplotlib.rc('font', size=18, family='sans-serif') # set all font to bigger
matplotlib.rc('figure', autolayout=True) # turn on autolayout

# adding dpi as a constant global so it can be changed later
DPI = 250

log = logging.getLogger() # only used inside ipynb_maker I guess

# HEADERENDER don't delete this line, it's used by ipynb maker

log = logging.getLogger('mastml') # the real logger

from .ipynb_maker import ipynb_maker # TODO: fix cyclic import
from .metrics import nice_names

def make_train_test_plots(run, path, is_classification, label, model, train_X, test_X, groups=None):
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

        title = 'train_normalized_error'
        plot_normalized_error(y_train_true, y_train_pred, join(path, title+'.png'), model, train_X)

        title = 'test_normalized_error'
        plot_normalized_error(y_test_true, y_test_pred, join(path, title+'.png'), model, test_X)

        title = 'train_cumulative_normalized_error'
        plot_cumulative_normalized_error(y_train_true, y_train_pred, join(path, title+'.png'), model, train_X)

        title = 'test_cumulative_normalized_error'
        plot_cumulative_normalized_error(y_test_true, y_test_pred, join(path, title+'.png'), model, test_X)

        title = 'train_residuals_histogram'
        plot_residuals_histogram(y_train_true, y_train_pred,
                                 join(path, title+'.png'), train_metrics,
                                 title=title, label=label)
        title = 'test_residuals_histogram'
        plot_residuals_histogram(y_test_true,  y_test_pred,
                                 join(path, title+'.png'), test_metrics,
                                 title=title, label=label)

### Core plotting utilities:

@ipynb_maker
def plot_confusion_matrix(y_true, y_pred, savepath, stats, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
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

    # make fig and ax, use x_align when placing text so things don't overlap
    x_align = 0.70
    fig, ax = make_fig_ax(aspect_ratio=0.5, x_align=x_align)

    #ax.set_title(title)

    #Get num_bins using smarter method
    num_bins = get_histogram_bins(y_df=y_df)

    # do the actual plotting
    ax.hist(y_df, bins=num_bins, color='b', edgecolor='k')#, histtype='stepfilled')

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
    filenames = list()
    y_train_true, y_train_pred, train_metrics, train_groups = train_quad
    y_test_true, y_test_pred, test_metrics, test_groups = test_quad

    # make diagonal line from absolute min to absolute max of any data point
    # using round because Ryan did - but won't that ruin small numbers??? TODO this
    max1 = max(y_train_true.max(), y_train_pred.max(),
               y_test_true.max(), y_test_pred.max())
    min1 = min(y_train_true.min(), y_train_pred.min(),
               y_test_true.min(), y_test_pred.min())
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
            log.debug(' '*12 + 'unique groups: ' +str(list(unique_groups)))
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'black', 'yellow',
                      'blue', 'red', 'green', 'purple', 'orange', 'black', 'yellow',
                      'blue', 'red', 'green', 'purple', 'orange', 'black', 'yellow',
                      'blue', 'red', 'green', 'purple', 'orange', 'black', 'yellow',
                      'blue', 'red', 'green', 'purple', 'orange', 'black', 'yellow',
                      'blue', 'red', 'green', 'purple', 'orange', 'black', 'yellow']
            markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o',
                       'v', 'v', 'v', 'v', 'v', 'v', 'v',
                       '^', '^', '^', '^', '^', '^', '^',
                       's', 's', 's', 's', 's', 's', 's',
                       'p', 'p', 'p', 'p', 'p', 'p', 'p',
                       'h', 'h', 'h', 'h', 'h', 'h', 'h']
            for groupcount, group in enumerate(unique_groups):
                markercount = groupcount*len(colors)-groupcount
                mask = groups == group
                log.debug(' '*12 + f'{group} group_percent = {np.count_nonzero(mask) / len(groups)}')
                handles[group] = ax.scatter(y_true[mask], y_pred[mask], label=group, color=colors[groupcount],
                                            marker=markers[groupcount], s=100, alpha=0.7)
            ax.legend(handles.values(), handles.keys(), loc='lower right', fontsize=12)

        # set axis labels
        ax.set_xlabel('True '+label, fontsize=16)
        ax.set_ylabel('Predicted '+label, fontsize=16)

        plot_stats(fig, stats, x_align=x_align, y_align=0.90)

        filename = 'predicted_vs_true_'+ title_addon + '.png'
        filenames.append(filename)
        fig.savefig(join(outdir, filename), dpi=DPI, bbox_inches='tight')

    return filenames

def plot_scatter(x, y, savepath, groups=None, xlabel='x', ylabel='y', label='target data'):
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
        for groupcount, group in enumerate(np.unique(groups)):
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'black', 'yellow',
                      'blue', 'red', 'green', 'purple', 'orange', 'black', 'yellow',
                      'blue', 'red', 'green', 'purple', 'orange', 'black', 'yellow',
                      'blue', 'red', 'green', 'purple', 'orange', 'black', 'yellow',
                      'blue', 'red', 'green', 'purple', 'orange', 'black', 'yellow',
                      'blue', 'red', 'green', 'purple', 'orange', 'black', 'yellow']
            markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o',
                       'v', 'v', 'v', 'v', 'v', 'v', 'v',
                       '^', '^', '^', '^', '^', '^', '^',
                       's', 's', 's', 's', 's', 's', 's',
                       'p', 'p', 'p', 'p', 'p', 'p', 'p',
                       'h', 'h', 'h', 'h', 'h', 'h', 'h']
            mask = groups == group
            ax.scatter(x[mask], y[mask], label=group, color=colors[groupcount], marker=markers[groupcount], s=100, alpha=0.7)
            ax.legend(loc='lower right', fontsize=12)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel('Value of '+label, fontsize=16)
    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')

@ipynb_maker
def plot_best_worst_split(y_true, best_run, worst_run, savepath,
                          title='Best Worst Overlay', label='target_value'):

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

    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')

@ipynb_maker
def plot_best_worst_per_point(y_true, y_pred_list, savepath, metrics_dict,
                              avg_stats, title='best worst per point', label='target_value'):
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

    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')

@ipynb_maker
def plot_predicted_vs_true_bars(y_true, y_pred_list, avg_stats,
                                savepath, title='best worst with bars', label='target_value'):
    " EVERYTHING MUST BE ARRAYS DONT GIVE ME DEM DF "
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

    ax.errorbar(y_true, means, yerr=standard_errors, fmt='o', markerfacecolor='blue', markeredgecolor='black', markersize=10,
                alpha=0.7, capsize=3)

    plot_stats(fig, avg_stats, x_align=x_align, y_align=0.90)

    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')

def plot_metric_vs_group(metric, groups, stats, avg_stats, savepath):
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

# Prediction intervals adapted from https://blog.datadive.net/prediction-intervals-for-random-forests/
def prediction_intervals(model, X, percentile=68):
    err_down = list()
    err_up = list()
    X_aslist = X.values.tolist()
    for x in range(len(X_aslist)):
        preds = list()
        if model.__class__.__name__=='RandomForestRegressor':
            for pred in model.estimators_:
                preds.append(pred.predict(np.array(X_aslist[x]).reshape(1,-1))[0])

        elif model.__class__.__name__=='GaussianProcessRegressor':
            preds = model.predict(X, return_std=True)

        e_down = np.percentile(preds, (100 - percentile) / 2. )
        e_up = np.percentile(preds, 100 - (100 - percentile) / 2.)
        if e_up == 0.0:
            e_up = 10**10
        if e_down == 0.0:
            e_down = 10**10
        err_down.append(e_down)
        err_up.append(e_up)
    return err_down, err_up

def plot_normalized_error(y_true, y_pred, savepath, model, X=None, avg_stats=None):
    path = os.path.dirname(savepath)
    # Here: if model is random forest or Gaussian process, get real error bars. Else, just residuals
    model_name = model.__class__.__name__
    # TODO: also add support for Gradient Boosted Regressor
    models_with_error_predictions = ['RandomForestRegressor', 'GaussianProcessRegressor']
    has_model_errors = False
    if model_name in models_with_error_predictions:
        has_model_errors = True
        if not avg_stats:
            err_down, err_up = prediction_intervals(model, X, percentile=68)

    if avg_stats:
        y_pred_ = np.array([nice_mean(y_p) for y_p in y_pred])
        y_true_ = y_true
    else:
        y_pred_ = y_pred
        y_true_ = y_true

    #Need to remove NaN's before plotting. These will be present when doing validation runs. Note NaN's only show up in y_pred_
    y_true_ = y_true_[~np.isnan(y_pred_)]
    y_pred_ = y_pred_[~np.isnan(y_pred_)]

    x_align = 0.64
    fig, ax = make_fig_ax(x_align=x_align)
    mu = 0
    sigma = 1
    residuals = (y_true_-y_pred_)/np.std(y_true_-y_pred_)
    density_residuals = gaussian_kde(residuals)
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, y_true_.shape[0])
    ax.plot(x, mlab.normpdf(x, mu, sigma), linewidth=4, color='blue', label="Analytical Gaussian")
    ax.plot(x, density_residuals(x), linewidth=4, color='green', label="Model Residuals")
    maxx = 5
    minn = -5
    maxy = max(max(density_residuals(x)), max(mlab.normpdf(x, mu, sigma)))
    miny = min(min(density_residuals(x)), min(mlab.normpdf(x, mu, sigma)))
    if not avg_stats:
        if has_model_errors:
            err_avg = [(abs(e1)+abs(e2))/2 for e1, e2 in zip(err_up, err_down)]
            model_errors = (y_true_-y_pred_)/err_avg
            density_errors = gaussian_kde(model_errors)
            ax.plot(x, density_errors(x), linewidth=4, color='purple', label="Model Errors")
            maxy = max(max(density_residuals(x)), max(mlab.normpdf(x, mu, sigma)), max(density_errors(x)))
            miny = min(min(density_residuals(x)), min(mlab.normpdf(x, mu, sigma)), min(density_errors(x)))
            # Save data to csv file
            data_dict = {"x values": x, "analytical gaussian": mlab.normpdf(x, mu, sigma),
                         "model residuals": density_residuals(x), "model errors": density_errors(x)}
            pd.DataFrame(data_dict).to_csv(savepath.split('.png')[0]+'.csv')
        else:
            # Save data to csv file
            data_dict = {"x values": x, "analytical gaussian": mlab.normpdf(x, mu, sigma),
                         "model residuals": density_residuals(x)}
            pd.DataFrame(data_dict).to_csv(savepath.split('.png')[0]+'.csv')
    if avg_stats:
        if has_model_errors:
            # Need to loop over all splits in directory and import saved model errors from csv files
            dirs = [d[0] for d in os.walk(path)]
            # Remove base directory from list
            del dirs[0]
            # Should only use average errors of test data since predictions are made using test data
            test_err_dfs = list()
            for dir in dirs:
                test_err_dfs.append(np.array(pd.read_csv(dir+'/'+'test_normalized_error.csv').loc[:, 'model errors']))
            test_err_dfs = trim_array(test_err_dfs)
            avg_test_model_errors = np.mean(test_err_dfs, axis=0)
            x_reduced = np.linspace(mu - 5 * sigma, mu + 5 * sigma, avg_test_model_errors.shape[0])
            ax.plot(x_reduced, avg_test_model_errors, linewidth=4, color='purple', label="Model Errors")

    ax.legend(loc=0, fontsize=12, frameon=False)
    ax.set_xlabel(r"$\mathrm{x}/\mathit{\sigma}$", fontsize=18)
    ax.set_ylabel("Probability density", fontsize=18)
    _set_tick_labels_different(ax, maxx, minn, maxy, miny)
    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')
    return

def plot_cumulative_normalized_error(y_true, y_pred, savepath, model, X=None, avg_stats=None):
    path = os.path.dirname(savepath)
    # Here: if model is random forest or Gaussian process, get real error bars. Else, just residuals
    model_name = model.__class__.__name__
    # TODO: also add support for Gradient Boosted Regressor
    models_with_error_predictions = ['RandomForestRegressor', 'GaussianProcessRegressor']
    has_model_errors = False
    if model_name in models_with_error_predictions:
        has_model_errors = True
        if not avg_stats:
            err_down, err_up = prediction_intervals(model, X, percentile=68)

    if avg_stats:
        y_pred_ = np.array([nice_mean(y_p) for y_p in y_pred])
        y_true_ = y_true
    else:
        y_pred_ = y_pred
        y_true_ = y_true

    #Need to remove NaN's before plotting. These will be present when doing validation runs. Note NaN's only show up in y_pred_
    y_true_ = y_true_[~np.isnan(y_pred_)]
    y_pred_ = y_pred_[~np.isnan(y_pred_)]

    x_align = 0.64
    fig, ax = make_fig_ax(x_align=x_align)

    analytic_gau = np.random.normal(0, 1, 10000)
    analytic_gau = abs(analytic_gau)
    n_analytic = np.arange(1, len(analytic_gau) + 1) / np.float(len(analytic_gau))
    X_analytic = np.sort(analytic_gau)
    residuals = abs((y_true_-y_pred_)/np.std(y_true_-y_pred_))
    n_residuals = np.arange(1, len(residuals) + 1) / np.float(len(residuals))
    X_residuals = np.sort(residuals) #r"$\mathrm{Predicted \/ Value}, \mathit{eV}$"
    ax.set_xlabel(r"$\mathrm{x}/\mathit{\sigma}$", fontsize=18)
    ax.set_ylabel("Fraction", fontsize=18)
    ax.step(X_residuals, n_residuals, linewidth=3, color='green', label="Model Residuals")
    ax.step(X_analytic, n_analytic, linewidth=3, color='blue', label="Analytical Gaussian")

    if not avg_stats:
        if has_model_errors:
            err_avg = [(abs(e1)+abs(e2))/2 for e1, e2 in zip(err_up, err_down)]
            model_errors = abs((y_true_-y_pred_)/err_avg)
            n_errors = np.arange(1, len(model_errors) + 1) / np.float(len(model_errors))
            X_errors = np.sort(model_errors)
            ax.step(X_errors, n_errors, linewidth=3, color='purple', label="Model Errors")
            # Save data to csv file
            data_dict = {"x analytical": X_analytic, "analytical gaussian": n_analytic, "x residuals": X_residuals,
                         "model residuals": n_residuals, "x errors": X_errors, "model errors": n_errors}
            # Save this way to avoid issue with different array sizes in data_dict
            df = pd.DataFrame.from_dict(data_dict, orient='index')
            df = df.transpose()
            df.to_csv(savepath.split('.png')[0]+'.csv', index=False)
        else:
            # Save data to csv file
            data_dict = {"x analytical": X_analytic, "analytical gaussian": n_analytic, "x residuals": X_residuals,
                         "model residuals": n_residuals}
            # Save this way to avoid issue with different array sizes in data_dict
            df = pd.DataFrame.from_dict(data_dict, orient='index')
            df = df.transpose()
            df.to_csv(savepath.split('.png')[0]+'.csv', index=False)
    if avg_stats:
        if has_model_errors:
            # Need to loop over all splits in directory and import saved model errors from csv files
            dirs = [d[0] for d in os.walk(path)]
            # Remove base directory from list
            del dirs[0]
            # Should only use average errors of test data since predictions are made using test data
            test_err_dfs = list()
            X_errors_dfs = list()
            for dir in dirs:
                test_err_dfs.append(np.array(pd.read_csv(dir + '/' + 'test_cumulative_normalized_error.csv').loc[:, 'model errors']))
                X_errors_dfs.append(np.array(pd.read_csv(dir + '/' + 'test_cumulative_normalized_error.csv').loc[:, 'x errors']))
            test_err_dfs = trim_array(test_err_dfs)
            X_errors_dfs = trim_array(X_errors_dfs)
            avg_test_model_errors = np.mean(test_err_dfs, axis=0)
            avg_X_errors = np.mean(X_errors_dfs, axis=0)
            ax.step(avg_X_errors, avg_test_model_errors, linewidth=4, color='purple', label="Model Errors")

    ax.legend(loc=0, fontsize=14, frameon=False)
    xlabels = np.linspace(2, 3, 3)
    ylabels = np.linspace(0.9, 1, 2)
    axin = zoomed_inset_axes(ax, 2.5, loc=7)
    axin.step(X_residuals, n_residuals, linewidth=3, color='green', label="Model Residuals")
    axin.step(X_analytic, n_analytic, linewidth=3, color='blue', label="Analytical Gaussian")
    if has_model_errors:
        if not avg_stats:
            axin.step(X_errors, n_errors, linewidth=3, color='purple', label="Model Errors")
        if avg_stats:
            axin.step(avg_X_errors, avg_test_model_errors, linewidth=3, color='purple', label="Model Errors")
    axin.set_xticklabels(xlabels, fontsize=8)
    axin.set_yticklabels(ylabels, fontsize=8)
    axin.set_xlim(2, 3)
    axin.set_ylim(0.9, 1)

    maxx = 5
    minn = 0
    maxy = 1
    miny = 0
    _set_tick_labels_different(ax, maxx, minn, maxy, miny)

    mark_inset(ax, axin, loc1=1, loc2=2)
    fig.savefig(savepath, dpi=DPI, bbox_inches='tight')
    return

def plot_1d_heatmap(xs, heats, savepath, xlabel='x', heatlabel='heats'):
    # Escape from error of passing tuples when optimzing neural net
    #TODO have more general solution
    try:
        fig, ax = make_fig_ax()
        ax.bar(xs, heats)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(heatlabel)

        fig.savefig(savepath, dpi=DPI, bbox_inches='tight')
    except TypeError:
        pass


def plot_2d_heatmap(xs, ys, heats, savepath,
                    xlabel='x', ylabel='y', heatlabel='heat'):
    # Escape from error of passing tuples when optimzing neural net
    #TODO have more general solution
    try:
        fig, ax = make_fig_ax()
        scat = ax.scatter(xs, ys, c=heats) # marker='o', lw=0, s=20, cmap=cm.plasma
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cb = fig.colorbar(scat)
        cb.set_label(heatlabel)

        fig.savefig(savepath, dpi=DPI, bbox_inches='tight')
    except TypeError:
        pass

def plot_3d_heatmap(xs, ys, zs, heats, savepath,
                    xlabel='x', ylabel='y', zlabel='z', heatlabel='heat'):
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

def plot_learning_curve(train_sizes, train_mean, test_mean, train_stdev, test_stdev, score_name, learning_curve_type, savepath='data_learning_curve'):

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
    ax.legend([h1, h2], ['train score', 'test score'], loc='center right', fontsize=12)
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
    plot_learning_curve_convergence(train_sizes, test_mean, score_name, learning_curve_type, savepath)

def plot_learning_curve_convergence(train_sizes, test_mean, score_name, learning_curve_type, savepath):
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

    if learning_curve_type == 'feature_learning_curve':
        # First, set number optimal features to all features in case stopping criteria not met
        n_features_optimal = len(test_mean)
        # Determine what the optimal number of features are to use
        stopping_criteria = 0.03
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

def trim_array(df_list):
    # TODO: a better way to handle arrays with very different shapes? Otherwise average only uses # of points of smallest array
    # Need to make arrays all same shapes if they aren't
    sizes = [df.shape[0] for df in df_list]
    size_min = min(sizes)
    df_list_ = list()
    for i, df in enumerate(df_list):
        if df.shape[0] > size_min:
            while df.shape[0] > size_min:
                df = np.delete(df, -1)
        df_list_.append(df)
    df_list = df_list_
    return df_list

def rounder(delta):
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

def stat_to_string(name, value):
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
    Print stats onto the image
    Goes off screen if they are too long or too many in number
    """

    stat_str = '\n'.join(stat_to_string(name, value)
                           for name,value in stats.items())

    fig.text(x_align, y_align, stat_str,
             verticalalignment='top', wrap=True, fontdict=font_dict, fontproperties=FontProperties(size=fontsize))

def make_fig_ax(aspect_ratio=0.5, x_align=0.65, left=0.10):
    """
    Using Object Oriented interface from
    https://matplotlib.org/gallery/api/agg_oo_sgskip.html
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
    Using Object Oriented interface from
    https://matplotlib.org/gallery/api/agg_oo_sgskip.html
    """
    # Set image aspect ratio:
    w, h = figaspect(aspect_ratio)
    fig = Figure(figsize=(w,h))
    FigureCanvas(fig)
    ax = fig.add_subplot(111, aspect=aspect)

    return fig, ax

def make_axis_same(ax, max1, min1):
    # fix up dem axis
    if max1 - min1 > 5:
        step = (int(max1) - int(min1)) // 3
        ticks = range(int(min1), int(max1)+step, step)
    else:
        ticks = np.linspace(min1, max1, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

def nice_mean(ls):
    " Returns NaN for empty list "
    if len(ls) > 0:
        return np.mean(ls)
    return np.nan

def nice_std(ls):
    " Returns NaN for empty list "
    if len(ls) > 0:
        return np.std(ls)
    return np.nan

def round_down(num, divisor):
    return num - (num%divisor)

def round_up(num, divisor):
    return float(math.ceil(num / divisor)) * divisor

def get_divisor(high, low):
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


# Credit: https://www.linkedin.com/pulse/ask-recursion-during-coding-interviews-identify-good-talent-veteanu/
# not used yet, should be used to refactor some of the min and max bits
def recursive_max(array):
    return max(
        recursive_max(e) if isinstance(e, Iterable) else e
        for e in array
    )

def recursive_min(array):
    return min(
        recursive_min(e) if isinstance(e, Iterable) else e
        for e in array
    )

def recursive_max_and_min(array):
    return recursive_max(array), recursive_min(array)

def _set_tick_labels(ax, maxx, minn):
    " sets x and y to the same range "
    _set_tick_labels_different(ax, maxx, minn, maxx, minn) # I love it when this happens

def _set_tick_labels_different(ax, max_tick_x, min_tick_x, max_tick_y, min_tick_y):
    " Use this when X and y are over completely diffent ranges. "

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

    ax.set_xticklabels(labels=ticklabels_x, fontsize=14)
    ax.set_yticklabels(labels=ticklabels_y, fontsize=14)

def _clean_tick_labels(tickvals, delta):
    tickvals_clean = list()
    if delta >= 100:
        for i, val in enumerate(tickvals):
            if i <= len(tickvals)-1:
                if tickvals[i]-tickvals[i-1] >= 100:
                    tickvals_clean.append(val)
    else:
        tickvals_clean = tickvals
    return tickvals_clean