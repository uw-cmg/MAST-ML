""" most of these plots take in (data, other_data, ..., savepath, stats, title='Default Title')
Where the data args are numpy arrays, savepath is a string, 
and stats is a list of pairs or tripletes of (name, value[, error])

"""

import os.path
import itertools

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure, figaspect
from matplotlib.ticker import MaxNLocator


# set all font to bigger
font = {'size'   : 18}
matplotlib.rc('font', **font)

# turn on autolayout (why is it not default?)
matplotlib.rc('figure', autolayout=True)

# HEADERENDER don't delete this line, it's used by ipynb maker

from .ipynb_maker import ipynb_maker

def parse_stat(data):
    """ takes in a pair or tripple of stats and returns a string for the plot 
    Also singleton tuples for Titles or something"""
    if len(data) == 1:
        return data[0]
    if len(data) == 2:
        name, value = data
        return f'{name}: {value}'
    if len(data) == 3:
        name, value, err = data
        return (f'{name}: \n\t{value}' + r'$\pm$' + f'{err}')

    raise Exception(f'{data} must be a pair or tripplet')

def plot_stats(fig, stats):
    """ print stats onto the image. Goes off screen if they are too long or too many in number """

    stat_str = '\n\n'.join(parse_stat(data) for data in stats)

    fig.text(0.62, 0.98, stat_str,
            verticalalignment='top', wrap=True# transform=ax.transAxes)
            )


def make_fig_ax(aspect='equal'):
    """ using OO interface from https://matplotlib.org/gallery/api/agg_oo_sgskip.html"""
    # set image aspect ratio. Needs to be wide enough or plot will shrink really skinny
    w, h = figaspect(0.6)
    fig = Figure(figsize=(w,h))
    FigureCanvas(fig)

    # these two lines are where the magic happens, trapping the figure on the left side
    # so we can make print text beside it
    gs = plt.GridSpec(1, 5)
    ax = fig.add_subplot(gs[0, 0:3], aspect=aspect)

    return fig, ax

@ipynb_maker
def plot_confusion_matrix(y_true, y_pred, savepath, stats, normalize=False, title='Confusion matrix',
        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    # calculate confusion matrix and lables in correct order
    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(list(set(y_true).intersection(set(y_pred))))

    fig, ax = make_fig_ax()

    ax.set_title(title)

    # create the colorbar, not really needed but everyones got 'em
    mappable = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(mappable)

    # set x and y ticks to labels
    tick_marks = range(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation='vertical', fontsize=18)

    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, rotation='vertical', fontsize=18)

    # draw number in the boxes
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    # plots the stats
    plot_stats(fig, stats)

    #plt.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.savefig(savepath)

@ipynb_maker
def plot_predicted_vs_true(y_true, y_pred, savepath, stats, title='predicted vs true'):

    fig, ax = make_fig_ax()

    ax.set_title(title)

    # do the actual plotting
    ax.scatter(y_true, y_pred, edgecolors=(0, 0, 0))

    # make diagonal line from absolute min to absolute max of any data point
    maxx = max(y_true.max(), y_pred.max())
    minn = min(y_true.min(), y_pred.min())
    ax.plot([minn, maxx], [minn, maxx], 'k--', lw=4)

    # set axis labels
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')

    plot_stats(fig, stats)

    fig.savefig(savepath)

@ipynb_maker
def plot_best_worst(y_true, y_pred_best, y_pred_worst, savepath, stats, title='Best Worst Overlay'):

    fig, ax = make_fig_ax()

    ax.set_title(title)

    # make diagonal line from absolute min to absolute max of any data point
    maxx = max(y_true.max(), y_pred_best.max(), y_pred_worst.max())
    minn = min(y_true.min(), y_pred_best.min(), y_pred_worst.min())
    ax.plot([minn, maxx], [minn, maxx], 'k--', lw=4, zorder=1)

    # do the actual plotting
    ax.scatter(y_true, y_pred_best, c='red', alpha=0.7, s=80, label='best', edgecolor='darkred', zorder=2)
    ax.scatter(y_true, y_pred_worst, c='blue', alpha=0.7, label='worst', edgecolor='darkblue', zorder=3)
    ax.legend()


    # set axis labels
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')

    plot_stats(fig, stats)

    fig.savefig(savepath)

@ipynb_maker
def plot_residuals_histogram(y_true, y_pred, savepath, stats, title='residuals histogram'):

    fig, ax = make_fig_ax(aspect='auto')

    ax.set_title(title)
    # do the actual plotting
    residuals = y_true - y_pred
    ax.hist(residuals, bins=30)

    # normal text stuff
    ax.set_xlabel('residual')
    ax.set_ylabel('frequency')

    # make y axis ints, because it is discrete
    #ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # shrink those margins
    fig.tight_layout()

    plot_stats(fig, stats)

    fig.savefig(savepath)
