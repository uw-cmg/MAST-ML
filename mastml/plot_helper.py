"""
A collection of functions which make plots (png files) using matplotlib.
Most of these plots take in (data, other_data, ..., savepath, stats, title).
Where the data args are numpy arrays, savepath is a string, and stats is an
ordered dictionary which maps names to either values, or mean-stdev pairs.

A plot can also take an "outdir" instead of a savepath. If this is the case,
it must return a list of filenames where it saved the figures.
"""
import math
import itertools
import warnings
from os.path import join
from collections import OrderedDict

# Ignore the harmless warning about the gelsd driver on mac.
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import RFECV # for feature learning curve

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure, figaspect
from matplotlib.ticker import MaxNLocator # TODO: used?
from matplotlib.animation import FuncAnimation
from matplotlib.font_manager import FontProperties

from .utils import RFECV_train_test

matplotlib.rc('font', size=18, family='sans-serif') # set all font to bigger
matplotlib.rc('figure', autolayout=True) # turn on autolayout

# HEADERENDER don't delete this line, it's used by ipynb maker

from .ipynb_maker import ipynb_maker # TODO: fix cyclic import
from .metrics import nice_names

def make_main_plots(run, path, is_classification):
    y_train_true, y_train_pred, y_test_true = \
        run['y_train_true'], run['y_train_pred'], run['y_test_true']
    y_test_pred, train_metrics, test_metrics = \
        run['y_test_pred'], run['train_metrics'], run['test_metrics']

    if is_classification:
        title = 'train_confusion_matrix'
        plot_confusion_matrix(y_train_true, y_train_pred,
                              join(path, title+'.png'), train_metrics,
                              title=title)
        title = 'test_confusion_matrix'
        plot_confusion_matrix(y_test_true, y_test_pred,
                              join(path, title+'.png'), test_metrics,
                              title=title)

    else: # is_regression
        plot_predicted_vs_true((y_train_true, y_train_pred, train_metrics),
                          (y_test_true,  y_test_pred,  test_metrics), path)

        title = 'train_residuals_histogram'
        plot_residuals_histogram(y_train_true, y_train_pred,
                                 join(path, title+'.png'), train_metrics,
                                 title=title)
        title = 'test_residuals_histogram'
        plot_residuals_histogram(y_test_true,  y_test_pred,
                                 join(path, title+'.png'), test_metrics,
                                 title=title)

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

    fig, ax = make_fig_ax()

    ax.set_title(title)

    # create the colorbar, not really needed but everyones got 'em
    mappable = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #fig.colorbar(mappable)

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

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.savefig(savepath, dpi=250)

@ipynb_maker
def plot_predicted_vs_true(train_triple, test_triple, outdir):
    filenames = list()
    y_train_true, y_train_pred, train_metrics = train_triple
    y_test_true, y_test_pred, test_metrics = test_triple

    # make diagonal line from absolute min to absolute max of any data point
    max1 = max(y_train_true.max(), y_train_pred.max(),
               y_test_true.max(), y_test_pred.max())
    min1 = min(y_train_true.min(), y_train_pred.min(),
               y_test_true.min(), y_test_pred.min())

    for y_true, y_pred, stats, title_addon in \
            (train_triple+('train',), test_triple+('test',)):

        # Set image aspect ratio:
        fig, ax = make_fig_ax()

        #ax.set_title('Predicted vs. True ' + title_addon)
        ax.plot([min1, max1], [min1, max1], 'k--', lw=2, zorder=1)

        # set tick labels
        maxx = round(max((max(y_pred), max(y_true))))
        minn = round(min((min(y_pred), min(y_true))))
        divisor = get_divisor(maxx, minn)
        max_tick = round_up(maxx, divisor)
        min_tick = round_down(minn, divisor=divisor)
        tickvals = np.linspace(min_tick, max_tick, num=5)
        tickvals = [int(val) for val in tickvals]
        ax.set_xticks(ticks=tickvals)
        ax.set_yticks(ticks=tickvals)
        ticklabels = [str(tick) for tick in tickvals]
        ax.set_xticklabels(labels=ticklabels)
        ax.set_yticklabels(labels=ticklabels)

        make_axis_same(ax, max1, min1)

        # do the actual plotting
        ax.scatter(y_true, y_pred, color='blue', edgecolors='black', zorder=2, alpha=0.7)
        ax.legend(loc='lower right', bbox_to_anchor=(1.25, 0), fontsize=12, frameon=False)

        # set axis labels
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')

        plot_stats(fig, stats, x_align=16/24, y_align=0.90)

        filename = 'predicted_vs_true_'+ title_addon + '.png'
        filenames.append(filename)
        fig.savefig(join(outdir, filename), dpi=250, bbox_inches='tight')

    return filenames

def get_histogram_bins(y_df):
    bin_dividers = np.linspace(y_df.shape[0], round(0.05*y_df.shape[0]), y_df.shape[0])
    bin_list = list()
    try:
        for divider in bin_dividers:
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

@ipynb_maker
def plot_residuals_histogram(y_true, y_pred, savepath,
                             stats, title='residuals histogram', xlabel='residuals'):

    # Set image aspect ratio:
    fig, ax = make_fig_ax()

    #ax.set_title(title)
    # do the actual plotting
    residuals = y_true - y_pred
    #Get num_bins using smarter method
    num_bins = get_histogram_bins(y_df=residuals)
    ax.hist(residuals, bins=num_bins, color='b', edgecolor='k')
    ax.legend(loc='lower right', bbox_to_anchor=(1.25, 0), fontsize=12, frameon=False)

    # normal text stuff
    ax.set_xlabel(xlabel)
    ax.set_ylabel('frequency')

    # make y axis ints, because it is discrete
    #ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # shrink those margins
    fig.tight_layout()

    plot_stats(fig, stats, x_align=16/24, y_align=0.90)

    fig.savefig(savepath, dpi=250, bbox_inches='tight')

@ipynb_maker
def plot_target_histogram(y_df, savepath, title='target histogram', xlabel='y values'):

    # Set image aspect ratio:
    fig, ax = make_fig_ax(aspect_ratio=0.5)

    #ax.set_title(title)

    #Get num_bins using smarter method
    num_bins = get_histogram_bins(y_df=y_df)

    # do the actual plotting
    ax.hist(y_df, bins=num_bins, color='b', edgecolor='k')#, histtype='stepfilled')
    #ax.legend(loc='lower right', bbox_to_anchor=(1.25, 0), fontsize=12, frameon=False)

    # normal text stuff
    ax.set_xlabel(xlabel)
    ax.set_ylabel('frequency')

    # make y axis ints, because it is discrete
    #ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # shrink those margins
    fig.tight_layout()

    plot_stats(fig, dict(y_df.describe()), x_align=0.70, y_align=0.90, fontsize=14)
    # Save input data stats to csv
    savepath_parse = savepath.split('target_histogram.png')[0]
    y_df.describe().to_csv(savepath_parse+'/''input_data_statistics.csv')

    fig.savefig(savepath, dpi=250, bbox_inches='tight')

def plot_scatter(x, y, savepath, groups=None, xlabel='x', ylabel='y'):
    # Set image aspect ratio:
    fig, ax = make_fig_ax()

    # set tick labels

    maxx = round(max((max(x), max(y))))
    minn = round(min((min(x), min(y))))
    divisor = get_divisor(maxx, minn)
    max_tick = round_up(maxx, divisor)
    min_tick = round_down(minn, divisor)
    tickvals = np.linspace(min_tick, max_tick, num=5)
    tickvals = [int(val) for val in tickvals]
    ax.set_xticks(ticks=tickvals)
    ax.set_yticks(ticks=tickvals)
    ticklabels = [str(tick) for tick in tickvals]
    ax.set_xticklabels(labels=ticklabels)
    ax.set_yticklabels(labels=ticklabels)

    if groups is None:
        ax.scatter(x, y, c='b', edgecolor='black',  zorder=2, s=80, alpha=0.7)
    else:
        for group in np.unique(groups):
            mask = groups == group
            ax.scatter(x[mask], y[mask], label=group)
        ax.legend(loc='lower left', bbox_to_anchor=(1, -.2))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(savepath, dpi=250, bbox_inches='tight')

@ipynb_maker
def plot_best_worst_split(best_run, worst_run, savepath,
                          title='Best Worst Overlay'):
    # Set image aspect ratio:
    fig, ax = make_fig_ax()

    # make diagonal line from absolute min to absolute max of any data point
    all_y = [best_run['y_test_true'], best_run['y_test_pred'],
             worst_run['y_test_true'], worst_run['y_test_pred']]
    maxx = max(y.max() for y in all_y)
    minn = min(y.min() for y in all_y)
    ax.plot([minn, maxx], [minn, maxx], 'k--', lw=2, zorder=1)

    # do the actual plotting
    ax.scatter(best_run['y_test_true'],  best_run['y_test_pred'],  c='red',
               alpha=0.7, label='best',  edgecolor='darkred',  zorder=2, s=80)
    ax.scatter(worst_run['y_test_true'], worst_run['y_test_pred'], c='blue',
               alpha=0.7, label='worst', edgecolor='darkblue', zorder=3)
    ax.legend(loc='lower right', bbox_to_anchor=(1.25, 0), fontsize=12)

    # set tick labels
    maxx = round(maxx)
    minn = round(minn)
    divisor = get_divisor(maxx, minn)
    max_tick = round_up(maxx, divisor)
    min_tick = round_down(minn, divisor)
    tickvals = np.linspace(min_tick, max_tick, num=5)
    tickvals = [int(val) for val in tickvals]
    ax.set_xticks(ticks=tickvals)
    ax.set_yticks(ticks=tickvals)
    ticklabels = [str(tick) for tick in tickvals]
    ax.set_xticklabels(labels=ticklabels)
    ax.set_yticklabels(labels=ticklabels)

    # set axis labels
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')

    #font_dict = {'size'   : 10, 'family' : 'sans-serif'}

    # Duplicate the stats dicts with an additional label
    best_stats = OrderedDict([('Best Run', None)])
    best_stats.update(best_run['test_metrics'])
    worst_stats = OrderedDict([('worst Run', None)])
    worst_stats.update(worst_run['test_metrics'])

    plot_stats(fig, best_stats, x_align=16/24, y_align=0.90)
    plot_stats(fig, worst_stats, x_align=16/24, y_align=0.60)

    #fig.tight_layout()
    fig.savefig(savepath, dpi=250, bbox_inches='tight')

@ipynb_maker
def plot_best_worst_per_point(y_true, y_pred_list, savepath, metrics_dict,
                              avg_stats, title='best worst per point'):
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

    fig, ax = make_fig_ax()

    # gather max and min
    all_vals = [val for val in worsts+bests if val is not None]
    max1 = max(all_vals)
    min1 = min(all_vals)

    make_axis_same(ax, max1, min1)

    # draw dashed horizontal line
    ax.plot([min1, max1], [min1, max1], 'k--', lw=2, zorder=1)

    # set axis labels
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')

    # set tick labels
    maxx = round(max((max(bests), max(worsts), max(new_y_true))))
    minn = round(min((min(bests), min(worsts), min(new_y_true))))
    divisor = get_divisor(maxx, minn)
    max_tick = round_up(maxx, divisor)
    min_tick = round_down(minn, divisor)
    tickvals = np.linspace(min_tick, max_tick, num=5)
    tickvals = [int(val) for val in tickvals]
    ax.set_xticks(ticks=tickvals)
    ax.set_yticks(ticks=tickvals)
    ticklabels = [str(tick) for tick in tickvals]
    ax.set_xticklabels(labels=ticklabels)
    ax.set_yticklabels(labels=ticklabels)

    ax.scatter(new_y_true, bests,  c='red',  alpha=0.7, label='best',
               edgecolor='darkred',  zorder=2, s=80)
    ax.scatter(new_y_true, worsts, c='blue', alpha=0.7, label='worst',
               edgecolor='darkblue', zorder=3)
    ax.legend(loc='lower right', bbox_to_anchor=(1.25, -0.1), fontsize=12)

    plot_stats(fig, avg_stats, x_align=15.5/24, y_align=0.57, fontsize=10)
    plot_stats(fig, worst_stats, x_align=15.5/24, y_align=0.77, fontsize=10)
    plot_stats(fig, best_stats, x_align=15.5/24, y_align=0.95, fontsize=10)
    fig.savefig(savepath, dpi=250, bbox_inches='tight')

@ipynb_maker
def plot_predicted_vs_true_bars(y_true, y_pred_list, avg_stats,
                                savepath, title='best worst with bars'):
    " EVERYTHING MUST BE ARRAYS DONT GIVE ME DEM DF "
    means = [nice_mean(y_pred) for y_pred in y_pred_list]
    standard_error_means = [nice_std(y_pred)/np.sqrt(len(y_pred))
                            for y_pred in y_pred_list]
    standard_errors = [nice_std(y_pred) for y_pred in y_pred_list]
    fig, ax = make_fig_ax()

    # gather max and min
    max1 = max(np.nanmax(y_true), np.nanmax(means))
    min1 = min(np.nanmin(y_true), np.nanmin(means))

    make_axis_same(ax, max1, min1)

    # draw dashed horizontal line
    ax.plot([min1, max1], [min1, max1], 'k--', lw=2, zorder=1)

    # set axis labels
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')

    # set tick labels
    maxx = round(max((max(means), max(y_true))))
    minn = round(min((min(means), min(y_true))))
    divisor = get_divisor(maxx, minn)
    max_tick = round_up(maxx, divisor)
    min_tick = round_down(minn, divisor)
    tickvals = np.linspace(min_tick, max_tick, num=5)
    tickvals = [int(val) for val in tickvals]
    ax.set_xticks(ticks=tickvals)
    ax.set_yticks(ticks=tickvals)
    ticklabels = [str(tick) for tick in tickvals]
    ax.set_xticklabels(labels=ticklabels)
    ax.set_yticklabels(labels=ticklabels)

    ax.errorbar(y_true, means, yerr=standard_errors, fmt='o', markerfacecolor='blue', markeredgecolor='black', alpha=0.7, capsize=3)
    ax.legend(loc='lower right', bbox_to_anchor=(1.25, 0), fontsize=12, frameon=False)

    plot_stats(fig, avg_stats, x_align=16/24, y_align=0.90)
    fig.savefig(savepath, dpi=250, bbox_inches='tight')

@ipynb_maker
def plot_violin(y_true, y_pred_list, savepath, title='best worst with bars'):
    # Make new data without empty prediction lists:
    y_true_new = []
    y_pred_list_new = []
    means = []
    for i in range(y_true.shape[0]):
        if len(y_pred_list[i]) > 0:
            y_true_new.append(y_true[i])
            y_pred_list_new.append(y_pred_list[i])
            means.append(np.nanmean(y_pred_list[i]))

    fig, ax = make_fig_ax(aspect='auto')

    # gather max and min
    max1 = max(np.nanmax(y_true_new), np.nanmax(means))
    min1 = min(np.nanmin(y_true_new), np.nanmin(means))

    make_axis_same(ax, max1, min1)

    # draw dashed horizontal line
    ax.plot([min1, max1], [min1, max1], 'k--', lw=4, zorder=1)

    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.violinplot(y_pred_list_new, y_true_new)

    plot_stats(fig, dict())
    fig.savefig(savepath, dpi=250)

def plot_1d_heatmap(xs, heats, savepath, xlabel='x', heatlabel='heats'):
    fig, ax = make_fig_ax(aspect='auto')
    ax.bar(xs, heats)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(heatlabel)
    fig.savefig(savepath, dpi=250)


def plot_2d_heatmap(xs, ys, heats, savepath,
                    xlabel='x', ylabel='y', heatlabel='heat'):
    fig, ax = make_fig_ax(aspect='auto')
    scat = ax.scatter(xs, ys, c=heats) # marker='o', lw=0, s=20, cmap=cm.plasma
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cb = fig.colorbar(scat)
    cb.set_label(heatlabel)
    fig.savefig(savepath, dpi=250)

def plot_3d_heatmap(xs, ys, zs, heats, savepath,
                    xlabel='x', ylabel='y', zlabel='z', heatlabel='heat'):
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

    fig.savefig(savepath, dpi=250)

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return [fig]
    anim = FuncAnimation(fig, animate, frames=range(0,90,5), blit=True)
    #anim.save(savepath+'.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
    anim.save(savepath+'.gif', fps=5, dpi=80, writer='imagemagick')

def plot_sample_learning_curve(model, X, y, scoring, cv=2, savepath='sample_learning_curve.png'):
    train_sizes, train_scores, valid_scores = learning_curve(model, X, y, scoring=scoring, cv=cv)
    fig, ax = make_fig_ax()
    h1 = ax.plot(train_sizes, train_scores, '-o', c='blue')[0]
    h2 = ax.plot(train_sizes, valid_scores, '-o', c='red')[0]
    ax.legend([h1, h2], ['train', 'test'], loc='lower left', bbox_to_anchor=(1, -.2))
    ax.set_xlabel('number of training samples')
    scoring_name = scoring._score_func.__name__
    ax.set_ylabel(scoring_name + ' score')
    fig.savefig(savepath, dpi=250)

def plot_feature_learning_curve(model, X, y, scoring=None, savepath='feature_learning_curve.png'):
    #RFECV.transform = RFECV.old_transform # damn you
    print(dir(RFECV))
    rfe = RFECV_train_test(model, scoring=scoring)
    rfe = rfe.fit(X, y)
    '''
    >>> oops
    (Pdb) rfe.support_
    >>> array([ True,  True,  True,  True,  True, False, False,  True,  True, False], dtype=bool)
    (Pdb) rfe.ranking_
    >>> array([1, 1, 1, 1, 1, 4, 2, 1, 1, 3])
    (Pdb) rfe.grid_scores_
    >>> array([ 0.11704555,  0.02829977,  0.09502728,  0.11660331,  0.11178014,
    >>>        0.09861614,  0.14987966,  0.14267309,  0.13662161,  0.14129173])
    '''
    fig, ax = make_fig_ax_square(aspect='auto')
    ax.set_xlabel("Number of features selected")
    ax.set_ylabel("CV score")
    ax.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_, label='test')
    ax.plot(range(1, len(rfe.grid_train_scores_) + 1), rfe.grid_train_scores_, label='train')
    ax.legend()
    fig.savefig(savepath, dpi=250)

### Helpers:

def parse_stat(name,value):
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

    stat_str = '\n'.join(parse_stat(name, value)
                           for name,value in stats.items())

    fig.text(x_align, y_align, stat_str,
             verticalalignment='top', wrap=True, fontdict=font_dict, fontproperties=FontProperties(size=fontsize))

def make_fig_ax(aspect_ratio=0.5):
    """
    Using Object Oriented interface from
    https://matplotlib.org/gallery/api/agg_oo_sgskip.html
    """
    # Set image aspect ratio:
    w, h = figaspect(aspect_ratio)
    fig = Figure(figsize=(w,h))
    FigureCanvas(fig)

    # these two lines are where the magic happens, trapping the figure on the
    # left side so we can make print text beside it
    gs = plt.GridSpec(1, 3)
    #ax = fig.add_subplot(gs[0:, 0:3], aspect=aspect)
    ax = fig.add_subplot(gs[0:, 0:2])
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
        ticks = np.linspace(min1, max1, 4)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

def nice_mean(ls):
    """
    Returns NaN for empty list
    """
    if len(ls) > 0:
        return np.mean(ls)
    return np.nan

def nice_std(ls):
    """ Explicity returns `None` for empty list, without raising a warning. """
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
    if delta > 100:
        divisor = 10
    if delta < 100:
        if delta > 10:
            divisor = 1
        if delta < 10:
            if delta > 1:
                divisor = 0.1
            if delta < 1:
                divisor = 0.05
    return divisor