"""
A collection of functions which make plots (png files) using matplotlib.
Most of these plots take in (data, other_data, ..., savepath, stats, title).
Where the data args are numpy arrays, savepath is a string, and stats is an
ordered dictionary which maps names to either values, or mean-stdev pairs.

A plot can also take an "outdir" instead of a savepath. If this is the case, 
it must return a list of filenames where it saved the figures.
"""

from os.path import join
import itertools

import numpy as np # TODO: used?

from sklearn.metrics import confusion_matrix

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure, figaspect
from matplotlib.ticker import MaxNLocator # TODO: used?
from matplotlib.animation import FuncAnimation

# set all font to bigger
font = {'size'   : 18,
        'family' : 'sans-serif'}
matplotlib.rc('font', **font)

# turn on autolayout (why is it not default?)
matplotlib.rc('figure', autolayout=True)

# HEADERENDER don't delete this line, it's used by ipynb maker

from .ipynb_maker import ipynb_maker # TODO: fix cyclic import

# maybe TODO: have all plot makers start with `plot`

def make_plots(run, path, is_classification):
    y_train_true, y_train_pred, y_test_true,  y_test_pred, train_metrics, test_metrics = \
        run['y_train_true'], run['y_train_pred'], run['y_test_true'],  run['y_test_pred'], run['train_metrics'], run['test_metrics']

    if is_classification:
        title = 'train_confusion_matrix'
        plot_confusion_matrix(y_train_true, y_train_pred, join(path, title+'.png'), train_metrics, title=title)
        title = 'test_confusion_matrix'
        plot_confusion_matrix(y_test_true,  y_test_pred, join(path, title+'.png'), test_metrics, title=title)

    else: # is_regression
        predicted_vs_true((y_train_true, y_train_pred, train_metrics),
                          (y_test_true,  y_test_pred,  test_metrics), path)

        title = 'train_residuals_histogram'
        plot_residuals_histogram(y_train_true, y_train_pred, join(path, title+'.png'), train_metrics, title=title)
        title = 'test_residuals_histogram'
        plot_residuals_histogram(y_test_true,  y_test_pred, join(path, title+'.png'), test_metrics, title=title)

    with open(join(path, 'stats.txt'), 'w') as f:
        f.write("TRAIN:\n")
        for name,score in train_metrics.items():
            f.write(f"{name}: {score}\n")
        f.write("TEST:\n")
        for name,score in test_metrics.items():
            f.write(f"{name}: {score}\n")


nice_names = {
    'accuracy_score': 'Acc',
    'confusion_matrix': 'Confusion Matrix',
    'f1_score': 'f1',
    'fbeta_score': 'fbeta score',
    'hamming_loss': 'Hamming Loss',
    'jaccard_similarity_score': 'Jaccard Similarity Score',
    'log_loss': 'Log Loss',
    'matthews_corrcoef': 'Matthews',
    'precision_recall_fscore_support': 'pr fscore',
    'precision_score': 'p',
    'recall_score': 'r',
    'roc_auc_score': 'roc auc',
    'roc_curve': 'roc',
    'zero_one_loss': '10ss',
    'explained_variance_score': 'explained variance',
    'mean_absolute_error': 'MAE',
    'mean_squared_error': 'MSE',
    'root_mean_squared_error': 'RMSE',
    'mean_squared_log_error': 'MSLE',
    'median_absolute_error': 'MeAE',
    'r2_score': '$r^2$',
    'r2_score_noint': '$r^2$ noint',
}


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


def plot_stats(fig, stats):
    #print(stats)
    """ print stats onto the image. Goes off screen if they are too long or too many in number """

    stat_str = '\n\n'.join(parse_stat(name, value) for name,value in stats.items())

    fig.text(0.69, 0.98, stat_str,
             verticalalignment='top', wrap=True)


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
def plot_confusion_matrix(y_true, y_pred, savepath, stats, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
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

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.savefig(savepath)


@ipynb_maker
def predicted_vs_true(train_triple, test_triple, outdir):
    filenames = list()
    y_train_true, y_train_pred, train_metrics = train_triple
    y_test_true, y_test_pred, test_metrics = test_triple

    # make diagonal line from absolute min to absolute max of any data point
    max1 = max(y_train_true.max(), y_train_pred.max(), y_test_true.max(), y_test_pred.max())
    min1 = min(y_train_true.min(), y_train_pred.min(), y_test_true.min(), y_test_pred.min())

    for y_true, y_pred, stats, title_addon in (train_triple+('train',), test_triple+('test',)):
        fig, ax = make_fig_ax()

        ax.set_title('Predicted vs. True ' + title_addon)
        ax.plot([min1, max1], [min1, max1], 'k--', lw=4, zorder=1)

        make_axis_same(ax, max1, min1)

        # do the actual plotting
        ax.scatter(y_true, y_pred, edgecolors=(0, 0, 0), zorder=2)

        # set axis labels
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')

        plot_stats(fig, stats)

        filename = 'predicted_vs_true_'+ title_addon + '.png'
        filenames.append(filename)
        fig.savefig(join(outdir, filename))

    return filenames



def make_axis_same(ax, max1, min1):
    # fix up dem axis
    if max1 - min1 > 5:
        step = (int(max1) - int(min1)) // 3
        ticks = range(int(min1), int(max1)+step, step)
    else:
        ticks = np.linspace(min1, max1, 4)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


@ipynb_maker
def plot_best_worst(best_run, worst_run, savepath, stats, title='Best Worst Overlay'):
    fig, ax = make_fig_ax()
    #ax.set_title(title)
    # make diagonal line from absolute min to absolute max of any data point
    all_y = [best_run['y_test_true'], best_run['y_test_pred'], worst_run['y_test_true'], worst_run['y_test_pred']]
    maxx = max(y.max() for y in all_y)
    minn = min(y.min() for y in all_y)
    ax.plot([minn, maxx], [minn, maxx], 'k--', lw=4, zorder=1)

    # do the actual plotting
    ax.scatter(best_run['y_test_true'],  best_run['y_test_pred'],  c='red',  alpha=0.7, label='best',  edgecolor='darkred',  zorder=2, s=80)
    ax.scatter(worst_run['y_test_true'], worst_run['y_test_pred'], c='blue', alpha=0.7, label='worst', edgecolor='darkblue', zorder=3)
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


@ipynb_maker
def target_histogram(y_df, savepath, title='target histogram'):

    fig, ax = make_fig_ax(aspect='auto')

    ax.set_title(title)
    # do the actual plotting
    ax.hist(y_df)#, histtype='stepfilled')

    # normal text stuff
    ax.set_xlabel('y values')
    ax.set_ylabel('frequency')

    # make y axis ints, because it is discrete
    #ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # shrink those margins
    fig.tight_layout()

    plot_stats(fig, dict(y_df.describe()))

    fig.savefig(savepath)


@ipynb_maker
def predicted_vs_true_bars(y_true, y_pred_list, savepath, title='best worst with bars'):
    means = [np.mean(y_pred) for y_pred in y_pred_list]
    standard_error_means = [np.std(y_pred)/np.sqrt(len(y_pred)) for y_pred in y_pred_list]
    fig, ax = make_fig_ax(aspect='auto')

    # gather max and min
    max1 = max(np.max(y_true), np.max(means))
    min1 = min(np.min(y_true), np.min(means))

    make_axis_same(ax, max1, min1)

    # draw dashed horizontal line
    ax.plot([min1, max1], [min1, max1], 'k--', lw=4, zorder=1)

    # set axis labels
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')

    ax.errorbar(y_true, means, yerr=standard_error_means, fmt='o', capsize=3)

    plot_stats(fig, dict())
    fig.savefig(savepath)


@ipynb_maker
def violin(y_true, y_pred_list, savepath, title='best worst with bars'):
    means = [np.mean(y_pred) for y_pred in y_pred_list]
    standard_error_means = [np.std(y_pred)/np.sqrt(len(y_pred)) for y_pred in y_pred_list]
    fig, ax = make_fig_ax(aspect='auto')

    # gather max and min
    max1 = max(np.max(y_true), np.max(means))
    min1 = min(np.min(y_true), np.min(means))

    make_axis_same(ax, max1, min1)

    # draw dashed horizontal line
    ax.plot([min1, max1], [min1, max1], 'k--', lw=4, zorder=1)

    # set axis labels
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')

    ax.violinplot(y_pred_list, y_true)

    plot_stats(fig, dict())
    fig.savefig(savepath)


@ipynb_maker
def best_worst_per_point(y_true, y_pred_list, savepath, title='best worst per point'):
    worsts = [max(yp, key=lambda y: abs(yt-y)) if yp else None for yt, yp in zip(y_true,y_pred_list)]
    bests  = [min(yp, key=lambda y: abs(yt-y)) if yp else None for yt, yp in zip(y_true,y_pred_list)]

    fig, ax = make_fig_ax(aspect='auto')

    # gather max and min
    all_vals = [val for val in worsts+bests if val is not None]
    max1 = max(all_vals)
    min1 = min(all_vals)

    make_axis_same(ax, max1, min1)

    # draw dashed horizontal line
    ax.plot([min1, max1], [min1, max1], 'k--', lw=4, zorder=1)

    # set axis labels
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')

    ax.scatter(y_true, bests,  c='red',  alpha=0.7, label='best',  edgecolor='darkred',  zorder=2, s=80)
    ax.scatter(y_true, worsts, c='blue', alpha=0.7, label='worst', edgecolor='darkblue', zorder=3)
    ax.legend()


    plot_stats(fig, dict())
    fig.savefig(savepath)


def plot_1d_heatmap(xs, heats, savepath, xlabel='x', heatlabel='heats'):
    fig, ax = make_fig_ax(aspect='auto')
    ax.bar(xs, heats)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(heatlabel)
    fig.savefig(savepath)


def plot_2d_heatmap(xs, ys, heats, savepath, xlabel='x', ylabel='y', heatlabel='heat'):
    fig, ax = make_fig_ax(aspect='auto')
    scat = ax.scatter(xs, ys, c=heats) # marker='o', lw=0, s=20, cmap=cm.plasma
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cb = fig.colorbar(scat)
    cb.set_label(heatlabel)
    fig.savefig(savepath)
    

def plot_3d_heatmap(xs, ys, zs, heats, savepath, xlabel='x', ylabel='y', zlabel='z', heatlabel='heat'):
    from mpl_toolkits.mplot3d import Axes3D # this import has side effects, needed for 3d plots
    w, h = figaspect(0.6) # set image aspect ratio. Needs to be wide enough or plot will shrink really skinny
    fig = Figure(figsize=(w,h))
    FigureCanvas(fig) # modifies fig in place
    ax = fig.add_subplot(111, projection='3d')

    scat = ax.scatter(xs, ys, zs, c=heats) # marker='o', lw=0, s=20, cmap=cm.plasma
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    cb = fig.colorbar(scat)
    cb.set_label(heatlabel)

    fig.savefig(savepath)

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return [fig]
    anim = FuncAnimation(fig, animate, frames=range(0,90,5), blit=True)
    #anim.save(savepath+'.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
    anim.save(savepath+'.gif', fps=5, dpi=80, writer='imagemagick')
