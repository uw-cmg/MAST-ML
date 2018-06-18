import itertools
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure, figaspect
from matplotlib.ticker import MaxNLocator

def plot_confusion_matrix(y_true, y_pred, filename, normalize=False, title='Confusion matrix',
        stats=dict(), cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(list(set(y_true).intersection(set(y_pred))))

    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    mappable = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(mappable)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes)
    ax.set_yticks(tick_marks, classes)

    # force ticks to integer: https://stackoverflow.com/a/34880501
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.savefig(filename)


# using OO interface from https://matplotlib.org/gallery/api/agg_oo_sgskip.html


def plot_predicted_vs_true(true, predicted, savepath, title='predicted vs true'):
    ####w, h = figaspect(2.)
    fig = Figure(figsize=(w,h))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    #ax.grid(True)
    ax.scatter(true, predicted, edgecolors=(0, 0, 0))
    ax.plot([true.min(), true.max()], [true.min(), true.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    fig.savefig(savepath)

def plot_residuals_histogram(true, pred, savepath, title='residuals histogram', stats=dict()):

    # make the aspect ration wide for text next to square-ish graph
    w, h = figaspect(0.7)

    fig = Figure(figsize=(w,h))
    FigureCanvas(fig)

    # these two lines are where the magic happens, trapping the figure on the left side
    # so we can make print text beside it
    gs = plt.GridSpec(2, 3)
    ax = fig.add_subplot(gs[0:2, 0:2], aspect='equal')


    ax.set_title(title)
    # do the actual plotting
    residuals = true - pred
    ax.hist(residuals, bins=30)

    # normal text stuff
    ax.set_xlabel('residual')
    ax.set_ylabel('frequency')

    # make y axis ints, because it is discrete
    #ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # shrink those margins
    fig.tight_layout()

    # print stats onto the image. Goes off screen if they are too long or too many in number
    text_height = 0.08
    for i, (name, val) in enumerate(stats.items()):
        y_pos = 1 - (text_height * i + 0.1)
        fig.text(0.7, y_pos, f'{name}: {val}')

    fig.savefig(savepath)
