import os
import matplotlib
import numpy as np
import data_parser
import matplotlib.pyplot as plt
from mean_error import mean_error
import data_analysis.printout_tools as ptools

def best_worst(Ydata, Y_predicted_best, Y_predicted_worst, 
        xlabel="Measured",
        ylabel="Predicted",
        savepath="",
        notelist_best=list(), 
        notelist_worst=list(), 
        *args, **kwargs):
    """Plot best (left) and worst (right) Predicted vs. Measured values.
    """
    matplotlib.rcParams.update({'font.size': 18})
    notestep = 0.07
    f, ax = plt.subplots(1, 2, figsize = (11,5))
    ax[0].scatter(Ydata, Y_predicted_best, c='black', s=10)
    [minx,maxx] = ax[0].get_xlim()
    [miny,maxy] = ax[0].get_ylim()
    gmax = max(maxx, maxy)
    gmin = min(minx, miny)
    ax[0].set_xticks(np.arange(gmin, gmax, 200))
    ax[0].set_yticks(np.arange(gmin, gmax, 200))
    ax[0].plot((gmin+1, gmax-1), (gmin+1, gmax-1), ls="--", c=".3")
    ax[0].set_title('Best Fit')
    notey = 0.88
    for note in notelist_best:
        ax[0].text(.05, notey, note, transform=ax[0].transAxes)
        notey = notey - notestep
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)

    ax[1].scatter(Ydata, Y_predicted_worst, c='black', s=10)
    [minx,maxx] = ax[1].get_xlim()
    [miny,maxy] = ax[1].get_ylim()
    gmax = max(maxx, maxy)
    gmin = min(minx, miny)
    ax[1].set_xticks(np.arange(gmin, gmax, 200))
    ax[1].set_yticks(np.arange(gmin, gmax, 200))
    ax[1].plot((gmin+1, gmax-1), (gmin+1, gmax-1), ls="--", c=".3")
    ax[1].set_title('Worst Fit')
    notey = 0.88
    for note in notelist_worst:
        ax[1].text(.05, notey, note, transform=ax[1].transAxes)
        notey = notey - notestep
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)

    f.tight_layout()
    f.savefig(os.path.join(savepath, "cv_best_worst"), dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

def single(Ydata, Y_predicted, 
        xlabel="Measured",
        ylabel="Predicted",
        savepath="",
        notelist=list(), 
        *args, **kwargs):
    """Plot Predicted vs. Measured values.
    """
    matplotlib.rcParams.update({'font.size': 18})
    notestep = 0.07
    plt.figure()
    plt.scatter(Ydata, Y_predicted_best, c='black', s=10)
    [minx,maxx] = plt.get_xlim()
    [miny,maxy] = plt.get_ylim()
    gmax = max(maxx, maxy)
    gmin = min(minx, miny)
    plt.set_xticks(np.arange(gmin, gmax, 200))
    plt.set_yticks(np.arange(gmin, gmax, 200))
    plt.plot((gmin+1, gmax-1), (gmin+1, gmax-1), ls="--", c=".3")
    notey = 0.88
    for note in notelist:
        plt.annotate(note, xy=(0.05, notey), xycoords="axes fraction",
                    fontsize=smallfont)
        notey = notey - notestep
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, "cv_singleplot"), dpi=200, bbox_inches='tight')
    plt.close()
    return
