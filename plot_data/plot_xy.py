import os
import matplotlib
import numpy as np
import data_parser
import matplotlib.pyplot as plt
from mean_error import mean_error
import data_analysis.printout_tools as ptools

def get_xy_sorted(xvals, yvals, verbose=0):
    """Sort x and y according to x. 
    """
    combarr = np.array([xvals, yvals],'float')
    if verbose > 0:
        print("Original:")
        print(combarr)
    sortedarr = combarr[:,np.argsort(combarr[0])]
    if verbose > 0:
        print("Sorted:")
        print(sortedarr)
    xsorted = sortedarr[0,:]
    ysorted = sortedarr[1,:]
    return [xsorted, ysorted]

def single(xvals, yvals, 
            plottype="scatter",
            xlabel="X",
            ylabel="Y",
            title="",
            savepath="",
            notelist=list(),
            ):
    matplotlib.rcParams.update({'font.size': 18})
    smallfont = 0.85*matplotlib.rcParams['font.size']
    fig, ax = plt.subplots(figsize=(10, 4))
    [xvals, yvals] = get_xy_sorted(xvals, yvals)
    if plottype == "scatter":
        plt.plot(xvals, yvals, linestyle = "None",
                    marker='o', markersize=8, markeredgewidth=2,
                    markeredgecolor='blue', markerfacecolor="None")
    elif plottype == "line":
        plt.plot(xvals, yvals, linestyle = "-",
                    linewidth=2,
                    color='blue')
    else:
        plt.plot(xvals, yvals, linestyle = "-",
                    linewidth=2,
                    color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(title) > 0:
        plt.title(title)
    notey = 0.88
    notestep = 0.07
    for note in notelist:
        plt.annotate(note, xy=(0.05, notey), xycoords="axes fraction",
                    fontsize=smallfont)
        notey = notey - notestep
    fig.savefig(os.path.join(savepath, "%s_vs_%s" % (ylabel, xlabel)), dpi=200, bbox_inches='tight')
    fig.clf()
    plt.close()
    return
