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
            xerr=None,
            yerr=None,
            xlabel="X",
            ylabel="Y",
            title="",
            savepath="",
            guideline=0,
            notelist=list(),
            *args, **kwargs):
    matplotlib.rcParams.update({'font.size': 18})
    smallfont = 0.85*matplotlib.rcParams['font.size']
    #fig, ax = plt.subplots(figsize=(10, 4))
    [xvals, yvals] = get_xy_sorted(xvals, yvals)
    darkblue="#00008B"
    mylinestyle = "-"
    mymarker = "o"
    if plottype == "scatter":
        mylinestyle = "None"
    elif plottype == "line":
        mymarker = "None"
    (_, caps, _) = plt.errorbar(xvals, yvals,
        xerr=xerr,
        yerr=yerr,
        linewidth=2,
        linestyle = mylinestyle, color=darkblue,
        markeredgewidth=2, markeredgecolor=darkblue,
        markerfacecolor="blue" , marker=mymarker,
        markersize=10)
    for cap in caps:
        cap.set_color(darkblue)
        cap.set_markeredgewidth(2)
    plt.margins(0.05)
    if guideline == 1: #also square the axes
        [minx,maxx] = plt.xlim()
        [miny,maxy] = plt.ylim()
        gmax = max(maxx, maxy)
        gmin = min(minx, miny)
        plt.xlim(gmin, gmax)
        plt.ylim(gmin, gmax)
        plt.plot((gmin, gmax), (gmin, gmax), ls="--", c=".3")
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
    savestr = "%s_vs_%s" % (ylabel.replace(" ","_"), xlabel.replace(" ","_"))
    plt.savefig(os.path.join(savepath, savestr), bbox_inches='tight')
    plt.close()
    return

def dual_overlay(xdata1, ydata1,
        xdata2, ydata2,
        label1="series 1",
        label2="series 2",
        xlabel="X",
        ylabel="Y",
        xerr1=None,
        yerr1=None,
        xerr2=None,
        yerr2=None,
        stepsize=1,
        savepath="",
        plotlabel="dual_overlay",
        guideline=0,
        fill=1,
        equalsize=1,
        notelist=list(), 
        *args, **kwargs):
    """Plot dual xy overlay
    """
    matplotlib.rcParams.update({'font.size': 18})
    smallfont = 0.85*matplotlib.rcParams['font.size']
    notestep = 0.07
    plt.figure()
    darkred="#8B0000"
    darkblue="#00008B"
    if fill == 1:
        face1 = "red"
        face2 = "blue"
    else:
        face1 = "None"
        face2 = "None"
    if equalsize == 1:
        size1 = 15
        size2 = 15
    else:
        size1 = 15
        size2 = 10
    if xerr1 is None:
        xerr1 = np.zeros(len(xdata1))
    if yerr1 is None:
        yerr1 = np.zeros(len(ydata1))
    if xerr2 is None:
        xerr2 = np.zeros(len(xdata2))
    if yerr2 is None:
        yerr2 = np.zeros(len(ydata2))
    (_, caps, _) = plt.errorbar(xdata1, ydata1,
        xerr=xerr1,
        yerr=yerr1,
        label=label1,
        linewidth=2,
        linestyle = "None", color=darkred,
        markeredgewidth=2, markeredgecolor=darkred,
        markerfacecolor=face1 , marker='o',
        markersize=size1)
    for cap in caps:
        cap.set_color(darkred)
        cap.set_markeredgewidth(2)
    (_, caps, _) = plt.errorbar(xdata2, ydata2, 
        xerr=xerr2,
        yerr=yerr2,
        label=label2,
        linewidth=2,
        linestyle = "None", color=darkblue,
        markeredgewidth=2, markeredgecolor=darkblue,
        markerfacecolor=face2 , marker='o',
        markersize=size2)
    for cap in caps:
        cap.set_color(darkblue)
        cap.set_markeredgewidth(2)
    lgd=plt.legend(loc = "lower right", 
                    fontsize=smallfont, 
                    numpoints=1,
                    fancybox=True) 
    lgd.get_frame().set_alpha(0.5) #translucent legend!
    [minx,maxx] = plt.xlim()
    [miny,maxy] = plt.ylim()
    gmax = max(maxx, maxy)
    gmin = min(minx, miny)
    steplist = np.arange(gmin, gmax + (0.5*stepsize), stepsize)
    plt.xticks(steplist)
    plt.yticks(steplist)
    plt.margins(0.05)
    if guideline == 1:
        plt.plot((gmin, gmax), (gmin, gmax), ls="--", c=".3")
    notey = 0.88
    for note in notelist:
        plt.annotate(note, xy=(0.05, notey), xycoords="axes fraction",
                    fontsize=smallfont)
        notey = notey - notestep
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, "%s" % plotlabel), dpi=200, bbox_inches='tight')
    plt.close()
    return
