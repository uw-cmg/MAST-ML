import os
import matplotlib
import numpy as np
import data_parser
import matplotlib.pyplot as plt
from mean_error import mean_error
import data_analysis.printout_tools as ptools
import matplotlib.dates as mdates
import time

def get_xy_sorted(xvals, yvals, xerr, yerr, verbose=0):
    """Sort x and y according to x. 
    """
    combarr = np.array([xvals, yvals, xerr, yerr],'float')
    if verbose > 0:
        print("Original:")
        print(combarr)
    sortedarr = combarr[:,np.argsort(combarr[0])]
    if verbose > 0:
        print("Sorted:")
        print(sortedarr)
    xsorted = sortedarr[0,:]
    ysorted = sortedarr[1,:]
    xerrsorted = sortedarr[2,:]
    yerrsorted = sortedarr[3,:]
    return [xsorted, ysorted, xerrsorted, yerrsorted]

def get_converted_epoch_xticks(xticks):
    """Matplotlib needs epoch days
    """
    tzseconds = time.timezone
    isdaylight = time.localtime().tm_isdst
    secadj_xticks = xticks - tzseconds
    if isdaylight:
        secadj_xticks = secadj_xticks + 3600.0
    numticks = len(secadj_xticks)
    adjusted_xticks = np.zeros(numticks)
    for xidx in range(0, numticks):
        epochday = matplotlib.dates.epoch2num(secadj_xticks[xidx])
        adjusted_xticks[xidx] = epochday        
    return adjusted_xticks

def single(xvals, yvals, 
            plottype="scatter",
            xerr=None,
            yerr=None,
            xlabel="X",
            ylabel="Y",
            title="",
            plotlabel="",
            savepath="",
            guideline=0,
            timex="",
            divide_x = None,
            divide_y = None,
            notelist=list(),
            *args, **kwargs):
    matplotlib.rcParams.update({'font.size': 18})
    smallfont = 0.85*matplotlib.rcParams['font.size']
    #fig, ax = plt.subplots(figsize=(10, 4))
    if xerr is None or (len(xerr) == 0):
        xerr = np.zeros(len(xvals))
    if yerr is None or (len(yerr) == 0):
        yerr = np.zeros(len(yvals))
    [xvals, yvals, xerr, yerr] = get_xy_sorted(xvals, yvals, xerr, yerr)
    if not (divide_x is None):
        xvals = xvals / float(divide_x)
        xerr = xerr / float(divide_x)
    if not (divide_y is None):
        yvals = yvals / float(divide_y)
        yerr = yerr / float(divide_y)
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
    if len(timex) > 0:
        myax = plt.gca()
        my_xticks = myax.get_xticks()
        adjusted_xticks = list()
        for tidx in range(0, len(my_xticks)):
            mytick = time.strftime(timex, time.localtime(my_xticks[tidx]))
            adjusted_xticks.append(mytick)
        myax.set_xticklabels(adjusted_xticks, rotation=90.0)
    savestr = "%s_vs_%s" % (ylabel.replace(" ","_"), xlabel.replace(" ","_"))
    if len(plotlabel) > 0:
        savestr = "%s_" % plotlabel + savestr
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
    xdatalist=list([xdata1, xdata2])
    ydatalist=list([ydata1, ydata2])
    xerrlist=list([xerr1, xerr2])
    yerrlist=list([yerr1, yerr2])
    labellist=list([label1,label2])

    kwargs=dict()
    kwargs['stepsize'] = stepsize
    kwargs['savepath'] = savepath
    kwargs['plotlabel'] = plotlabel
    kwargs['guideline'] = guideline
    kwargs['fill'] = fill
    kwargs['equalsize'] = equalsize
    kwargs['notelist'] = notelist
    kwargs['xlabel'] = xlabel
    kwargs['ylabel'] = ylabel
    kwargs['xdatalist'] = xdatalist
    kwargs['ydatalist'] = ydatalist
    kwargs['xerrlist'] = xerrlist
    kwargs['yerrlist'] = yerrlist
    kwargs['labellist'] = labellist
    
    multiple_overlay(**kwargs)
    return

def multiple_overlay(xdatalist=list(), ydatalist=list(), labellist=list(),
        xlabel="X",
        ylabel="Y",
        xerrlist=list(),
        yerrlist=list(),
        stepsize=1,
        savepath="",
        plotlabel="multiple_overlay",
        guideline=0,
        fill=1,
        equalsize=1,
        notelist=list(), 
        *args, **kwargs):
    """Plot multiple xy overlay
    """
    numlines=len(xdatalist)
    if numlines > 6:
        print("Only 6 lines supported.")
        print("Exiting.")
        return
    if not(len(ydatalist) == numlines):
        print("Not enough y data. Exiting.")
        return
    if not(len(labellist) == numlines):
        print("Not enough labels. Exiting.")
        return
    if not(len(xerrlist) == numlines):
        print("Not enough x error data. Use empty lists for no error.")
        return
    if not(len(yerrlist) == numlines):
        print("Not enough y error data. Use empty lists for no error.")
        return
    matplotlib.rcParams.update({'font.size': 18})
    smallfont = 0.85*matplotlib.rcParams['font.size']
    notestep = 0.07
    plt.figure()
    #colors are red, blue, green, purple, brown, gray
    outlines=["#8B0000","#00008B","#004400","#542788","#b35806","#252525"]
    if fill == 1:
        faces=["red","blue","green","#6a51a3","orange","#bdbdbd"]
    else:
        faces=["None","None","None","None","None","None"]
    bigsize=15
    smallsize=10
    if equalsize == 1:
        sizes=[bigsize,bigsize,bigsize,bigsize,bigsize,bigsize]
    else:
        sizes=[bigsize,smallsize,smallsize,smallsize,smallsize,smallsize]
    markers=['o','o','s','d','^','v']
    for nidx in range(0, numlines):
        label = labellist[nidx]
        xdata = xdatalist[nidx]
        ydata = ydatalist[nidx]
        xerr = xerrlist[nidx]
        if (xerr is None) or len(xerr) == 0:
            xerr = np.zeros(len(xdata))
        yerr = yerrlist[nidx]
        if (yerr is None) or len(yerr) == 0:
            yerr = np.zeros(len(ydata))
        (_, caps, _) = plt.errorbar(xdata, ydata,
            xerr=xerr,
            yerr=yerr,
            label=label,
            linewidth=2,
            linestyle = "None", color=outlines[nidx],
            markeredgewidth=2, markeredgecolor=outlines[nidx],
            markerfacecolor=faces[nidx] , marker=markers[nidx],
            markersize=sizes[nidx])
        for cap in caps:
            cap.set_color(outlines[nidx])
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
