import os
import matplotlib
import numpy as np
import data_parser
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mean_error import mean_error
import data_analysis.printout_tools as ptools
import time

def simple_histogram(xvals,
            num_bins = 50,
            bin_width = None,
            start_val = None,
            end_val = None,
            bin_list = None,
            bin_space = 3,
            savepath="",
            xlabel="X",
            ylabel="Count",
            title="",
            timex="",
            tick_divide=None,
            plotlabel="histogram",
            notelist=list(),
            ):
    """
        Args:
            bin_list <str>: Comma-separated string of values, 
                            starting with leftmost bin-left and
                            ending with rightmost bin-right, e.g. '0,1,2,3'
    """
    matplotlib.rcParams.update({'font.size': 18})
    smallfont = 0.85*matplotlib.rcParams['font.size']
    darkblue="#00008B"
    #http://matplotlib.org/1.2.1/examples/pylab_examples/histogram_demo.html
    if bin_list is None:
        if start_val is None:
            minval = min(xvals)
        else:
            if type(start_val) == str:
                if len(timex) > 0:
                    start_val = time.mktime(time.strptime(start_val, timex))
            minval = float(start_val)
        if end_val is None:
            maxval = max(xvals)
        else:
            if type(end_val) == str:
                if len(timex) > 0:
                    end_val = time.mktime(time.strptime(end_val, timex))
            maxval = float(end_val)
        if bin_width is None:
            bin_width = (maxval - minval) / int(num_bins)
        else:
            bin_width = float(bin_width)
        blist = np.arange(minval, maxval + bin_width, bin_width)
    else: 
        blist = np.array(bin_list.split(","),'float')
    n_per_bin, bins, patches = plt.hist(xvals, bins=blist, normed=0,
                facecolor='blue', alpha=0.75)
    ## add a 'best fit' line
    #bestfit = mlab.normpdf( bins, mu, sigma)
    #plt.plot(bins, bestfit, linestyle='--', linewidth=1,
    #                color = darkblue)
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
    myax = plt.gca()
    myax.set_xlim(bins[0],bins[-1])
    lenbins = len(bins)
    if lenbins > 6: #plot only a few xticks
        binskip = int(bin_space)
        ticklist = list()
        binct = 1000000
        for binval in bins:
            if binct > binskip:
                ticklist.append(binval)
                binct = 1
            binct = binct + 1
        if not (bins[-1] in ticklist):
            ticklist.append(bins[-1])
        myax.set_xticks(ticklist)
    else:
        myax.set_xticks(bins)
    if len(timex) > 0:
        myax = plt.gca()
        my_xticks = myax.get_xticks()
        adjusted_xticks = list()
        for tidx in range(0, len(my_xticks)):
            mytick = time.strftime(timex, time.localtime(my_xticks[tidx]))
            adjusted_xticks.append(mytick)
        myax.set_xticklabels(adjusted_xticks, rotation=90.0)
    else:
        if not(tick_divide is None):
            my_xticks = myax.get_xticks()
            adjusted_xticks = my_xticks / float(tick_divide)
            myax.set_xticklabels(adjusted_xticks)
    savestr = "%s_%s" % (plotlabel, xlabel.replace(" ","_"))
    plt.savefig(os.path.join(savepath, savestr), bbox_inches='tight')
    plt.close()
    headerline = "bin,n_per_bin"
    bins = np.array(bins,'float')
    n_per_bin = np.append(n_per_bin,None) #last bin value is just the end of the bins
    n_per_bin = np.array(n_per_bin, 'float')
    myarray = np.array([bins, n_per_bin]).transpose()
    csvname = os.path.join(savepath, "%s.csv" % savestr)
    ptools.array_to_csv(csvname, headerline, myarray)
    return

