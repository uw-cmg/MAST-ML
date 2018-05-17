import os
import matplotlib
import numpy as np
#import data_parser
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
#from mean_error import mean_error
#import data_analysis.printout_tools as ptools
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
            guideline = 1,
            climbing_percent = 1,
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
    fig, axh = plt.subplots()
    n_per_bin, bins, patches = axh.hist(xvals, bins=blist, normed=0,
                facecolor='blue', edgecolor='black', alpha=0.75)
    if int(guideline) > 0:
        from scipy.stats import norm
        import matplotlib.mlab as mlab
        if int(guideline) == 1:
            (mu,sigma) = norm.fit(xvals)
            guidey = mlab.normpdf( bins, mu, sigma)
        elif int(guideline) == 2:
            from scipy.stats import lognorm
            log_xvals = np.ma.log(xvals)
            (mu, sigma) = norm.fit(log_xvals)
            scale = np.exp(mu)
            sparam = sigma
            loc = mu
            guidey = lognorm.pdf( bins, sparam, loc, scale)
        guidey = guidey * sum(n_per_bin) * bin_width # scale up the fit
        axh.plot(bins, guidey, linestyle='--', color = darkblue, linewidth = 1)
        #https://www.mathworks.com/matlabcentral/newsreader/view_thread/32136.html?
    axh.set_xlabel(xlabel)
    axh.set_ylabel(ylabel)
    if int(climbing_percent) == 1:
        cplist = list()
        ntotal = sum(n_per_bin)
        cplen = len(n_per_bin)
        cumulative = 0.0
        if ntotal == 0:
            cplist = np.zeros(cplen+1)
        else:
            cplist.append(0) #left point of first bin
            for bidx in range(0, cplen): #goes to right point of last bin
                binn = n_per_bin[bidx]
                if binn:
                    cumulative = cumulative + binn
                percval = cumulative/ntotal
                cplist.append(percval)
        cplist = np.array(cplist,'float') * 100.0 #percent
        axp = axh.twinx()
        darkred="#8B0000"
        axp.plot(bins, cplist, linestyle='-', color = darkred, linewidth = 1)
        axp.set_ylabel('Cumulative percentage', color=darkred)
        axp.tick_params('y',colors=darkred)
    if len(title) > 0:
        plt.title(title)
    notey = 0.88
    notestep = 0.07
    for note in notelist:
        plt.annotate(note, xy=(0.05, notey), xycoords="axes fraction",
                    fontsize=smallfont)
        notey = notey - notestep
    axh.set_xlim(bins[0],bins[-1])
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
        axh.set_xticks(ticklist)
    else:
        axh.set_xticks(bins)
    if len(timex) > 0:
        my_xticks = axh.get_xticks()
        adjusted_xticks = list()
        for tidx in range(0, len(my_xticks)):
            mytick = time.strftime(timex, time.localtime(my_xticks[tidx]))
            adjusted_xticks.append(mytick)
        axh.set_xticklabels(adjusted_xticks, rotation=90.0)
    else:
        if not(tick_divide is None):
            my_xticks = axh.get_xticks()
            adjusted_xticks = my_xticks / float(tick_divide)
            axh.set_xticklabels(adjusted_xticks)
    savestr = "%s_%s" % (plotlabel, xlabel.replace(" ","_"))
    plt.savefig(os.path.join(savepath, savestr), bbox_inches='tight')
    plt.close()
    headerline = "bin,n_per_bin"
    bins = np.array(bins,'float')
    n_per_bin = np.append(n_per_bin,None) #last bin value is just the end of the bins
    n_per_bin = np.array(n_per_bin, 'float')
    myarray = np.array([bins, n_per_bin]).transpose()
    csvname = os.path.join(savepath, "%s.csv" % savestr)
    #ptools.array_to_csv(csvname, headerline, myarray)
    return

