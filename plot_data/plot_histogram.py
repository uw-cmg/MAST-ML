import os
import matplotlib
import numpy as np
import data_parser
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mean_error import mean_error
import data_analysis.printout_tools as ptools

def simple_histogram(xvals,
            num_bins = 50,
            savepath="",
            xlabel="X",
            ylabel="Count",
            title="",
            plotlabel="histogram",
            notelist=list(),
            ):
    matplotlib.rcParams.update({'font.size': 18})
    smallfont = 0.85*matplotlib.rcParams['font.size']
    darkblue="#00008B"
    #http://matplotlib.org/1.2.1/examples/pylab_examples/histogram_demo.html
    n_per_bin, bins, patches = plt.hist(xvals, num_bins, normed=0,
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

