import os
import matplotlib
import numpy as np
import data_parser
import matplotlib.pyplot as plt
from mean_error import mean_error
import data_analysis.printout_tools as ptools

def get_steps(gmin, gmax, resolution=4):
    grange = (gmax - gmin)
    gstep = grange / float(resolution) #
    if gstep > 1000:
        roundnum = -3
    elif gstep > 100:
        roundnum = -2
    elif gstep > 10:
        roundnum = -1
    elif gstep > 1:
        roundnum = 0
    elif gstep > 0.1:
        roundnum = 1
    elif gstep > 0.01:
        roundnum = 2
    gstep = np.round(gstep, roundnum)
    gstart = np.round(gmin - gstep, roundnum)
    gend = np.round(gmax + gstep, roundnum)
    steplist = np.arange(gstart, gend, gstep)
    return steplist

def vs_leftoutgroup(group_value_list = None, 
            rms_list=None,
            group_label_list = None,
            xlabel="Left out group",
            ylabel="RMSE",
            title="",
            savepath="",
            notelist=list(),
            ):
    # graph rmse vs left-out group
    matplotlib.rcParams.update({'font.size': 18})
    smallfont = 0.85*matplotlib.rcParams['font.size']
    fig, ax = plt.subplots(figsize=(10, 4))
    if group_value_list is None:
        group_value_list = np.arange(0, len(group_label_list))
    skipticks = np.ceil(len(group_value_list)/8)
    xticks = np.arange(0, max(group_value_list) + 1, skipticks, dtype='int')
    if group_label_list is None:
        plt.xticks(xticks)
    else:
        xticklabels = list()
        for xtick in xticks:
            xticklabels.append(group_label_list[xtick])
        plt.xticks(xticks, xticklabels)
    ax.scatter(group_value_list, rms_list, color='black', s=10)
    #plot zero line
    ax.plot((0, max(group_value_list)+1), (0, 0), ls="--", c=".3") 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(title) > 0:
        ax.set_title(title)
    notey = 0.88
    notestep = 0.07
    for note in notelist:
        plt.annotate(note, xy=(0.05, notey), xycoords="axes fraction",
                    fontsize=smallfont)
        notey = notey - notestep
    for x in np.argsort(rms_list)[-5:]:
        if group_label_list is None:
            alabel = group_value_list[x]
        else:
            alabel = group_label_list[x]
        ax.annotate(s = alabel,
                    xy = (group_value_list[x], rms_list[x]),
                    fontsize=smallfont)
    fig.savefig(os.path.join(savepath, "leavegroupout_cv"), dpi=200, bbox_inches='tight')
    fig.clf()
    plt.close()
    return
