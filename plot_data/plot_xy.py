import os
import matplotlib
import numpy as np
import data_parser
import matplotlib.pyplot as plt
from mean_error import mean_error
import data_analysis.printout_tools as ptools
import matplotlib.dates as mdates
import time
import heapq

def get_xy_sorted(xvals, yvals, xerr, yerr, verbose=0):
    """Sort x and y according to x. 
    """
    arraylist = list()
    arraylist.append(xvals)
    arraylist.append(yvals)
    if xerr is None:
        dummy_xerr = np.zeros(len(xvals))
        arraylist.append(dummy_xerr)
    else:
        arraylist.append(xerr)
    if yerr is None:
        dummy_yerr = np.zeros(len(yvals)) #len(yvals) should also be len(xvals)
        arraylist.append(dummy_yerr)
    else:
        arraylist.append(yerr)
    combarr = np.array(arraylist,'float')
    if verbose > 0:
        print("Original:")
        print(combarr)
    sortedarr = combarr[:,np.argsort(combarr[0])]
    if verbose > 0:
        print("Sorted:")
        print(sortedarr)
    xsorted = sortedarr[0,:]
    ysorted = sortedarr[1,:]
    if xerr is None:
        xerrsorted = None
    else:
        xerrsorted = sortedarr[2,:]
    if yerr is None:
        yerrsorted = None
    else:
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
            startx=None,
            endx=None,
            divide_x = None,
            divide_y = None,
            notelist=list(),
            marklargest=0,
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
    if not(startx == None):
        if type(startx) == str:
            if len(timex) > 0:
                startx = time.mktime(time.strptime(startx, timex))
            else:
                startx = float(startx)
        if endx == None:
            raise ValueError("startx must be paired with endx")
        if type(endx) == str:
            if len(timex) > 0:
                endx = time.mktime(time.strptime(endx, timex))
            else:
                endx = float(endx)
        plt.xlim([startx,endx])
    if guideline == 1: #also square the axes
        [minx,maxx] = plt.xlim()
        [miny,maxy] = plt.ylim()
        gmax = max(maxx, maxy)
        gmin = min(minx, miny)
        plt.xlim(gmin, gmax)
        plt.ylim(gmin, gmax)
        plt.plot((gmin, gmax), (gmin, gmax), ls="--", c=".3")
    plt.margins(0.05)
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
    if int(marklargest) > 0:
        import heapq
        maxidxlist = heapq.nlargest(int(marklargest), range(len(yvals)), 
                        key=lambda x: yvals[x])
        for midx in maxidxlist:

            mxval = xvals[midx]
            mxval = "%3.0f" % mxval
            plt.annotate("%s" % mxval, 
                    xy=(xvals[midx],yvals[midx]),
                    horizontalalignment = "left",
                    verticalalignment = "bottom",
                    fontsize=smallfont)
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
        stepsize=None,
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
        stepsize=None,
        savepath="",
        plotlabel="multiple_overlay",
        guideline=0,
        timex="",
        startx=None,
        endx=None,
        whichyaxis=list(),
        notelist=list(), 
        marklargest="0,0,0,0,0,0",
        mlabellist=None,
        markers="o,o,s,d,^,v",
        linestyles="None,None,None,None,None,None",
        outlines="#8B0000,#00008B,#004400,#542788,#b35806,#252525",
        faces="red,blue,green,#6a51a3,orange,#bdbdbd",
        sizes="15,10,8,8,8,8",
        *args, **kwargs):
    """Plot multiple xy overlay with same x axis
    """
    guideline = int(guideline)
    #VERIFICATION
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
        print("Not enough x error data. Use python None for no error.")
        return
    if not(len(yerrlist) == numlines):
        print("Not enough y error data. Use python None for no error.")
        return
    if len(whichyaxis) == 0:
        whichyaxis = np.ones(numlines)
    else:
        whichyaxis = whichyaxis.split(",")
    if not (len(whichyaxis) == numlines):
        print("Not enough axis choice data. whichyaxis should be a list of 1's and 2's.")
        return
    whichyaxis = np.array(whichyaxis, 'float')
    if sum(whichyaxis) > numlines: #has some 2's
        doubley = True
    else:
        doubley = False
    if doubley:
        ylabels = ylabel.split(",")
        if not (len(ylabels) == numlines):
            print("Not enough y label data.")
            return
    #PLOTTING
    matplotlib.rcParams.update({'font.size': 18})
    smallfont = 0.85*matplotlib.rcParams['font.size']
    notestep = 0.07
    plt.figure()
    faces = faces.split(",")
    outlines = outlines.split(",")
    linestyles = linestyles.split(",")
    markers = markers.split(",")
    sizes=np.array(sizes.split(","),'float')
    fig, ax1 = plt.subplots()
    if doubley:
        ax2 = ax1.twinx()
    for nidx in range(0, numlines):
        label = labellist[nidx]
        xdata = xdatalist[nidx]
        ydata = ydatalist[nidx]
        xerr = xerrlist[nidx]
        yerr = yerrlist[nidx]
        [xdata,ydata,xerr,yerr] = get_xy_sorted(xdata,ydata,xerr,yerr)
        whichy = whichyaxis[nidx]
        if whichy == 1:
            (_, caps, _) = ax1.errorbar(xdata, ydata,
                xerr=xerr,
                yerr=yerr,
                label=label,
                linewidth=2,
                linestyle = linestyles[nidx], color=outlines[nidx],
                markeredgewidth=2, markeredgecolor=outlines[nidx],
                markerfacecolor=faces[nidx] , marker=markers[nidx],
                markersize=sizes[nidx])
        else:
            (_, caps, _) = ax2.errorbar(xdata, ydata,
                xerr=xerr,
                yerr=yerr,
                label=label,
                linewidth=2,
                linestyle = linestyles[nidx], color=outlines[nidx],
                markeredgewidth=2, markeredgecolor=outlines[nidx],
                markerfacecolor=faces[nidx] , marker=markers[nidx],
                markersize=sizes[nidx])
        for cap in caps:
            cap.set_color(outlines[nidx])
            cap.set_markeredgewidth(2)
    #AXIS LABELS
    if doubley:
        ylabel1 = ""
        ylabel2 = ""
        for nidx in range(0, numlines):
            if whichyaxis[nidx] == 1:
                ylabel1 = ylabel1 + ylabels[nidx] + "; "
            else:
                ylabel2 = ylabel2 + ylabels[nidx] + "; "
        ylabel1 = ylabel1[:-2] #remove trailing semicolon
        ylabel2 = ylabel2[:-2] #remove trailing semicolon
        ax1.set_ylabel(ylabel1)
        ax2.set_ylabel(ylabel2)
    else:
        ax1.set_ylabel(ylabel)
    plt.xlabel(xlabel)
    #X-AXIS RANGE
    if not(startx == None):
        if type(startx) == str:
            if len(timex) > 0:
                startx = time.mktime(time.strptime(startx, timex))
            else:
                startx = float(startx)
        if endx == None:
            raise ValueError("startx must be paired with endx")
        if type(endx) == str:
            if len(timex) > 0:
                endx = time.mktime(time.strptime(endx, timex))
            else:
                endx = float(endx)
        ax1.set_xlim([startx,endx])
        if doubley:
            ax2.set_xlim([startx,endx])
    #X and Y-AXIS RANGES FOR SQUARE PLOT
    [minx,maxx] = ax1.get_xlim() #should always be the same for x2
    if guideline == 1: #square the axes according to stepsize and draw line
        if doubley:
            raise ValueError("Cannot plot multiple y and also square axes.")
        [miny,maxy] = ax1.get_ylim()
        gmax = max(maxx, maxy)
        gmin = min(minx, miny)
        ax1.set_xlim([gmin,gmax]) #set both X and Y limits
        ax1.set_ylim([gmin,gmax])
        plt.plot((gmin, gmax), (gmin, gmax), ls="--", c=".3")
    else:
        gmax = maxx
        gmin = minx
    #XTICKS AND POSSIBLY YTICKS
    if not (stepsize == None):
        stepsize = float(stepsize)
        steplist = np.arange(gmin, gmax + (0.5*stepsize), stepsize)
        if len(steplist < 1000): #don't allow too many ticks
            ax1.set_xticks(steplist)
            if doubley:
                ax2.set_xticks(steplist)
            if guideline == 1:
                ax1.set_yticks(steplist) 
    plt.margins(0.05)
    #ANNOTATIONS
    notey = 0.88
    for note in notelist:
        plt.annotate(note, xy=(0.05, notey), xycoords="axes fraction",
                    fontsize=smallfont)
        notey = notey - notestep
    #ANNOTATIONS FOR LARGEST
    marklargest = np.array(marklargest.split(","),'int')
    for nidx in range(0, numlines):
        marknum = marklargest[nidx]
        if marknum == 0: #no marking
            continue
        if int(guideline) == 0: #just rank on y
            torank = ydatalist[nidx]
        else: #rank on distance from x-y guideline
            torank = np.abs(ydatalist[nidx] - xdatalist[nidx])
        meantorank = np.nanmean(torank) #get rid of NaN's, esp. on CV test
        mynans = np.isnan(torank)
        torank[mynans] = meantorank #NaN's become means, so not high or low
        maxidxlist = heapq.nlargest(marknum, range(len(torank)), 
                        key=lambda x: torank[x])
        print(maxidxlist)
        if mlabellist is None:
            mlabellist = np.copy(xdatalist)
        for midx in maxidxlist:
            print(xdatalist[nidx][midx])
            print(ydatalist[nidx][midx])
            mxval = mlabellist[nidx][midx]
            mxval = "%3.0f" % mxval
            plt.annotate("%s" % mxval, 
                    xy=(xdatalist[nidx][midx],ydatalist[nidx][midx]),
                    horizontalalignment = "left",
                    verticalalignment = "bottom",
                    fontsize=smallfont)
    #X-AXIS RELABELING
    if len(timex) > 0:
        my_xticks = ax1.get_xticks()
        adjusted_xticks = list()
        for tidx in range(0, len(my_xticks)):
            mytick = time.strftime(timex, time.localtime(my_xticks[tidx]))
            adjusted_xticks.append(mytick)
        ax1.set_xticklabels(adjusted_xticks, rotation=90.0)
        if doubley:
            ax2.set_xticklabels(adjusted_xticks, rotation=90.0)
    #LEGEND
    if doubley:
        lgd2=ax2.legend(loc = "lower right",
                        bbox_to_anchor=(1.0,1.0),
                        fontsize=smallfont, 
                        numpoints=1,
                        fancybox=True) 
        lgd2.get_frame().set_alpha(0.5) #translucent legend!
        ax2.set_ylabel(ylabel2)
        lgd1=ax1.legend(loc = "lower left",
                    bbox_to_anchor=(0.0,1.0),
                    fontsize=smallfont, 
                    numpoints=1,
                    fancybox=True) 
    else:
        if guideline:
            loc1 = "lower right"
        else:
            loc1 = "best"
        lgd1=ax1.legend(loc = loc1, 
                    fontsize=smallfont, 
                    numpoints=1,
                    fancybox=True) 
    lgd1.get_frame().set_alpha(0.5) #translucent legend!
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, "%s" % plotlabel), dpi=200, bbox_inches='tight')
    plt.close()
    #PRINT DATA
    for nidx in range(0, numlines):
        label = labellist[nidx]
        savecsv = os.path.join(savepath,"data_%s.csv" % label)
        savecsv = savecsv.replace(" ","_")
        headerstr="%s" % xlabel
        myarrlist = list()
        myarrlist.append(xdatalist[nidx])
        if not(xerr is None):
            headerstr = headerstr + ",xerr,"
            myarrlist.append(xerrlist[nidx])
        headerstr = headerstr + "%s" % label
        myarrlist.append(ydatalist[nidx])
        if not (yerr is None):
            headerstr = headerstr + ",yerr"
            myarrlist.append(yerrlist[nidx])
        myarray = np.asarray(myarrlist).transpose()
        ptools.mixed_array_to_csv(savecsv, headerstr, myarray)
    return
