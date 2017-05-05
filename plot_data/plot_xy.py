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
    try:
        combarr = np.array(arraylist,'float')
    except ValueError as ve:
        import traceback
        print("Error combining array")
        print("Shapes for arraylist xvals, yvals, xerr, yerr")
        for cidx in range(0,4):
            print(arraylist[cidx].shape)
        raise ValueError(traceback.print_exc())
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
            save_path="",
            guideline=0,
            timex="",
            startx=None,
            endx=None,
            stepsize=None,
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
    kwargs=dict()
    kwargs['xdatalist'] = list([xvals])
    kwargs['ydatalist'] = list([yvals])
    kwargs['xlabel'] = xlabel
    kwargs['ylabel'] = ylabel
    kwargs['xerrlist'] = list([xerr])
    kwargs['yerrlist'] = list([yerr])
    kwargs['stepsize'] = stepsize
    if plotlabel == "":
        plotlabel = "%s_vs_%s" % (ylabel, xlabel)
    kwargs['plotlabel'] = plotlabel
    kwargs['labellist'] = list(["_%s" % plotlabel]) #keep out of legend
    kwargs['save_path'] = save_path
    kwargs['guideline'] = guideline
    kwargs['timex'] = timex
    kwargs['startx'] = startx
    kwargs['endx'] = endx
    kwargs['notelist'] = notelist
    kwargs['whichyaxis'] = "1"
    kwargs['marklargest'] = "%i" % marklargest
    kwargs['mlabellist'] = None
    multiple_overlay(**kwargs)
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
        save_path="",
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
    kwargs['save_path'] = save_path
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
        save_path="",
        plotlabel="multiple_overlay",
        guideline=0,
        timex="",
        startx=None,
        endx=None,
        whichyaxis="",
        notelist=list(), 
        marklargest="0,0,0,0,0,0",
        mlabellist=None,
        markers="o,o,s,d,^,v",
        linestyles="None,None,None,None,None,None",
        outlines="#8B0000,#00008B,#004400,#542788,#b35806,#252525",
        faces="red,blue,green,#6a51a3,orange,#bdbdbd",
        sizes="15,10,8,8,8,8",
        legendloc=None,
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
    if not(type(faces) is list):
        faces = faces.split(",")
    if not(type(outlines) is list):
        outlines = outlines.split(",")
    if not(type(linestyles) is list):
        linestyles = linestyles.split(",")
    if not(type(markers) is list):
        markers = markers.split(",")
    if not(type(sizes) is list):
        sizes=np.array(sizes.split(","),'float')
    else:
        sizes = np.array(sizes, 'float') #make sure they are floats
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
    if type(marklargest) is str:
        marklargest = np.array(marklargest.split(","),'int')
    elif type(marklargest) is list:
        marklargest = np.array(marklargest,'int')
    else:
        raise ValueError("marklargest %s could not be identified." % marklargest)
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
            mxval = mlabellist[nidx][midx]
            try:
                mxval = "%3.0f" % mxval
            except TypeError:
                pass
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
        try:
            lgd2.get_frame().set_alpha(0.5) #translucent legend!
        except AttributeError: # no labeled lines
            pass
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
        if not(legendloc is None):
            loc1 = legendloc
        lgd1=ax1.legend(loc = loc1, 
                    fontsize=smallfont, 
                    numpoints=1,
                    fancybox=True) 
    try:
        lgd1.get_frame().set_alpha(0.5) #translucent legend!
    except AttributeError: # no labeled lines
        pass
    plt.tight_layout()
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    plt.savefig(os.path.join(save_path, "%s" % plotlabel), dpi=200, bbox_inches='tight')
    plt.close()
    #PRINT DATA
    for nidx in range(0, numlines):
        label = labellist[nidx]
        nospace_label = label.replace(" ","_")
        savecsv = os.path.join(save_path,"data_%s.csv" % nospace_label)
        headerstr="%s," % xlabel
        myarrlist = list()
        myarrlist.append(xdatalist[nidx])
        if not(xerr is None):
            headerstr = headerstr + "xerr,"
            myarrlist.append(xerrlist[nidx])
        headerstr = headerstr + "%s," % label
        myarrlist.append(ydatalist[nidx])
        if not (yerr is None):
            headerstr = headerstr + "yerr,"
            myarrlist.append(yerrlist[nidx])
        headerstr=headerstr[:-1] #remove last comma
        myarray = np.asarray(myarrlist).transpose()
        ptools.mixed_array_to_csv(savecsv, headerstr, myarray)
    return
