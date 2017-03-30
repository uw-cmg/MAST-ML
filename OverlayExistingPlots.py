import matplotlib
import matplotlib.pyplot as plt
import data_parser
import numpy as np
import data_analysis.printout_tools as ptools
import plot_data.plot_xy as plotxy
import os
import portion_data.get_test_train_data as gttd
############
# Overlay plots
# Tam Mayeshiba 2017-03-24
############
def execute(model, data, savepath,
        csvlist="",
        xfieldlist="",
        yfieldlist="",
        xerrfieldlist="",
        yerrfieldlist="",
        xlabel="",
        ylabel="",
        labellist="",
        plotlabel="overlay",
        guideline=0,
        equalsize=1,
        fill=1,
        timex="",
        stepsize="1.0",
        startx=None,
        endx=None,
        *args, **kwargs):
    """Overlay plots
        Args:
            csvlist <str>: comma-delimited list of csv names
                            Currently only supports two csvs. 
            xfieldlist <str>: comma-delimited list of x-field names, to
                                match with csvlist
            xerrfieldlist <str>: comma-delimited list of x error field names, to
                                match with csvlist
            yfieldlist <str>: comma-delimited list of y-field names, to
                                match with csvlist
            yerrfieldlist <str>: comma-delimited list of y error field names, to
                                match with csvlist
    """
    stepsize = float(stepsize)
    csvs = csvlist.split(",")
    print(csvs)
    xfields = xfieldlist.split(",")
    yfields = yfieldlist.split(",")
    
    if not (len(csvs) == len(xfields)):
        print("Length of x field list not match length of csv list.")
        print("Exiting.")
        return
    if not (len(csvs) == len(yfields)):
        print("Length of y field list does not match length of csv list.")
        print("Exiting.")
        return
    if len(xerrfieldlist) > 0:
        xerrfields = xerrfieldlist.split(",")
        if not (len(xerrfields) == len(xfields)):
            print("Length of x error field list does not match length of x field list.")
            print("Exiting.")
            return
    else:
        xerrfields=list()
    if len(yerrfieldlist) > 0:
        yerrfields = yerrfieldlist.split(",")
        if not (len(yerrfields) == len(yfields)):
            print("Length of y error field list does not match length of y field list.")
            print("Exiting.")
            return
    else:
        yerrfields=list()
    
    xdatas=list()
    ydatas=list()
    xerrs=list()
    yerrs=list()
    for pidx in range(0, len(csvs)):
        print("Getting data from %s" % csvs[pidx])
        data = data_parser.parse(csvs[pidx].strip())
        xdata = np.asarray(data.get_data(xfields[pidx].strip())).ravel()
        ydata = np.asarray(data.get_data(yfields[pidx].strip())).ravel()
        xerrdata = None
        yerrdata = None
        if len(xerrfields) > 0:
            xerrfield = xerrfields[pidx].strip()
            if not(xerrfield == ""):
                xerrdata=np.asarray(data.get_data(xerrfield)).ravel()
        if len(yerrfields) > 0:
            yerrfield = yerrfields[pidx].strip()
            if not(yerrfield == ""):
                yerrdata=np.asarray(data.get_data(yerrfield)).ravel()
        xdatas.append(xdata)
        ydatas.append(ydata)
        xerrs.append(xerrdata)
        yerrs.append(yerrdata)
    if xlabel == "":
        xlabel="%s" % xfields
    if ylabel == "":
        ylabel="%s" % yfields
    if labellist == "":
        labellist=list()
        for csvname in csvs:
            labellist.append(os.path.basename(csvs[0]).split(".")[0])
    else:
        labellist = labellist.split(",")
    kwargs=dict()
    kwargs['xdatalist'] = xdatas
    kwargs['ydatalist'] = ydatas
    kwargs['labellist'] = labellist
    kwargs['xlabel'] = xlabel
    kwargs['ylabel'] = ylabel
    kwargs['xerrlist'] = xerrs
    kwargs['yerrlist'] = yerrs
    kwargs['stepsize'] = stepsize
    kwargs['savepath'] = savepath
    kwargs['plotlabel'] = plotlabel
    kwargs['guideline'] = guideline
    kwargs['fill'] = fill
    kwargs['equalsize'] = equalsize
    kwargs['timex'] = timex
    kwargs['startx'] = startx
    kwargs['endx'] = endx
    notelist=list()
    kwargs['notelist'] = notelist
    #for key,value in kwargs.items():
    #    print(key,":",value)
    print("Plotting.")
    plotxy.multiple_overlay(**kwargs)
    return
