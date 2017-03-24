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
        label1="",
        label2="",
        plotlabel="overlay",
        stepsize="1.0",
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
    
    datadict=dict()
    for pidx in range(0, len(csvs)):
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
        datadict[pidx]=dict()
        datadict[pidx]['xdata'] = xdata
        datadict[pidx]['ydata'] = ydata
        datadict[pidx]['xerrdata'] = xerrdata
        datadict[pidx]['yerrdata'] = yerrdata
    kwargs=dict()
    kwargs['savepath'] = savepath
    kwargs['stepsize'] = stepsize
    if xlabel == "":
        xlabel="%s, %s" % (xfields[0],xfields[1])
    if ylabel == "":
        ylabel="%s, %s" % (yfields[0],yfields[1])
    kwargs['xlabel'] = xlabel
    kwargs['ylabel'] = ylabel
    if label1 == "":
        label1=os.path.basename(csvs[0]).split(".")[0]
    if label2 == "":
        label2=os.path.basename(csvs[1]).split(".")[0]
    kwargs['label1'] = label1
    kwargs['label2'] = label2
    plotxy.dual_overlay(datadict[0]['xdata'], datadict[0]['ydata'],
        datadict[1]['xdata'], datadict[1]['ydata'],
        xerr1=datadict[0]['xerrdata'],
        yerr1=datadict[0]['yerrdata'],
        xerr2=datadict[1]['xerrdata'],
        yerr2=datadict[1]['yerrdata'],
        **kwargs)
    return
