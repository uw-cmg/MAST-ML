#!/usr/bin/env python
##############
# Data verification
#   - create per-alloy plots for hardening vs. log(fluence)
# TTM 2017-02-08
##############

import pymongo
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from bson.objectid import ObjectId

def get_alloy_list(db, clist, verbose=1):
    alloys = list()
    for cname in clist:
        alloys.extend(db[cname].distinct("Alloy"))
    alloy_set = set(alloys)
    alloys = list(alloy_set) #unique names
    alloys.sort() #sort in place
    if verbose > 0:
        print(alloys)
    return alloys

def make_per_alloy_plots(db, clist, pathstem="", verbose=0):
    """Make per-alloy plots
        Args:
            db <Mongo client>: database
            clist <list of str>: list of collection names
            pathstem <str>: path for figures
    """
    if not os.path.isdir(pathstem):
        os.mkdir(pathstem)
    alloys = get_alloy_list(db, clist, verbose)
    markerlist = ['o','x','^','s','d','v','+']
    markersize = 10
    fontsize = 18
    markeredgewidth = 3
    #for alloy in alloys[0:2]: #switch with below and uncomment for testing
    for alloy in alloys:
        print("Plotting alloy %s" % alloy)
        plt.figure()
        plt.hold(True)
        plt.title(alloy, fontsize=fontsize)
        plt.xlabel("Fluence (n/cm$^{2}$)", fontsize=fontsize)
        plt.ylabel("$\Delta\sigma_{y}$ (MPa)", fontsize=fontsize)
        plt.tick_params(labelsize=fontsize)
        for cname in clist:
            cseries=list()
            results = db[cname].find({"Alloy":alloy},{"fluence_n_cm2":1,
                    "delta_sigma_y_MPa":1})
            for result in results:
                cseries.append([result["fluence_n_cm2"],result["delta_sigma_y_MPa"]])
            cseries = np.array(cseries, 'float')
            marker = markerlist[hash_string(cname) % len(markerlist)]
            if cseries.size > 0:
                plt.plot(cseries[:,0],cseries[:,1], marker = marker,
                    markersize = markersize, markeredgewidth = markeredgewidth,
                    markerfacecolor = "None", linewidth = 0,
                    markeredgecolor = string_to_color(cname),label=cname)
        plt.xscale('log')
        lgd=plt.legend(loc="best", fontsize = fontsize, fancybox = True)
        lgd.get_frame().set_alpha(0.5) #translucent legend!
        plt.tight_layout()
        plt.savefig(os.path.join(pathstem, "%s_verification.png" % alloy))
        plt.close()
    return

def hash_string(mystr=""):
    """Adapted from http://stackoverflow.com/questions/11120840/hash-string-into-rgb-color, Jeff Foster and clayzermk1
    """
    hashval = 5381
    for ict in range(0, len(mystr)):
        hashval = ((hashval << 5) + hashval) + ord(mystr[ict])
    return hashval

def string_to_color(mystr=""):
    """Adapted from http://stackoverflow.com/questions/11120840/hash-string-into-rgb-color, Jeff Foster and clayzermk1
    """
    hashval = hash_string(mystr)
    rval = (hashval & 0xFF0000) >> 16;
    gval= (hashval & 0x00FF00) >> 8;
    bval = hashval & 0x0000FF;
    rstr = format(rval, 'x')[-2:].zfill(2)
    gstr = format(gval, 'x')[-2:].zfill(2)
    bstr = format(bval, 'x')[-2:].zfill(2)
    color_str = "#" + rstr + gstr + bstr
    return color_str

if __name__=="__main__":
    print("Warning: use through DataImportAndExport.py, not on its own")
    from pymongo import MongoClient
    dbname="dbtt_23"
    client = MongoClient('localhost', 27017)
    db = client[dbname]
    clist=["expt_ivar","cd1_ivar","cd2_ivar","cd2_lwr"]
    make_per_alloy_plots(db, clist, sys.argv[1])
