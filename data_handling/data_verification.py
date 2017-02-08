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

def make_per_alloy_plots(db, clist, verbose=0):
    """Make per-alloy plots
        Args:
            db <Mongo client>: database
            clist <list of str>: list of collection names
    """
    print("Maybe would be cleaner to use per-set spreadsheets/dbs, so that")
    print("each spreadsheet would have singly-named delta_sigma_y_MPa")
    alloys = get_alloy_list(db, clist, verbose)
    #for alloy in alloys:
    for alloy in alloys[0:2]:
        expt_dsy=list()
        cd_ivar_2016_dsy=list()
        cd_ivar_2017_dsy=list()
        cd_lwr_2017_dsy=list()
        print(alloy)
        for cname in clist:
            if "ivar" in cname:
                results = db[cname].find({"Alloy":alloy},{"fluence_n_cm2":1,
                        "delta_sigma_y_MPa":1,
                        "delta_sigma_y_MPa_cd_ivar_2016":1,
                        "delta_sigma_y_MPa_cd_ivar_2017":1})
                for result in results:
                    expt_dsy.append([result["fluence_n_cm2"],result["delta_sigma_y_MPa"]])
                    cd_ivar_2016_dsy.append([result["fluence_n_cm2"],
                                result["delta_sigma_y_MPa_cd_ivar_2016"]])
                    cd_ivar_2017_dsy.append([result["fluence_n_cm2"],
                                result["delta_sigma_y_MPa_cd_ivar_2017"]])
            elif "lwr" in cname:
                results = db[cname].find({"Alloy":alloy},{"fluence_n_cm2":1,
                        "CD_delta_sigma_y_MPa":1}) 
                for result in results:
                    cd_lwr_2017_dsy.append([result["fluence_n_cm2"],
                                result["CD_delta_sigma_y_MPa"]])
        expt_dsy = np.array(expt_dsy,'float')
        cd_ivar_2016_dsy = np.array(cd_ivar_2016_dsy,'float')
        cd_ivar_2017_dsy = np.array(cd_ivar_2017_dsy,'float')
        cd_lwr_2017_dsy = np.array(cd_lwr_2017_dsy,'float')
        plt.figure()
        plt.hold(True)
        plt.scatter(expt_dsy[:,0],expt_dsy[:,1],color="r",label="expt")
        plt.scatter(cd_ivar_2016_dsy[:,0],cd_ivar_2016_dsy[:,1],color="b",label="cd_ivar_2016")
        plt.scatter(cd_ivar_2017_dsy[:,0],cd_ivar_2017_dsy[:,1],color="g",label="cd_ivar_2017")
        plt.scatter(cd_lwr_2017_dsy[:,0],cd_lwr_2017_dsy[:,1],color="black",label="cd_lwr_2017")
        plt.xscale('log')
        plt.title(alloy)
        plt.xlabel("log fluence (n/cm2)")
        plt.ylabel("delta sigma y (MPa)")
        plt.legend(loc="upper left")
        plt.savefig("%s_verification.png" % alloy)
    return

if __name__=="__main__":
    print("Warning: use through DataImportAndExport.py, not on its own")
    from pymongo import MongoClient
    dbname="dbtt_16"
    client = MongoClient('localhost', 27017)
    db = client[dbname]
    clist=["ivar_ivarplus","lwr_2017"]
    make_per_alloy_plots(db, clist)
