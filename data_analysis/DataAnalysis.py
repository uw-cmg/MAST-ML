#!/usr/bin/env python
###################
# Data analysis full test suite
# Tam Mayeshiba 2017-02-13
#
# This script is intended to be run on a local computer with
# previously generated .csv files.
#
# Prerequisites:
# 1. Must have starting import csv files available.
#
###################
import numpy as np
import os
import sys
import traceback
import subprocess
import time
import data_parser

def get_feature_list():
    features=list()
    features.append("N(at_percent_Cu)")
    features.append("N(at_percent_Ni)")
    features.append("N(at_percent_Mn)")
    features.append("N(at_percent_P)")
    features.append("N(at_percent_Si)")
    features.append("N(at_percent_C)")
    #features.append("N(log(fluence_n_cm2))")
    #features.append("N(log(flux_n_cm2_sec))")
    features.append("N(temperature_C)")
    #features.append("N(log(eff fl 100p=26))")
    features.append("N(log(eff fl 100p=20))")
    features.append("N(log(eff fl 100p=10))")
    return features

def get_gkrr_hyperparams(testpath, csvname="grid_scores.csv", verbose=0):
    dirname = os.path.dirname(testpath)
    mname = "KRRGridSearch"
    mdir = os.path.join(dirname, mname)
    hdict=dict()
    hdict["alpha"] = 0.0
    hdict["gamma"] = 0.0
    if not os.path.isdir(mdir):
        print("No hyperparameter results yet.")
        return hdict
    hdata_path = os.path.join(mdir, csvname) 
    hdata = data_parser.parse(hdata_path)
    hdata.set_y_feature("rms")
    rms = hdata.get_y_data() 
    param_feats = list(hdict.keys())
    param_feats.sort() #SORTED keys
    if verbose > 0:
        print("parameter features: %s" % param_feats)
    hdata.set_x_features(param_feats)
    params = hdata.get_x_data()
    small_idx = rms.index(min(rms))
    if verbose > 0:
        print("index number: %i" % small_idx)
    param_vals = params[small_idx]
    for pidx in range(0, len(param_feats)):
        param = param_feats[pidx]
        param_val = param_vals[pidx]
        hdict[param] = param_val
    if verbose > 0:
        print(hdict)
    return hdict

def write_config_file(testpath, dsetname, testname, testdict):
    fname = os.path.join(testpath, "default.conf") #currently only takes default.conf as test name??
    lines = list()
    lines.append("[default]")
    lines.append("data_path = ../../%s_ivar.csv" % dsetname)
    lines.append("save_path = %s/{}.png" % testpath)
    lines.append("lwr_data_path = ../../%s_lwr.csv" % dsetname)
    features = get_feature_list()
    xstr = "X = "
    for feature in features:
        xstr = xstr + feature + ","
    xstr = xstr[:-1] #remove trailing comma
    lines.append(xstr)
    lines.append("Y = delta_sigma_y_MPa")
    lines.append("weights = False")
    #
    lines.append("[AllTests]")
    lines.append("data_path = ${default:data_path}")
    lines.append("save_path = ${default:save_path}")
    lines.append("lwr_data_path = ${default:lwr_data_path}")
    lines.append("weights = ${default:weights}")
    lines.append("X = ${default:X}")
    lines.append("Y = ${default:Y}")
    lines.append("model = gkrr_model")
    lines.append("test_cases = %s" % testname)  
    #
    lines.append("[gkrr_model]")
    print("Change hardcode")
    hdict = get_gkrr_hyperparams(testpath, "grid_scores.csv")
    for hkey in hdict.keys():
        lines.append("%s = %3.8f" % (hkey, hdict[hkey]))
    lines.append("coef0 = 1")
    lines.append("degree = 3")
    lines.append("kernel = rbf")
    #
    lines.append("[%s]" % testname)
    for key, value in testdict[dsetname][testname].items():
        lines.append("%s = %s" % (key, value)) 
    linesn = list()
    for line in lines:
        linesn.append("%s\n" % line)
    fhandle = open(fname, 'w')
    fhandle.writelines(linesn)
    fhandle.close()
    return

def do_analysis(testpath, scriptpath):
    curdir = os.getcwd()
    os.chdir(testpath)
    rproc = subprocess.Popen("nice -n 19 python %s/AllTests.py" % scriptpath, shell=True,
                    stdout = subprocess.PIPE,
                    stderr = subprocess.PIPE)
    rproc.wait()
    (status,message)=rproc.communicate()
    print(status.decode('utf-8'))
    print(message.decode('utf-8'))
    os.chdir(curdir)
    return

def main(datapath, scriptpath):
    testdict=dict() #could get from a file later
    dnames = ["expt","cd1","cd2"]
    for dname in dnames:
        testdict[dname] = dict()
    grid_density = 20
    #testdict["expt"]["KRRGridSearch"] = {"grid_density":grid_density}
    #testdict["cd1"]["KRRGridSearch"] = {"grid_density":grid_density}
    #
    testdict["cd2"]["KRRGridSearch"] = {"grid_density":grid_density}
    testdict["cd2"]["KFold_CV"] = {"num_runs":20,"num_folds":5}
    testdict["cd2"]["LeaveOutAlloyCV"] = {}
    testdict["cd2"]["FullFit"] = {}
    testdict["cd2"]["PredictionVsFluence"] = {}
    testdict["cd2"]["ExtrapolateToLWR"] = {}
    for dsetname in testdict.keys():
        dpath = os.path.join(datapath, dsetname)
        if not os.path.isdir(dpath):
            os.mkdir(dpath)
        testnames = list(testdict[dsetname].keys())
        testnames.sort()
        if "KRRGridSearch" in testnames: #Do grid search first, always
            testnames.remove("KRRGridSearch")
            testnames.insert(0,"KRRGridSearch")
        print(testnames)
        for testname in testnames:
            tpath = os.path.join(dpath, testname)
            if not os.path.isdir(tpath):
                os.mkdir(tpath)
            write_config_file(tpath, dsetname, testname, testdict)
            do_analysis(tpath, scriptpath)
    return

if __name__ == "__main__":
    datapath = "../../../data/DBTT_mongo/data_exports_dbtt_05_20170214_124539"
    scriptpath = "../"
    datapath = os.path.abspath(datapath)
    scriptpath = os.path.abspath(scriptpath)
    #get_gkrr_hyperparams(os.path.join(datapath,"cd2","PredictionVsFluence"))
    main(datapath, scriptpath)
    print("Files in %s" % datapath)
    sys.exit()
