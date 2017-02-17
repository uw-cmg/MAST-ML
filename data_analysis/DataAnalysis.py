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
import data_analysis.additional_plots as ap

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

def get_gkrr_hyperparams(testpath, testdict, csvname="grid_scores.csv", verbose=0):
    setpath = os.path.dirname(testpath)
    dsetname = os.path.basename(setpath)
    if "hyperfrom" in testdict[dsetname].keys():
        dfrom = testdict[dsetname]["hyperfrom"]
    else:
        dfrom = dsetname
    toppath = os.path.dirname(setpath)
    mname = "KRRGridSearch"
    hdata_path = os.path.join(toppath, dfrom, mname, csvname) 
    hdict=dict()
    hdict["alpha"] = 0.0
    hdict["gamma"] = 0.0
    if not os.path.isfile(hdata_path):
        print("No hyperparameter results yet.")
        return hdict
    print("Using hyperparams from %s" % hdata_path)
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
    ivarcsv = testdict[dsetname]["csvs"]["ivar"]
    lwrcsv = testdict[dsetname]["csvs"]["lwr"]
    lines.append("data_path = ../../%s.csv" % ivarcsv)
    lines.append("lwr_data_path = ../../%s.csv" % lwrcsv)
    lines.append("save_path = %s/{}.png" % testpath)
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
    hdict = get_gkrr_hyperparams(testpath, testdict, "grid_scores.csv")
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
    ofname = os.path.join(testpath,"output")
    with open(ofname, 'w') as ofile:
        rproc=subprocess.Popen("nice -n 19 python %s/AllTests.py" % scriptpath, 
                        shell=True,
                        stdout = ofile,
                        #stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE)
    rproc.wait()
    (status,message)=rproc.communicate()
    if not status == None:
        print(status.decode('utf-8'))
    if not message == None:
        print(message.decode('utf-8'))
    os.chdir(curdir)
    return

def main(datapath, scriptpath):
    testdict=dict() #could get from a file later
    grid_density = 20 #orig 20
    num_runs = 200 #orig 200
    num_folds = 5
    #
    testdict["1exptalltemp"]=dict()
    #testdict["1exptalltemp"]["KRRGridSearch"] = {"grid_density":grid_density}
    #testdict["1exptalltemp"]["KFold_CV"] = {"num_runs":num_runs,"num_folds":num_folds}
    #testdict["1exptalltemp"]["FullFit"] = {}
    testdict["1exptalltemp"]["csvs"] ={"ivar":"expt_ivaralltemp", "lwr":"cd1_lwr"}
    #
    testdict["2cd1alltemp"]=dict()
    #testdict["2cd1alltemp"]["KRRGridSearch"] = {"grid_density":grid_density}
    #testdict["2cd1alltemp"]["KFold_CV"] = {"num_runs":num_runs,"num_folds":num_folds}
    #testdict["2cd1alltemp"]["LeaveOutAlloyCV"] = {}
    #testdict["2cd1alltemp"]["FullFit"] = {}
    #testdict["2cd1alltemp"]["ExtrapolateToLWR"] = {}
    testdict["2cd1alltemp"]["PredictionVsFluence"] = {"temp_filter":290}
    testdict["2cd1alltemp"]["csvs"] ={"ivar":"cd1_ivaralltemp", "lwr":"cd1_lwr"}
    #
    testdict["3expt"]=dict()
    #testdict["3expt"]["KRRGridSearch"] = {"grid_density":grid_density}
    #testdict["3expt"]["KFold_CV"] = {"num_runs":num_runs,"num_folds":num_folds}
    #testdict["3expt"]["LeaveOutAlloyCV"] = {}
    #testdict["3expt"]["FullFit"] = {}
    testdict["3expt"]["PredictionVsFluence"] = {}
    #testdict["3expt"]["ExtrapolateToLWR"] = {}
    testdict["3expt"]["csvs"] ={"ivar":"expt_ivar", "lwr":"cd1_lwr"}
    testdict["3expt"]["hyperfrom"] = "1exptalltemp"
    #
    testdict["4cd1"]=dict()
    #testdict["4cd1"]["KRRGridSearch"] = {"grid_density":grid_density}
    #testdict["4cd1"]["KFold_CV"] = {"num_runs":num_runs,"num_folds":num_folds}
    #testdict["4cd1"]["FullFit"] = {}
    #testdict["4cd1"]["LeaveOutAlloyCV"] = {}
    testdict["4cd1"]["PredictionVsFluence"] = {}
    #testdict["4cd1"]["ExtrapolateToLWR"] = {}
    testdict["4cd1"]["csvs"] ={"ivar":"cd1_ivar", "lwr":"cd1_lwr"}
    testdict["4cd1"]["hyperfrom"] = "2cd1alltemp"
    #
    #testdict["5cd2"]=dict()
    #testdict["5cd2"]["KRRGridSearch"] = {"grid_density":grid_density}
    #testdict["5cd2"]["KFold_CV"] = {"num_runs":num_runs,"num_folds":num_folds}
    #testdict["5cd2"]["LeaveOutAlloyCV"] = {}
    #testdict["5cd2"]["FullFit"] = {}
    #testdict["5cd2"]["PredictionVsFluence"] = {}
    #testdict["5cd2"]["ExtrapolateToLWR"] = {}
    #testdict["5cd2"]["csvs"] ={"ivar":"cd2_ivar", "lwr":"cd2_lwr"}
    dsets = list(testdict.keys())
    dsets.sort()
    for dsetname in dsets:
        dpath = os.path.join(datapath, dsetname)
        if not os.path.isdir(dpath):
            os.mkdir(dpath)
        testnames = list(testdict[dsetname].keys())
        testnames.sort()
        if "KRRGridSearch" in testnames: #Do grid search first, always
            testnames.remove("KRRGridSearch")
            testnames.insert(0,"KRRGridSearch")
        if "csvs" in testnames:
            testnames.remove("csvs")
        if "hyperfrom" in testnames:
            testnames.remove("hyperfrom")
        print(testnames)
        for testname in testnames:
            tpath = os.path.join(dpath, testname)
            if not os.path.isdir(tpath):
                os.mkdir(tpath)
            write_config_file(tpath, dsetname, testname, testdict)
            do_analysis(tpath, scriptpath)
        if ("FullFit" in testnames) and ("KFold_CV" in testnames):
            ap.cross_validation_full_fit_plot(dpath)
    return

if __name__ == "__main__":
    datapath = "../../../data/DBTT_mongo/data_exports_dbtt_17_20170216_160031"
    scriptpath = "../"
    datapath = os.path.abspath(datapath)
    scriptpath = os.path.abspath(scriptpath)
    main(datapath, scriptpath)
    print("Files in %s" % datapath)
    sys.exit()
