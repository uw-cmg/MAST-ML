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

def get_feature_list():
    features=list()
    features.append("N(at_percent_Cu)")
    features.append("N(at_percent_Ni)")
    features.append("N(at_percent_Mn)")
    features.append("N(at_percent_P)")
    features.append("N(at_percent_Si)")
    features.append("N(at_percent_C)")
    features.append("N(log(fluence_n_cm2))")
    features.append("N(log(flux_n_cm2_sec))")
    features.append("N(temperature_C)")
    features.append("N(log(eff fl 100p=26))")
    return features

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
    lines.append("alpha = 0.002682696") 
    lines.append("gamma = 0.61054023")
    #for CD 2 set: 0.002682696  0.61054023 RMSE 38.22
    # automate putting in alpha and gamma
    # 7.84759970351e-05,0.194748303991,38.0207617107 #among lowest RMSEs; 
    # matches previously-reported alpha and gamma from Jerit
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
    grid_density = 8
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
        for testname in testdict[dsetname]:
            tpath = os.path.join(dpath, testname)
            if not os.path.isdir(tpath):
                os.mkdir(tpath)
            write_config_file(tpath, dsetname, testname, testdict)
            do_analysis(tpath, scriptpath)
    return

if __name__ == "__main__":
    datapath = "../../../data/DBTT_mongo/data_exports_dbtt_36_20170213_120942"
    scriptpath = "../"
    datapath = os.path.abspath(datapath)
    scriptpath = os.path.abspath(scriptpath)
    main(datapath, scriptpath)
    print("Files in %s" % datapath)
    sys.exit()
