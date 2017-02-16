#!/usr/bin/env python
###################
# Data analysis additional plots
# Tam Mayeshiba 2017-02-15
###################
import numpy as np
import os
import sys
import traceback
import subprocess
import time
import matplotlib
import matplotlib.pyplot as plt
from mean_error import mean_error
from sklearn.metrics import mean_squared_error

def cross_validation_full_fit_plot():

    matplotlib.rcParams.update({'font.size': 18})
    fullfit_data = np.genfromtxt("FullFit/FullFit_data.csv", dtype=float, delimiter=',', names=True) 
    kfold_data = np.genfromtxt("KFold_CV/KFoldCV_data.csv", dtype=float, delimiter=',', names=True) 
    fullmeas = fullfit_data['Measured']
    fullpred = fullfit_data['Predicted']
    cvbestmeas = kfold_data['Measured']
    cvbestpred = kfold_data['Predicted_best']
    full_rms = mean_squared_error(fullmeas, fullpred)
    full_me = mean_error(fullmeas, fullpred)
    cvbest_rms = mean_squared_error(cvbestmeas, cvbestpred)
    cvbest_me = mean_error(cvbestmeas, cvbestpred)

    plt.figure()
    plt.hold(True)
    plt.plot(cvbestmeas, cvbestpred, markerfacecolor='red', 
        markeredgecolor="None", marker='o',markersize='15',linestyle="None") #best
    plt.plot(fullmeas, fullpred,markerfacecolor='black', 
        markeredgecolor="None",marker='o',markersize='8',linestyle="None")
    
    [xmin, xmax] = plt.gca().get_xlim()
    [ymin, ymax] = plt.gca().get_ylim()
    gmin = min(xmin, ymin)
    gmax = max(xmax, ymax)
    plt.axis('equal')
    plt.plot(np.arange(gmin,gmax,10),np.arange(gmin,gmax,10),':',color="gray")
    
    plt.annotate('Fit RMSE: {:.2f} MPa'.format(full_rms), xy=(0.05,0.90), color="black", xycoords='axes fraction')
    plt.annotate('CV RMSE: {:.2f} MPa'.format(cvbest_rms), xy=(0.05,0.83), color="red", xycoords='axes fraction')
    plt.annotate('Mean Error: {:.2f} MPa'.format(full_me), xy=(0.05,0.76), color="black", xycoords='axes fraction')
    plt.annotate('Mean Error: {:.2f} MPa'.format(cvbest_me), xy=(0.05,0.69), color="red", xycoords='axes fraction')
    
    plt.xlabel("Measured IVAR+ $\Delta\sigma_{y}$ (MPa)")
    plt.ylabel("GKRR predicted $\Delta\sigma_{y}$ (MPa)")
    plt.tight_layout()
    plt.savefig("cross_validation_full_fit.png")
    return


