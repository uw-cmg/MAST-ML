#!/usr/bin/env python
###################
# Data analysis printout tools
# Tam Mayeshiba 2017-02-15
###################
import numpy as np
import os
import sys
import traceback
import subprocess
import time

def array_to_csv(csvname, headerstring, array, fmtstr=None):
    """Save a numeric-only array to a CSV file
    """
    with open(csvname,"w") as dfile:
        dfile.write("%s\n" % headerstring)
    
    with open(csvname,"ab") as dfile:
        if fmtstr == None:
            np.savetxt(dfile, array, delimiter=",")
        else: 
            np.savetxt(dfile, array, fmt=fmtstr, delimiter=",")
    return

def mixed_array_to_csv(csvname, headerstring, array):
    """Save a mixed array to a CSV file
    """
    with open(csvname,"w") as dfile:
        dfile.write("%s\n" % headerstring)
        (nlines, ncols) = array.shape
        for ridx in range(0, nlines):
            line = ""
            for cidx in range(0, ncols):
                line = line + "%s," % array[ridx][cidx]
            line = line[:-1] #remove last comma
            line = line + "\n"
            dfile.write(line)
    return
