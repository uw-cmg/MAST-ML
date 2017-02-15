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

def array_to_csv(csvname, headerstring, array, fmtstr="%3.3f"):
    """Save an array to a CSV file
    """
    with open(csvname,"w") as dfile:
        dfile.write("%s\n" % headerstring)
    
    with open(csvname,"ab") as dfile:
        np.savetxt(dfile, array, fmt=fmtstr, delimiter=",")

    return
