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
from data_handling import DataImportAndExport
from data_analysis import DataAnalysis

def main(importpath, scriptpath):
    exportpath = DataImportAndExport.main(importpath)
    DataAnalysis.main(exportpath, scriptpath)
    return

if __name__ == "__main__":
    importpath = "../../../data/DBTT_mongo/imports_201702"
    scriptpath = "../"
    scriptpath = os.path.abspath(scriptpath)
    importpath = os.path.abspath(importpath)
    main(importpath, scriptpath)
    sys.exit()
