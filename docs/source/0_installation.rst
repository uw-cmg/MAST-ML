#####################
Installing MASTML
#####################

*************
Requirements
*************

========
Hardware
========
Computer capable of running python 3.

=========
Data
=========

* Numeric data in a CSV file. There must be at least some target feature data, so that models can be fit.

* (Advanced: Data in a database with an associated python class that can extract the data to a CSV file.)

**************
Installation
**************

===============
Install Python3
===============

Install Python 3: for easier installation of numpy and scipy dependencies, use either:

* anaconda https://www.continuum.io/downloads OR

* conda lite https://conda.io/miniconda.html

Create a conda environment:

Anaconda::

    conda create --name TEST_ML numpy scipy matplotlib pandas sphinx nose 

Miniconda::

    conda create --name TEST_ML

Activate the environment::

    source activate TEST_ML
    (For Windows: activate TEST_ML)

=====================================
Install the MAST-ml-private package
=====================================

Git clone the repository  (github.com/uw-cmg/MAST-ml-private)
Clone from “master” 
Ask for access if you cannot find this code.

Check status.github.com for issues if you believe github may be malfunctioning

``python setup.py install`` (inside the MAST-ml-private folder) (might not work)
If “SyntaxError: Invalid syntax” occurs at lines 16, 17, or 34 try wrapping prints in () for windows [e.g. print ("Python Version %d.%d.%d found" % (sys.version_info[0], sys.version_info[1], sys.version_info[2])) ]

If setup.py didn’t work, dependencies you need are:      

numpy>=1.11.2
scipy>=0.18.1
pandas>=0.19.2
matplotlib>=1.5.3
configobj>=5.0.6
validator
scikit-learn>=0.18.1
pymongo>=3.4.0
pymatgen>=4.6.0
PeakUtils>=1.0.3

-------------------------
imports that don’t work 
-------------------------
First try anaconda install, and if that gives errors try pip install
Example: conda install numpy , or pip install numpy
Put MAST-ml-private’s folder in your PYTHONPATH if it isn’t already
Adding to PYTHONPATH and getting it to stay there
If you get a missing import error, go ahead and install any other missing packages and submit a github ticket.

*******************
Startup
*******************

==================
Set up a folder
==================
#. Make a folder (let’s call it my_test)

#. Copy basic_test_data.csv into this folder.

#. Copy the mastmlvalidation*.conf files (both of them) into this folder. In the General Setup section, these mastmlvalidation*.conf files should read save_path = string (etc; keyword and then type)

#. Copy basicinput.conf into this folder

#. Change directories to be in this folder (cd my_test)

========================
Run the MASTML command
========================

Format is ``python <path to MASTML>/MASTML.py <config file name>``

For example::
    
    python MAST-ml-private/MASTML.py basicinput.conf

This is a terminal command. If using Anaconda GUI, try using Anaconda’s Python Shell

================
Check output
================

``index.html`` should be created, linking to certain representative plots for each test. (see test/main_test/index.html for an example)

Output will be located in subfolders in your test folder (e.g. “my_test/SingleFit…”) and should also be available as links in index.html

