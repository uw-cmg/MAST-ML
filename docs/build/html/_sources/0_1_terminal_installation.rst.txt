******************************************************************
Terminal installation (Linux or linux-like terminal on Mac)
******************************************************************

===============
Install Python3
===============

Install Python 3: for easier installation of numpy and scipy dependencies,
download Anaconda from https://www.continuum.io/downloads

---------------------------------
Create a conda environment
---------------------------------

Create an environment::

    conda create --name TEST_ML numpy scipy matplotlib pandas pymongo configobj nbformat scikit-learn
    source activate TEST_ML
    conda install --channel matsci pymatgen

(may also be the "materials" channel for pymatgen)
(may also need conda create --name TEST_ML python=3.5 numpy scipy...etc. if pymatgen requires python 3.5)

-----------------------------
Set up Juptyer notebooks
-----------------------------
There is no separate setup for Jupyter notebooks necessary;
once MASTML has been run and created a notebook, then in the terminal,
navigate to a directory housing the notebook and type::

    jupyter notebook

and a browser window with the notebook should appear.

=====================================
Install the MAST-ml-private package
=====================================

Git clone the repository, for example::

    git clone ssh://git@github.com/uw-cmg/MAST-ml-private
    OR
    git clone ssh://git@github.com/uw-cmg/MAST-ml-private --branch dev

Clone from “master” unless instructed specifically to use another branch.
Ask for access if you cannot find this code.

Check status.github.com for issues if you believe github may be malfunctioning

Run::

    python setup.py install

(inside the MAST-ml-private folder) (might not work)
If “SyntaxError: Invalid syntax” occurs at lines 16, 17, or 34 try wrapping prints in () for windows [e.g. print ("Python Version %d.%d.%d found" % (sys.version_info[0], sys.version_info[1], sys.version_info[2])) ]

If setup.py didn’t work, dependencies you need are::

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
    mlxtend
    nbformat
    citrination_client==2.1.0

-------------------------
imports that don’t work
-------------------------
First try anaconda install, and if that gives errors try pip install
Example: conda install numpy , or pip install numpy
Put MAST-ml-private’s folder in your PYTHONPATH if it isn’t already
Adding to PYTHONPATH and getting it to stay there
If you get a missing import error, go ahead and install any other missing packages and submit a github ticket.