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

    conda create --name MAST_ML python=3.7
    conda activate MAST_ML
    pip install mastml

-----------------------------
Set up Juptyer notebooks
-----------------------------
There is no separate setup for Jupyter notebooks necessary;
once MASTML has been run and created a notebook, then in the terminal,
navigate to a directory housing the notebook and type::

    jupyter notebook

and a browser window with the notebook should appear.


=====================================
Install the MAST-ML package
=====================================

Pip install MAST-ML from PyPi::

    pip install mastml

Alternatively, git clone the Github repository, for example::

    git clone https://github.com/uw-cmg/MAST-ML

Clone from “master” unless instructed specifically to use another branch.
Ask for access if you cannot find this code.

Check status.github.com for issues if you believe github may be malfunctioning

Run::

    python setup.py install

-------------------------
Imports that don’t work
-------------------------
First try anaconda install, and if that gives errors try pip install
Example: conda install numpy , or pip install numpy
Put the path to the installed MAST-ML folder in your PYTHONPATH if it isn’t already