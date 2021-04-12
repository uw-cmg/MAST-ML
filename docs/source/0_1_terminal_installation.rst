**************************************************************************
Terminal installation (Linux or linux-like terminal environment e.g. Mac)
**************************************************************************

This documentation provides a few ways to install MAST-ML. If you don't have python 3 on your system, begin
with the section "Install Python3". If you already have python 3 installed,

===============
Install Python3
===============

Install Python 3: for easier installation of numpy and scipy dependencies,
download Anaconda from https://www.continuum.io/downloads

-----------------------------------------------
Create a conda environment (if using Anaconda)
-----------------------------------------------

Create an anaconda python environment::

    conda create --name MAST_ML_env python=3.7
    conda activate MAST_ML_env

-------------------------------------------------------
Create a virtualenv environment (if not using Anaconda)
-------------------------------------------------------

Create a virtualenv environment::

    python3 -m venv MAST_ML_env
    source MAST_ML_env/bin/activate


=====================================
Install the MAST-ML package via PyPi
=====================================

Pip install MAST-ML from PyPi::

    pip install mastml


=====================================
Install the MAST-ML package via Git
=====================================

As an alternative to PyPi, you can git clone the Github repository, for example::

    git clone --single-branch --branch master https://github.com/uw-cmg/MAST-ML

Once the branch is downloaded, install the needed dependencies with::

    pip install -r MAST-ML/requirements.txt

Note that MAST-ML will need to be imported from within the MAST-ML directory as mastml is not
located in the usual spot where python looks for imported packages.

-----------------------------
Set up Juptyer notebooks
-----------------------------
There is no separate setup for Jupyter notebooks necessary;
once MAST-ML has been run and created a notebook, then in the terminal,
navigate to a directory housing the notebook and type::

    jupyter notebook

and a browser window with the notebook should appear.

-------------------------
Imports that don’t work
-------------------------
First try anaconda install, and if that gives errors try pip install
Example: conda install numpy , or pip install numpy
Put the path to the installed MAST-ML folder in your PYTHONPATH if it isn’t already