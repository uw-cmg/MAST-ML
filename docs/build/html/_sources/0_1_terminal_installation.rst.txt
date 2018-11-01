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
Install the MAST-ML package
=====================================

Git clone the repository, for example::

    git clone ssh://git@github.com/uw-cmg/MAST-ML
    OR
    git clone ssh://git@github.com/uw-cmg/MAST-ML --branch master

Clone from “master” unless instructed specifically to use another branch.
Ask for access if you cannot find this code.

Check status.github.com for issues if you believe github may be malfunctioning

Run::

    python setup.py install

If setup.py didn’t work, manually pip install the following dependencies:
Dependencies::

    certifi==2018.4.16,
    chardet==3.0.,
    citrination-client==2.1.0,
    configobj==5.0.6,
    cycler==0.10.0,
    decorator==4.3.0,
    dominate==2.3.1,
    idna==2.7,
    ipython-genutils==0.2.0,
    jsonschema==2.6.0,
    jupyter-core==4.4.0,
    kiwisolver==1.0.1,
    matplotlib==2.2.2,
    mlxtend==0.12.0,
    monty==1.0.3,
    mpmath==1.0.0,
    nbformat==4.4.0,
    nose==1.3.7,
    numpy==1.15.0rc2,
    palettable==3.1.1,
    pandas==0.23.3,
    PyDispatcher==2.0.5,
    pymatgen==2018.6.27,
    pyparsing==2.2.0,
    pypif==2.1.0,
    python-dateutil==2.7.3,
    pytz==2018.5,
    requests==2.19.1,
    ruamel.yaml==0.15.42,
    scikit-learn==0.19.1,
    scipy==1.1.0,
    six==1.11.0,
    spglib==1.10.3.65,
    sympy==1.2,
    tabulate==0.8.2,
    traitlets==4.3.2,
    urllib3==1.23,

-------------------------
Imports that don’t work
-------------------------
First try anaconda install, and if that gives errors try pip install
Example: conda install numpy , or pip install numpy
Put the path to the installed MAST-ML folder in your PYTHONPATH if it isn’t already
