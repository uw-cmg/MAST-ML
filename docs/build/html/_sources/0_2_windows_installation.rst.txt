***************************
Windows installation
***************************

==================
Install Python3
==================

Install Python 3: for easier installation of numpy and scipy dependencies,
download anaconda from https://www.continuum.io/downloads

---------------------------------
Create a conda environment
---------------------------------

From the Anaconda Navigator, go to Environments and create a new environment
Select python version **3.5** (3.6 may work with later version of the pymatgen
conda package)

Under "Channels", along with defaults channel, "Add" the "materials" channel.
The Channels list should now read::

    defaults
    materials

(may be the "matsci" channel instead of the "materials" channel;
this channel is used to install pymatgen)

Search for and select the following packages::

    numpy==1.15.0rc2
    scipy==1.1.0
    matplotlib==2.2.2
    nbformat==4.4.0
    pandas==0.23.3
    configobj==5.0.6
    scikit-learn==0.19.1
    pymatgen==2018.6.27

Click on "Apply" at the bottom to install all of the packages.

To get a terminal, from the Anaconda Navigator, go to
Environments, make sure the environment is selected, press the green arrow
button, and select Open terminal.

From the terminal, pip install the remaining dependencies::

    certifi==2018.4.16
    chardet==3.0.
    citrination-client==2.1.0
    cycler==0.10.0
    decorator==4.3.0
    dominate==2.3.1
    idna==2.7
    ipython-genutils==0.2.0
    jsonschema==2.6.0
    jupyter-core==4.4.0
    kiwisolver==1.0.1
    mlxtend==0.12.0
    monty==1.0.3
    mpmath==1.0.0
    nose==1.3.7
    palettable==3.1.1
    PyDispatcher==2.0.5
    pyparsing==2.2.0
    pypif==2.1.0
    python-dateutil==2.7.3
    pytz==2018.5
    requests==2.19.1
    ruamel.yaml==0.15.42
    six==1.11.0
    spglib==1.10.3.65
    sympy==1.2
    tabulate==0.8.2
    traitlets==4.3.2
    urllib3==1.23

-------------------------------------------------
Set up the Spyder IDE and Jupyter notebooks
-------------------------------------------------
From the Anaconda Navigator, go to Home
With the newly created environment selected, click on "Install" below Jupyter.
Click on "Install" below Spyder.

Once the MASTML has been run and has created a jupyter notebook (run MASTML
from a location inside the anaconda environment, so that the notebook will
also be inside the environment tree), from the Anaconda Navigator, go to
Environments, make sure the environment is selected, press the green arrow
button, and select Open jupyter notebook.

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

-------------------------
Imports that don’t work
-------------------------
First try anaconda install, and if that gives errors try pip install
Example: conda install numpy , or pip install numpy
Put the path to the installed MAST-ML folder in your PYTHONPATH if it isn’t already