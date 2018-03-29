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

    numpy
    scipy
    matplotlib
    nbformat
    pandas
    configobj
    pymongo
    scikit-learn
    pymatgen
    
Click on "Apply" at the bottom to install all of the packages.

To get a terminal, from the Anaconda Navigator, go to
Environments, make sure the environment is selected, press the green arrow
button, and select Open terminal. 

From the terminal::

    pip install citrination-client==2.1.0
    pip install PeakUtils
    pip install mlxtend
    pip install validator

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
Install the MAST-ml-private package
=====================================

Install github (for example, the Github Desktop from desktop.github.com)

Clone the repository at /uw-cmg/MAST-ml-private
Clone from “master” or "dev"
Ask for access if you cannot find this code.

Check status.github.com for issues if you believe github may be malfunctioning

-------------------------
imports that don’t work 
-------------------------
First try anaconda install, and if that gives errors try pip install
Example: conda install numpy , or pip install numpy
Put MAST-ml-private’s folder in your PYTHONPATH if it isn’t already
Adding to PYTHONPATH and getting it to stay there
If you get a missing import error, go ahead and install any other missing packages and submit a github ticket.