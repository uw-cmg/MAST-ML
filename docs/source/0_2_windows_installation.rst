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

        "aflow",
        "atomicwrites",
        "attrs>=17.4.0",
        "certifi>=2018.4.16",
        "chardet>=3.0.4",
        "citeproc-py>=0.4",
        "citrination-client>=4.6.0",
        "configobj>=5.0.6",
        "cryptography",
        "cycler>=0.10.0",
        "Cython>=0.29.13",
        "decorator>=4.3.0",
        "dominate>=2.3.5",
        "duecredit",
        "et_xmlfile",
        "forestci>=0.3",
        "future>=0.16.0",
        "globus_nexus_client",
        "globus_sdk",
        "httplib2",
        "idna>=2.7",
        "importlib-metadata>=0.12",
        "ipython-genutils>=0.2.0",
        "jdcal",
        "jsonschema>=4.4.0",
        "jwt",
        "keras>=2.2.4",
        "kiwisolver>=1.0.1",
        "matminer>=0.5.5",
        "matplotlib>=2.2.2",
        "mdf_forge>=0.6.1J",
        "mdf-toolbox>=0.4.7",
        "mlxtend>=0.12.0",
        "monty>=4.4.0",
        "networkx>=2.1",
        "nose>=1.3.7",
        "numpy>=1.16.2",
        "openpyxl",
        "packaging",
        "palettable>=3.1.1",
        "pandas>=0.24.2",
        "pint>=0.8.1",
        "plotly",
        "pluggy>=0.12",
        "py>=1.5.0",
        "PyDispatcher>=2.0.5",
        "pymatgen>=2019.1.24",
        "pymongo>=3.6.1",
        "pyparsing>=2.2.0",
        "pypif>=2.1.0",
        "pytest>=5.0.1",
        "python-dateutil>=2.7.3",
        "pytz>=2018.5",
        "pyyaml>=4.2b1",
        "requests>=2.20.0",
        "retrying",
        "ruamel.yaml>=0.15.42",
        "scikit-learn>=0.20.3",
        "scikit-optimize>=0.5.2",
        "scipy>=1.2.1",
        "six>=1.11.0",
        "spglib>=1.10.3.65",
        "sympy>=1.2",
        "tabulate>=0.8.2",
        "tensorflow>=1.13.1",
        "tqdm>=4.23.1",
        "traitlets>=4.3.2",
        "urllib3>=1.24.2",
        "wcwidth",
        "xgboost",
        "xlrd",
        "zipp"
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