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
Select python version **3.6**

Under "Channels", along with defaults channel, "Add" the "materials" channel.
The Channels list should now read::

    defaults
    materials

(may be the "matsci" channel instead of the "materials" channel;
this channel is used to install pymatgen)

Search for and select the following packages::

    numpy==1.18.4
    scipy==1.4.1
    matplotlib==3.2.1
    nbformat==5.0.6
    pandas==1.0.4
    configobj==5.0.6
    scikit-learn==0.22.2
    pymatgen==2020.4.29

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
        "folium>=0.2.1",
        "forestci>=0.3",
        "future>=0.16.0",
        "globus_nexus_client",
        "globus_sdk",
        "httplib2",
        "idna==2.7",
        "imageio>=2.3.0",
        "imgaug>=0.2.5",
        "importlib-metadata>=0.12",
        "ipython-genutils>=0.2.0",
        "jdcal",
        "jsonschema>=3.0.2",
        "jwt",
        "keras>=2.2.4",
        "kiwisolver>=1.0.1",
        "matminer>=0.6.3",
        "matplotlib>=3.1.1",
        "mdf_forge>=0.6.1J",
        "mdf-toolbox>=0.4.7",
        "mlxtend==0.12.0",
        "monty>=1.0.2",
        "nbformat",
        "networkx>=2.1",
        "nose>=1.3.7",
        "numpy>=1.16.2",
        "openpyxl",
        "packaging",
        "palettable>=3.1.1",
        "pandas>=0.24.2",
        "pint>=0.8.1",
        "plotly>=4.5.0",
        "pluggy>=0.12",
        "psutils>=5.5.1",
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
        "scikit-learn==0.22.2",
        "scikit-optimize>=0.5.2",
        "scipy>=1.2.1",
        "six>=1.11.0",
        "spglib>=1.10.3.65",
        "sympy>=1.2",
        "tabulate>=0.8.2",
        "tensorflow>=1.13.1",
        "tqdm>=4.23.1",
        "traitlets>=4.3.2",
        "urllib3<1.25,>=1.24.2",
        "wcwidth",
        "xlrd",
        "zipp"

MAST-ML also needs the packages *dlhub* and *xgboost* but the dependencies do not play nice with the other packages listed above and/or Windows setups require extra steps to install these packages. Please try to manually install these packages using `pip install <package name>`.

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

========================================================================
Windows 10 install: step-by-step guide (credit Joe Kern)
========================================================================

First, figure out if your computer is 32 or 64-bit. Type “system information” in your search bar. Look at system type. x86 is a 32-bit computer, x64 is a 64-bit.

Second, download an environment manager. Environments are directories in your computer that store dependencies. For instance, one program you run might be dependent on version 1.0 of another program x. However, another program you have might be dependent on version 2.0 of program x. Having multiple environments allows you utilize both programs and dependencies on your computer. I will recommend you download anaconda, not because it is the best, but because it is an environment manager I know how to get working with MAST-ML. Feel free to experiment with other managers. Download the Python 3.7 version at https://www.anaconda.com/distribution/, just follow the installation instructions. Pick the graphical installer that corresponds with your computer system (64 bit or 32 bit).

Third, download Visual studio. Some of the MAST-ML dependencies require C++ distributables in order to run. Visual Studio Code is a code editor made for Windows 10. The dependencies for MAST-ML will look in the Visual Studio Code folder for these C++ distributables when they download. There may be another way to download these these C++ distributables without Visual Studio Code, but I am not sure how to do that. Go here to download https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017

Fourth, download Visual Studio with C++ build tools and restart the computer

Fifth, Open anaconda navigator. Click Environments and create at the bottom. Name it MASTML and make it Python 3.6. DO NOT MAKE IT Python 3.7 or Python version 2.6 or 2.7. Some dependencies do not work with those other version.

Sixth, click the arrow next to your environment name and open a command shell. In the command line type “pip install “ and then copy paste the dependency names from the dependency file into your command prompt.

Seventh, test if MAST-ML runs. There are multiple ways to do this, but I will outline one. Navigate to your MAST-ML folder in the command prompt. To do this, you need to know the command ‘cd’. Typing ‘cd’ will let you change the directory you command prompt is operating in. In order to navigate to your mast-ml folder, right click the folder and click properties. Copy the location and in the command prompt type ‘cd’ and paste the location after. Add a ‘\Mast-ml’ or whatever your folder is called to the end of the pasted value so you can get to mastml

Finally, copy paste python -m mastml.mastml_driver mastml/tests/conf/example_input.conf mastml/tests/csv/example_data.csv -o results/mastml_tutorial into your command prompt and run. If it all works, you’re good to go.
