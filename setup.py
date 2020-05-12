########################################################################
# This is the setup script for the MAterials Simulation Toolkit
#   machine-learning module (MAST-ML)
# Creator and Maintainer: UW-Madison MAST-ML Development Team
#
########################################################################

from __future__ import print_function
import sys
from setuptools import setup, find_packages

###Python version check
#print "Python version detected: %s" % sys.version_info
if sys.version_info[0] < 3:
    print('Python Version %d.%d.%d found' % (sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    print('Python version >= 3 needed!')
    sys.exit(0)

#print("Python version 3.6.6 is REQUIRED")
#print("Check with `python --version`")

# One of the techniques from https://packaging.python.org/guides/single-sourcing-package-version/
verstr = "unknown"
try:
    verstr = open("VERSION", "rt").read().strip()
except EnvironmentError:
    pass # Okay, there is no version file.

setup(
    name="mastml",
    packages=['mastml', 'mastml.legos', 'mastml.magpie', 'mastml.tests.conf', 'mastml.tests.csv'],
    package_data={'mastml.magpie': ["*.*"], 'mastml.tests': ["*.*"], 'mastml.tests.conf' : ["example_input.conf", "MASTML_fullinputfile.conf"], 'mastml.tests.csv' : ["example_data.csv"]},
    include_package_data = True,
    version=verstr,
    install_requires=[
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
        "dlhub_sdk",
        "dominate>=2.3.5",
        "duecredit",
        "et_xmlfile",
        "folium>=0.2.1",
        "forestci>=0.3",
        "future>=0.16.0",
        "globus_nexus_client",
        "globus_sdk",
        "httplib2",
        "idna>=2.7",
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
        "psutil>=5.5.1",
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
        "requests==2.23.0",
        "retrying",
        "ruamel.yaml>=0.15.42",
        "scikit-learn==0.20.3",
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
        #"xgboost", #Note that sometimes xgboost has issues installing. Comment out this line if install fails on xgboost pip install
        "xlrd",
        "zipp"],
    author="MAST Development Team, University of Wisconsin-Madison Computational Materials Group",
    author_email="ddmorgan@wisc.edu",
    url="https://github.com/uw-cmg/MAST-ML",
    license="MIT",
    description="MAterials Simulation Toolkit - Machine Learning",
    long_description="MAterials Simulation Toolkit for Machine Learning (MASTML)",
    keywords=["MAST","materials","simulation","MASTML","machine learning"],
)
