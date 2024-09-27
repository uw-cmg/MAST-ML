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
if sys.version_info[0] < 3:
    print('Python Version %d.%d.%d found' % (sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    print('Python version >= 3.11 needed!')
    sys.exit(0)
if sys.version_info[0] >= 3:
    if sys.version_info[1] < 10:
        print('Python Version %d.%d.%d found' % (sys.version_info[0], sys.version_info[1], sys.version_info[2]))
        print('Python version >= 3.10 needed!')
        sys.exit(0)

# One of the techniques from https://packaging.python.org/guides/single-sourcing-package-version/
verstr = "3.2.0"

try:
    verstr = open("VERSION", "rt").read().strip()
except EnvironmentError:
    pass # Okay, there is no version file.

setup(
    name="mastml",
    packages=['mastml', 'mastml.magpie', 'mastml.tests.unit_tests', 'mastml.data'],
    package_data={'mastml.magpie': ["*.*"], 'mastml.tests.unit_tests': ["*.*"], 'mastml.data': ["*.*"]},
    include_package_data = True,
    version=verstr,

    install_requires=[
        "scikit-learn",
        "scikit-optimize",
        "citrination-client",
        "foundry-ml",
        #"globus_nexus_client",
        #"globus_sdk",
        "matminer",
        "matplotlib",
        "mdf_forge",
        "mdf-toolbox",
        "numpy",
        "openpyxl",
        "pandas",
        "pathos",
        "pykan==0.0.5"
        "pymatgen",
        "pyyaml",
        "scikit-learn-extra",
        "scipy",
        "shap",
        "sphinx-automodapi",
        "statsmodels",
        "madml",
        "udocker",
        "transfernet",
        "forestci"
        ],

    author="MAST Development Team, University of Wisconsin-Madison Computational Materials Group",
    author_email="ddmorgan@wisc.edu",
    url="https://github.com/uw-cmg/MAST-ML",
    license="MIT",
    description="MAterials Simulation Toolkit - Machine Learning",
    long_description="MAterials Simulation Toolkit for Machine Learning (MASTML)",
    keywords=["MAST","materials","simulation","MASTML","machine learning"],
)
