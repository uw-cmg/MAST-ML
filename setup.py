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
verstr = "3.1.6"
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
    install_requires=["citrination-client", 
                      "dlhub_sdk", 
                      "foundry-ml", 
                      "globus_nexus_client", 
                      "globus_sdk", "matminer", 
                      "matplotlib", 
                      "mdf_forge", 
                      "mdf-toolbox", 
                      "numpy", 
                      "openpyxl", 
                      "pandas", 
                      "pathos", 
                      "pymatgen", 
                      "pyyaml>=5.4.1", 
                      "scikit-learn==1.1.3", 
                      "scikit-optimize", 
                      "scikit-learn-extra", 
                      "scipy", 
                      "shap", 
                      "sphinx-automodapi", 
                      "statsmodels"],
    #install_requires=[
    #    "citrination-client",
    #    "dlhub_sdk",
    #    "globus_nexus_client",
    #    "globus_sdk",
    #    "matminer",
    #    "matplotlib",
    #    "mdf_forge",
    #    "mdf-toolbox",
    #    "mlxtend",
    #    "numpy",
    #    "openpyxl",
    #    "pandas",
    #    "pymatgen==2021.3.9",
    #    "scikit-learn>=0.24.1",
    #    "scikit-learn-extra",
    #    "scikit-optimize",
    #    "shap",
    #    "scipy",
    #    "sphinx-automodapi",
    #    "pathos",
    #    ],
    author="MAST Development Team, University of Wisconsin-Madison Computational Materials Group",
    author_email="ddmorgan@wisc.edu",
    url="https://github.com/uw-cmg/MAST-ML",
    license="MIT",
    description="MAterials Simulation Toolkit - Machine Learning",
    long_description="MAterials Simulation Toolkit for Machine Learning (MASTML)",
    keywords=["MAST","materials","simulation","MASTML","machine learning"],
)
