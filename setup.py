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
    packages=['mastml', 'mastml.magpie', 'mastml.tests.conf', 'mastml.tests.csv'],
    package_data={'mastml.magpie': ["*.*"], 'mastml.tests': ["*.*"], 'mastml.tests.conf' : ["example_input.conf", "MASTML_fullinputfile.conf"], 'mastml.tests.csv' : ["example_data.xlsx"]},
    include_package_data = True,
    version=verstr,
    install_requires=[
        "citrination-client", #>=4.6.0
        "dlhub_sdk", #>=0.9.3
        "globus_nexus_client",
        "globus_sdk",
        "matminer", #>=0.6.3
        "matplotlib", #>=3.1.1
        "mdf_forge", #>=0.6.1J
        "mdf-toolbox", #>=0.5.5
        "mlxtend",
        "numpy", #>=1.16.2
        "pandas", #>=0.24.2
        "pymatgen", #>=2019.1.24
        "scikit-learn", #>=0.23.1
        "scikit-optimize", #>=0.5.2
        "scipy", #>=1.2.1,
        ],
    author="MAST Development Team, University of Wisconsin-Madison Computational Materials Group",
    author_email="ddmorgan@wisc.edu",
    url="https://github.com/uw-cmg/MAST-ML",
    license="MIT",
    description="MAterials Simulation Toolkit - Machine Learning",
    long_description="MAterials Simulation Toolkit for Machine Learning (MASTML)",
    keywords=["MAST","materials","simulation","MASTML","machine learning"],
)
