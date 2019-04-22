########################################################################
# This is the setup script for the MAterials Simulation Toolkit
#   machine-learning module (MASTML)
# Creator: Tam Mayeshiba
# Maintainer: Robert Max Williams
# Last updated: 2018-06-20
#  _________________________________
# / No one knows where the original \
# \ setup.py came from.             /
#  ---------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
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

print("Python version 3.6.6 is REQUIRED")
print("Check with `python --version`")

# One of the techniques from https://packaging.python.org/guides/single-sourcing-package-version/
verstr = "unknown"
try:
    verstr = open("VERSION", "rt").read().strip()
except EnvironmentError:
    pass # Okay, there is no version file.

setup(
    name="mastml", 
    packages=['mastml', 'mastml.legos', 'mastml.search', 'mastml.magpie'],
    package_data={'mastml.magpie': ['*.table']},
    version=verstr,
    install_requires=[
        "citrination-client>=4.6.0",
        "configobj>=5.0.6",
        "dominate>=2.3.1",
        "keras>=2.2.2",
        "matminer>=0.4.3",
        "matplotlib>=2.2.2",
        "mlxtend>=0.12.0",
        "nbformat>=4.4.0",
        "numpy>=1.15.0",
        "pandas>=0.23.3",
        "pymatgen>=2018.6.27",
        "scikit-learn>=0.19.1",
        "scikit-optimize>=0.5.2",
        "scipy>=1.1.0"
        ],
    author="MAST Development Team, University of Wisconsin-Madison Computational Materials Group",
    author_email="ddmorgan@wisc.edu",
    url="https://github.com/uw-cmg/MAST-ML",
    license="MIT",
    description="MAterials Simulation Toolkit - Machine Learning",
    long_description="MAterials Simulation Toolkit for Machine Learning (MASTML)",
    keywords=["MAST","materials","simulation","MASTML","machine learning"],
)
