########################################################################
# This is the setup script for the MAterials Simulation Toolkit
#   machine-learning module (MASTML)
# Maintainer: Tam Mayeshiba
# Last updated: 2017-05-23
########################################################################
from setuptools.command.install import install
from setuptools import setup, find_packages
import sys
import os
import re

###Python version check
#print "Python version detected: %s" % sys.version_info
if sys.version_info[0] < 3:
    print "Python Version %d.%d.%d found" % (sys.version_info[0], sys.version_info[1], sys.version_info[2])
    print "Python version >= 3 needed!"
    sys.exit(0)

###Version load, adapted from http://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package/3619714#3619714
PKG = "MAST"
VERSIONFILE = os.path.join(PKG, "_version.py")
verstr = "unknown"
try:
    verstrline = open(VERSIONFILE, "rt").read()
except EnvironmentError:
    pass # Okay, there is no version file.
else:
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        verstr = mo.group(1)
    else:
        print "unable to find version in %s" % (VERSIONFILE,)
        raise RuntimeError("if %s.py exists, it is required to be well-formed" % (VERSIONFILE,))

setup(
        name="MASTML",
        packages=find_packages(),
        version=verstr,
        install_requires=["numpy>=1.11.2", 
                "scipy>=0.18.1", 
                "pandas>=0.19.2",
                "matplotlib>=1.5.3",
                "configobj>=5.0.6",
                #"validator",
                "matminer>=0.0.9",
                "scikit-learn>=0.18.1",
                "pymongo>=3.4.0",
                "pymatgen>=4.6.0",
                "PeakUtils>=1.0.3",
                "citrination-client",
                "mlxtend",
                ],
        author="MAST Development Team, University of Wisconsin-Madison Computational Materials Group",
        author_email="ddmorgan@wisc.edu",
        url="",
        license="Private",
        description="MAterials Simulation Toolkit - Machine Learning",
        long_description="MAterials Simulation Toolkit for Machine Learning (MASTML)",
        keywords=["MAST","materials","simulation","MASTML","machine learning"],
)


