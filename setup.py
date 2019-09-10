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

#print("Python version 3.6.6 is REQUIRED")
#print("Check with `python --version`")

# One of the techniques from https://packaging.python.org/guides/single-sourcing-package-version/
verstr = "unknown"
try:
    verstr = open("VERSION", "rt").read().strip()
except EnvironmentError:
    pass # Okay, there is no version file.

setup(
    name="mastml", # TODO  should this be MAST-ML?
    packages=find_packages(),
    version=verstr,
    install_requires=[
        "certifi==2018.4.16",
        "chardet==3.0.4",
        "citrination-client==4.6.0",
        "configobj==5.0.6",
        "cycler==0.10.0",
        "Cython==0.29.13",
        "decorator==4.3.0",
        "dominate==2.3.5",
        "forestci==0.3",
        "idna==2.7",
        "ipython-genutils==0.2.0",
        "jsonschema==2.6.0",
        "jupyter-core==4.4.0",
        "keras==2.2.4",
        "kiwisolver==1.0.1",
        "matminer==0.5.5",
        "matplotlib==2.2.2",
        "mlxtend==0.12.0",
        "monty==1.0.3",
        "mpmath==1.0.0",
        "nbformat==4.4.0",
        "nose==1.3.7",
        "numpy==1.16.2",
        "openpyxl",
        "palettable==3.1.1",
        "pandas==0.24.2",
        "PyDispatcher==2.0.5",
        "pymatgen==2019.1.24",
        "pyparsing==2.2.0",
        "pypif==2.1.0",
        "pytest==5.0.1",
        "python-dateutil==2.7.3",
        "pytz==2018.5",
        "requests==2.20.0",
        "ruamel.yaml==0.15.42",
        "scikit-learn==0.20.3",
        "scikit-optimize==0.5.2",
        "scipy==1.2.1",
        "six==1.11.0",
        "spglib==1.10.3.65",
        "sympy==1.2",
        "tabulate==0.8.2",
        "tensorflow==1.13.1",
        "traitlets==4.3.2",
        "urllib3>=1.24.2",
        "xgboost",
        "xlrd"],
    author="MAST Development Team, University of Wisconsin-Madison Computational Materials Group",
    author_email="ddmorgan@wisc.edu",
    url="https://github.com/uw-cmg/MAST-ML",
    license="MIT",
    description="MAterials Simulation Toolkit - Machine Learning",
    long_description="MAterials Simulation Toolkit for Machine Learning (MASTML)",
    keywords=["MAST","materials","simulation","MASTML","machine learning"],
)
