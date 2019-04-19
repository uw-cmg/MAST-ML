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
    name="mastml", # TODO  should this be MAST-ML?
    packages=find_packages(),
    version=verstr,
    install_requires=[
        'certifi', 'chardet', 'citrination-client', 'configobj', 'cycler',
        'decorator', 'dominate', 'idna', 'ipython-genutils', 'jsonschema',
        'jupyter-core', 'kiwisolver', 'matplotlib', 'mlxtend', 'monty', 'mpmath',
        'nbformat', 'nose', 'numpy', 'palettable', 'pandas', 'PyDispatcher',
        'pymatgen', 'pyparsing', 'pypif', 'python-dateutil', 'pytz', 'requests',
        'ruamel.yaml', 'scikit-learn', 'scipy', 'six', 'spglib', 'sympy', 'tabulate',
        'traitlets', 'urllib3'],
    author="MAST Development Team, University of Wisconsin-Madison Computational Materials Group",
    author_email="ddmorgan@wisc.edu",
    url="https://github.com/uw-cmg/MAST-ML",
    license="MIT",
    description="MAterials Simulation Toolkit - Machine Learning",
    long_description="MAterials Simulation Toolkit for Machine Learning (MASTML)",
    keywords=["MAST","materials","simulation","MASTML","machine learning"],
)
