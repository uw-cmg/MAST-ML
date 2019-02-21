===============
Install Python3
===============

Install Python 3: for easier installation of numpy and scipy dependencies,
download Anaconda from https://www.continuum.io/downloads

---------------------------------
Create a conda environment
---------------------------------

Create an environment::

    conda create --name TEST_ML numpy scipy matplotlib pandas pymongo configobj nbformat scikit-learn
    source activate TEST_ML
    conda install --channel matsci pymatgen

(may also be the "materials" channel for pymatgen)
(may also need conda create --name TEST_ML python=3.5 numpy scipy...etc. if pymatgen requires python 3.5)

-----------------------------
Set up Juptyer notebooks
-----------------------------
There is no separate setup for Jupyter notebooks necessary;
once MASTML has been run and created a notebook, then in the terminal,
navigate to a directory housing the notebook and type::

    jupyter notebook

and a browser window with the notebook should appear.