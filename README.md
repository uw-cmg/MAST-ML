# Materials Simulation Toolkit for Machine Learning (MAST-ML)

MAST-ML is an open-source Python package designed to broaden and accelerate the use of machine learning in materials science research

[![Actions Status](https://github.com/uw-cmg/MAST-ML/workflows/python_package/badge.svg)](https://github.com/uw-cmg/MAST-ML/actions)

<a href='https://mastmldocs.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/mastmldocs/badge/?version=latest' alt='Documentation Status' />
</a>

## MAST-ML version 3.x Major Updates
* MAST-ML no longer uses an input file. The core functionality and workflow of MAST-ML has been rewritten to be more conducive to use in a Jupyter notebook environment. This major change has made the code more modular and transparent, and we believe more intuitive and easier to use in a research setting. The last version of MAST-ML to have input file support was version 2.0.20 on PyPi.

* Each component of MAST-ML can be run in a Jupyter notebook environment, either locally or through a cloud-based service like Google Colab. As a result, we have completely reworked our use-case tutorials and examples. All of these MAST-ML tutorials are in the form of Jupyter notebooks and can be found in the mastml/examples folder on Github.

* An active part of improving MAST-ML is to provide an automated, quantitative analysis of model domain assessement and model prediction uncertainty quantification (UQ). Version 3.x of MAST-ML includes more detailed implementation of model UQ using new and established techniques.

## Contributors

University of Wisconsin-Madison Computational Materials Group:
* Prof. Dane Morgan
* Dr. Ryan Jacobs
* Dr. Tam Mayeshiba
* Ben Afflerbach
* Dr. Henry Wu

University of Kentucky contributors:
* Luke Harold Miles
* Robert Max Williams
* Matthew Turner
* Prof. Raphael Finkel

## MAST-ML documentation
* An overview of code documentation and tutorials for getting started with MAST-ML can be found here:

```
https://mastmldocs.readthedocs.io/en/latest/
```

## Funding

This work was and is funded by the National Science Foundation (NSF) SI2 award No. 1148011 and DMREF award number DMR-1332851


## Citing MAST-ML


If you find MAST-ML useful, please cite the following publication:

Jacobs, R., Mayeshiba, T., Afflerbach, B., Miles, L., Williams, M., Turner, M., Finkel, R., Morgan, D.,
"The Materials Simulation Toolkit for Machine Learning (MAST-ML): An automated open source toolkit to accelerate data-
driven materials research", Computational Materials Science 175 (2020), 109544. https://doi.org/10.1016/j.commatsci.2020.109544



## Code Repository

MAST-ML is freely available via Github: 

```
https://github.com/uw-cmg/MAST-ML

git clone https://github.com/uw-cmg/MAST-ML
```

MAST-ML can also be installed via pip:

```
pip install mastml
```
