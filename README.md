# Materials Simulation Toolkit for Machine Learning (MAST-ML)

MAST-ML is an open-source Python package designed to broaden and accelerate the use of machine learning in materials science research

<img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/uw-cmg/MAST-ML">

<img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/mastml">

<a href='https://mastmldocs.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/mastmldocs/badge/?version=latest' alt='Documentation Status' />
</a>

## Run example notebooks in Google Colab:

* Tutorial 1: Getting Started with MAST-ML:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/MAST-ML/blob/master/examples/MASTML_Tutorial_1_GettingStarted.ipynb)

* Tutorial 2: Data Import and Cleaning:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/MAST-ML/blob/master/examples/MASTML_Tutorial_2_DataImport.ipynb)

* Tutorial 3: Feature Engineering:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/MAST-ML/blob/master/examples/MASTML_Tutorial_3_FeatureEngineering.ipynb)

* Tutorial 4: Models and Data Splitting Tests:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/MAST-ML/blob/master/examples/MASTML_Tutorial_4_Models_and_Tests.ipynb)

* Tutorial 5: Left out data, nested cross validation, and optimized models:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/MAST-ML/blob/master/examples/MASTML_Tutorial_5_NestedCV_and_OptimizedModels.ipynb)

* Tutorial 6: Model error analysis and uncertainty quantification:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/MAST-ML/blob/master/examples/MASTML_Tutorial_6_ErrorAnalysis_UncertaintyQuantification.ipynb)

* Tutorial 7: Model predictions with guide rails:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/MAST-ML/blob/master/examples/MASTML_Tutorial_7_ModelPredictions_with_Guide_Rails.ipynb)

## Contributors

University of Wisconsin-Madison Computational Materials Group:
* Prof. Dane Morgan
* Dr. Ryan Jacobs
* Lane Schultz
* Dr. Tam Mayeshiba
* Dr. Ben Afflerbach
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

This work was funded by the National Science Foundation (NSF) SI2 award number 1148011

This work was funded by the National Science Foundation (NSF) DMREF award number DMR-1332851

This work was funded by the National Science Foundation (NSF) CSSI award number 1931298


## Citing MAST-ML, Error bar and Domain of Applicability Tools

If you find MAST-ML useful, please cite the following publication:

Jacobs, R., Mayeshiba, T., Afflerbach, B., Miles, L., Williams, M., Turner, M., Finkel, R., Morgan, D.,
"The Materials Simulation Toolkit for Machine Learning (MAST-ML): An automated open source toolkit to accelerate data-
driven materials research", Computational Materials Science 175 (2020), 109544. https://doi.org/10.1016/j.commatsci.2020.109544

If you find the uncertainty quantification (error bar) approaches useful, please cite the following publication:

Palmer, G., Du, S., Politowicz, A., Emory, J. P., Yang, X., Gautam, A., Gupta, G., Li, Z., Jacobs, R., Morgan, D.,
"Calibration after bootstrap for accurate uncertainty quantification in regression models", npj Computational Materials 8 115 (2022). https://doi.org/10.1038/s41524-022-00794-8

If you find the domain of applicability approaches useful, please cite the following publication:

Schultz, L. E., Wang, Y., Jacobs, R., Morgan, D., "Determining Domain of Machine Learning Models using Kernel Density Estimates: Applications in Materials Property Prediction", arXiv (2024). 
https://doi.org/10.48550/arXiv.2406.05143 

## Installation

MAST-ML can be installed via pip:

```
pip install mastml
```

Clone from Github: 

```
git clone https://github.com/uw-cmg/MAST-ML
```

# Changelog

## MAST-ML version 3.2.x Major Updates from April 2024
* Integration of domain of applicability approach using kernel density estimates based on MADML package: https://github.com/leschultz/materials_application_domain_machine_learning

* Refinement of tutorials, addition of new tutorial for domains of applicability

* Updates to plotting routines for error bar analysis

* Many small bug fixes and updates to conform to updated versions of package dependencies

## MAST-ML version 3.1.x Major Updates from July 2022
* Refinement of tutorials, addition of Tutorial 7, Colab links as badges added for easier use.

* mastml_predictor module added to help streamline making predictions (with option to include error bars) on new test data.

* Basic parallelization added, which is especially useful for speeding up nested CV runs with many inner splits.

* EnsembleModel now handles ensembles of GPR and XGBoost models.

* Numerous improvements to plotting, including new plots (QQ plot), better axis handling and error bars (RvE plot), plotting and stats separated per group if groups are specified.

* Improvements to feature selection methods. EnsembleModelFeatureSelector includes dummy feature references, added SHAP-based selector

* Added assessment of baseline tests like comparing metrics to predicting the data average or permuted data test

* Many miscellaneous bug fixes.


## MAST-ML version 3.0.x Major Updates from July 2021
* MAST-ML no longer uses an input file. The core functionality and workflow of MAST-ML has been rewritten to be more conducive to use in a Jupyter notebook environment. This major change has made the code more modular and transparent, and we believe more intuitive and easier to use in a research setting. The last version of MAST-ML to have input file support was version 2.0.20 on PyPi.

* Each component of MAST-ML can be run in a Jupyter notebook environment, either locally or through a cloud-based service like Google Colab. As a result, we have completely reworked our use-case tutorials and examples. All of these MAST-ML tutorials are in the form of Jupyter notebooks and can be found in the mastml/examples folder on Github.

* An active part of improving MAST-ML is to provide an automated, quantitative analysis of model domain assessement and model prediction uncertainty quantification (UQ). Version 3.x of MAST-ML includes more detailed implementation of model UQ using new and established techniques.
