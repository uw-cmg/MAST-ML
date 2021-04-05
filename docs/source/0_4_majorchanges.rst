*****************************
MAST-ML version 3.x
*****************************

===========================
New changes to MAST-ML
===========================

As of MAST-ML version update 3.x and going forward, there are some significant changes to MAST-ML for users to
be aware of:

MAST-ML major updates:

    - MAST-ML no longer uses an input file. The core functionality and workflow of MAST-ML has been rewritten to be more conducive to use in a Jupyter notebook environment. This major change has made the code more modular and transparent, and we believe more intuitive and easier to use in a research setting. The last version of MAST-ML to have input file support was version 2.0.20 on PyPi.

    - Each component of MAST-ML can be run in a Jupyter notebook environment, either locally or through a cloud-based service like Google Colab. As a result, we have completely reworked our use-case tutorials and examples. All of these MAST-ML tutorials are in the form of Jupyter notebooks and can be found in the mastml/examples folder on Github.

    - An active part of improving MAST-ML is to provide an automated, quantitative analysis of model domain assessement and model prediction uncertainty quantification (UQ). Version 3.x of MAST-ML includes more detailed implementation of model UQ using new and established techniques.

MAST-ML minor updates:

    - More straightforward implementation of left-out test data, both designated manually by the user and via nested cross validation.

    - Improved integration of feature generation schemes in complimentary materials informatics packages, particularly matminer.

    - Improved data import schema based on locally-stored files, and via downloading data hosted on databases including Figshare, matminer, Materials Data Facility, and Foundry.

    - Support for generalized ensemble models with user-specified choice of model type to use as the weak learner, including support for ensembles of Keras-based neural networks.