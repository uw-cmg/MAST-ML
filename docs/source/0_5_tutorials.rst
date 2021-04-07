*****************************************************
Overview of MAST-ML tutorials and examples
*****************************************************

===========================
MAST-ML tutorials
===========================

There are numerous MAST-ML tutorial and example Jupyter notebooks. These notebooks
can be found in the mastml/examples folder. Here, a brief overview of the contents
of each tutorial is provided:

Tutorial 1: Getting Started (MASTML_Tutorial_1_GettingStarted.ipynb):
    In this notebook, we will perform a first, basic run where we:
        1. Import example data of Boston housing prices
        2. Define a data preprocessor to normalize the data
        3. Define a linear regression model and kernel ridge model to fit the data
        4. Evaluate each of our models with 5-fold cross validation

Tutorial 2: Data Import and Cleaning (MASTML_Tutorial_2_DataImport.ipynb):
    In this notebook, we will learn different ways to download and import data into a MAST-ML run.
        1. Import model datasets from scikit-learn
        2. Conduct different data cleaning methods
        3. Import and prepare a real dataset that is stored locally
        4. Download data from various materials databases

Tutorial 3: Feature Generation and Selection (MASTML_Tutorial_3_FeatureEngineering.ipynb):
    In this notebook, we will learn different ways to generate, preprocess, and select features:
        1. Generate features based on material composition
        2. Generate one-hot encoded features based on group labels
        3. Preprocess features to be normalized
        4. Select features using an ensemble model-based approach
        5. Select features using forward selection
        6. Generate learning curves using a basic feature selection approach

Tutorial 4: Model Fits and Data Split Tests (MASTML_Tutorial_4_Models_and_Tests.ipynb):
    In this notebook, we will learn how to run a few different types of models on a select
    dataset, and conduct a few different types of data splits to evaluate our model performance. In
    this tutorial, we will:
        1. Run a variety of model types from the scikit-learn package
        2. Run a bootstrapped ensemble of linear ridge regression models
        3. Compare performance of scikit-learn's gradient boosting method and XGBoost
        4. Compare performance of scikit-learn's neural network and Keras-based neural network regressor
        5. Compare model performance using random k-fold cross validation and leave out group cross validation
        6. Explore the limits of model performance when up to 90% of data is left out using leave out percent cross validation

Tutorial 5: Left-out data, Nested cross-validation, and Optimized models (MASTML_Tutorial_5_NestedCV_and_OptimizedModels.ipynb):
    In this notebook, we will perform more advanced model fitting routines, including nested cross validation
    and hyperparameter optimization. In this tutorial, we will learn how to use MAST-ML to:
        1. Assess performance on manually left-out test data
        2. Perform nested cross validation to assess model performance on unseen data
        3. Optimize the hyperparameters of our models to create the best model

Tutorial 6: Model Error Analysis, Uncertainty Quantification (MASTML_Tutorial_6_ErrorAnalysis_UncertaintyQuantification.ipynb):
    In this notebook tutorial, we will learn about how MAST-ML can be used to:
        1. Assess the true and predicted errors of our model, and some useful measures of their statistical distributions
        2. Explore different methods of quantifying and calibrating model uncertainties.
        3. Compare the uncertainty quantification behavior of Bayesian and ensemble-based models.