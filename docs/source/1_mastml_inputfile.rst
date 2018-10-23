#####################
MAST-ML Input File
#####################

This document provides an overview of the various sections and fields of the MAST-ML input file.

A full template input file can be downloaded here: :download:`MASTML_InputFile <MASTML_InputFile.conf>`

*************
Input file sections
*************

========
General Setup
========
The "GeneralSetup" section of the input file allows the user to specify an assortment of basic MAST-ML parameters, ranging
from which column names in the CSV file to use as features for fitting (i.e. X data) or to fit to (i.e. y data), as well
as which metrics to employ in fitting a model, among other things.

[GeneralSetup]
    input_features = Auto
    target_feature = Reduced barrier (eV)
    randomizer = False
    metrics = Auto
    not_input_features = Host element, Solute element, predict_Pt
    grouping_feature = Host element
    validation_columns = predict_Pt

=========
Data Cleaning
=========
The "DataCleaning" section of the input file allows the user to clean their data to remove rows or columns that contain
empty or NaN fields, or fill in these fields using imputation or principal component analysis methods.

[DataCleaning]
    cleaning_method = remove
    imputation_strategy = mean

=========
Clustering
=========
The "Clustering" section of the input file allows the user to group their data using different clustering algorithms.

[Clustering]
    [[KMeans_5Clusters]]
        n_clusters = 5
    [[SpectralClustering_14Clusters]]
        n_clusters = 14