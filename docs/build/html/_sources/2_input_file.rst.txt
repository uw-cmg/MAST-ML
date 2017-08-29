#####################
Input file 
#####################
This documentation is looking at the **dev** branch. Not all changes may be in master yet.

********
General
********

* Any lines in the input file may be commented out using #. Characters after # will be ignored as comments

* For explanations of the keywords associated with sections like SingleFit, there will be html documentation in the future. For now, look at the docstring for the class on github, like SingleFit.py

*********
Sections
*********

.. _general-setup:

================
General Setup
================
This section has general setup information common to the entire input file.

Example::

    [General Setup]
     
        save_path = ./
        input_features = x1, x2, x3
        target_feature = y_feature
        target_error_feature = error
        labeling_features = group, group_num
        grouping_feature = group
        config_files_path = ../MASTML_config_files
        normalize_features = True

* **save_path** is the root save path for all of the tests in the input file.

* **target_feature** is what should be predicted.
 
* **target_error_feature** is the error data for the target_feature.
 
* **labeling_features** are the names and number identifiers for groups. This generally overlaps with grouping_feature unless you want a different naming method.
 
* **grouping_feature** is the feature to create groups on for any of the group tests.
 
* **input_features** are all the other features that should be used in the prediction. Use ``input_features = Auto`` to make everything except target feature into input_features.

* **config_files_path** is the path to the general configuration files for MASTML 

* **normalize_features** indicates whether to automatically normalize the input features.

======================
CSV Setup (optional)
======================
This section is used to call a python routine that will automatically create a CSV file. Most users should skip this section.

Example::

    [CSV Setup]
        setup_class = test.random_data.MakeRandomData
        save_path = ../random_data
        random_seed = 0

* **setup_class** is the setup class, which must have a run() method.

* All following keywords will be passed to the class __init__() method as a kwargs dictionary.

.. _data-setup:

====================
Data Setup
====================
Example::

    [Data Setup]
     
        [[Initial]]
        data_path = ./basic_test_data.csv
        weights = False

* **data_path** is the path to the CSV file.

* **weights** has no effect as of now.

* Multiple subsections with different names for different datasets may follow after the Initial subsection.

================================
Feature Generation (optional)
================================
This section generates features. 

Example::

    [Feature Generation]
        add_magpie_features = True
        add_materialsproject_features = False
        materialsproject_apikey = 'USE_YOUR_API_KEY' # 
        #add_citrine_features = False #This won't work right now
        #citrine_apikey = 'USE_YOUR_CITRINE_API_KEY'

* A ``Materials composition`` column must be included, using common materials naming formats, for example, "H2O".

* Note that you may need to copy the magpiedata folder into your test folder for magpie features to work (DEV)

===============================
Feature Selection (optional)
===============================
This section reduces the number of input features, especially useful if many new features were generated through the Feature Generation section.

Example::

    [Feature Selection] #optional
        remove_constant_features = True
        feature_selection_algorithm = univariate #Choose either RFE, univariate, stability
        selection_type = regression # Choose either regression or classification
        number_of_features_to_keep = 10

==========================
Models and Tests to Run
==========================
This section sets up which models and tests to run.

Example::

    [Models and Tests to Run]
     
        models = gkrr_model
        test_cases = SingleFit_withy
        #test_cases = ParamOptGA, SingleFit_withy

* Many models and tests may be specified in the following sections, but only the ones listed here will actually be run. This behavior can be useful if you only want to rerun some individual tests.

* Use a comma to separate multiple models or tests. 

* For multiple models, each test in ``test_cases`` will be run with each model specified in ``models``.

* Test and model names must correspond exactly to the names given in the ``Models`` and ``Test Parameters`` sections.

================================
Test Parameters
================================
This section specifies the parameters for tests.
Note that parameters are subject to change. Documentation mismatches or missing documentation should be reported via a github issue.

Example::

    [Test Parameters]
        [[ParamOptGA]]
        training_dataset = Initial
        testing_dataset = Initial
        num_folds = 2
        num_cvtests = 20
        num_gas = 2
        population_size = 50
        convergence_generations = 2
        max_generations = 10
        fix_random_for_testing = 1
        num_parents = 10
        use_multiprocessing = 2 #usually number of processors - 1
        #additional_feature_methods =
     
        [[SingleFit_withy]]
        training_dataset = Initial
        testing_dataset  = Initial
        xlabel = Measured target
        ylabel = Target prediction
        stepsize = 1.0

* Names in [[ ]] must exactly match names listed in the test_cases keyword of the Models and Tests to Run section. 

* Names must follow the format Testname_YourNamingScheme. Testname options are given below, while YourNamingScheme can be any combination of numbers or letters that you can use to provide more information, for example, KFoldCV_5fold

Test names available are:

    *   SingleFit
    *   SingleFitPerGroup
    *   SingleFitGrouped
    *   KFoldCV
    *   LeaveOneOutCV
    *   LeaveOutPercentCV
    *   LeaveOutGroupCV
    *   PredictionVsFeature
    *   PlotNoAnalysis
    *   ParamOptGA (note that only hyperparameters in basicinput.conf are optimized)

Separate documentation will later be available for each class.
 
ParamOptGA:

* fix_random_for_testing will always seed random numbers

-------------------------------
Common Test Parameter options
-------------------------------

* **training_dataset**: This is the data the model will be fit to.

* **testing_dataset**: This is the data that will be predicted and shown on most plots.
 
* **xlabel** and **ylabel**: labels for generated plots. 
 
* **plot_filter_out**: This specifies data to be filtered and not included on plots. The general form is::

    plot_filter_out = feature_name;operator;value

The feature name can be any feature from your data.
 
* **mark_outlying_groups**: This a numeric value of the number of groups to write notations for on the output plots starting with the groups that performed worst.

=========================
Model Parameters
=========================
This section sets up the model parameters. If ParamOptGA is used, model parameters will be optimized, and numeric model parameters set here will not be used.

Example::

    [Model Parameters]
     
        [[gkrr_model]]
        alpha = 0.003019951720
        gamma = 3.467368504525
        coef0 = 1
        degree = 3
        kernel = rbf

* Model names should correspond to the model names in the Models and Tests to Run section

* For explanations of individual model parameters, see the sklearn documentation at http://scikit-learn.org/stable/

* Supported models for hyperparameter optimization (DEV) are:

    * gkrr_model (sklearn.kernel_ridge.KernelRidge)
    
    * decision_tree_model (sklearn.tree.DecisionTreeRegressor)

    * randomforest_model (sklearn.ensemble.RandomForestRegressor)

    * nn_model (sklearn.neural_network.MLPRegressor)
