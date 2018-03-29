#####################
Input files
#####################
This documentation references the **ryan_updates_2018-03-08** branch. Not all changes may be in master yet.

A sample input file with all available section is included in the /MASTML_input_file_template/ directory

**********************************************
General notes on data file structure
**********************************************

* MAST-ML currently reads in data files in a csv format.

* Each column in the data file represents one feature and the first row should be text with a name for that feature

* To check that a csv is formatted correctly try to open the file as plain text (using notepad or a similar app). If formatted correctly you should see each value seperated by a single comma with no extra spaces or commas added in.

**********************************************
General notes on configuration file structure
**********************************************

* Any lines in the input file may be commented out using #. Characters after # will be ignored as comments

* For more detailed explanations of the keywords in each section check section of the documentation on the specific test of interest.

* For keywords that accept multiple inputs comma seperate each input.

* Single square brackets denote sections of the configuration file.

* Double square brackets denote subsections specific to each section.

***********************************
Sections of the configuration file
***********************************

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

* **save_path** is the root save path for all of the tests in the input file.

* **input_features** are all the other features that should be used in the prediction. Use ``input_features = Auto`` to make everything except target_feature into input_features.

* **target_feature** is what should be predicted. This names should end in _regression for regression models and _classification for classification models. Make sure this also matches the naming in the csv data file as well.

.. _data-setup:

====================
Data Setup
====================
Example::

    [Data Setup]
     
        [[Initial]]
        data_path = ./basic_test_data.csv
        weights = False

        [[Dataset_2]]
        data_path = ./second_datat_set.csv
        weights = False

* **data_path** is the path to the CSV file.

* **weights** has no effect as of now.

* Multiple subsections with different names for different datasets may follow after the Initial subsection.

================================
Feature Normalization
================================
This section generates features.

Example::

    [Feature Generation]
        normalize_x_features = True
        normalize_y_features = False
        feature_normalization_type = standardize
        feature_scale_min = 0
        feature_scale_max = 1

* **feature_normalization_type** can be set to **standardize** or **normalize**

* **standardize** rescales the data to have a mean of 0 and and standard deviation of 1.

* **normalize** rescales the data linearly to have a minimum value of **feature_scale_min** and a maximum of **feature_scale_max**.

================================
Feature Generation
================================
This section generates features. 

Example::

    [Feature Generation]
        perform_feature_generation = True
        add_magpie_features = True
        add_materialsproject_features = False
        add_citrine_features = False
        materialsproject_apikey = ####
        citrine_apikey = ####

* A ``Materials composition`` column must be included in the data file to generate features, using common materials naming formats, for example, "H2O".

* Multiple columns can also be included named as **Material composition 1** to **Material composition N** to specify specific groupings within a composition.

* To skip this section set **add_magpie_features** to False

* The materials project and citrine features may not be working right now.

===============================
Feature Selection
===============================
This section reduces the number of input features, especially useful if many new features were generated through the Feature Generation section.

Example::

    [Feature Selection]
        perform_feature_selection = True
        remove_constant_features = True
        feature_selection_algorithm = univariate_feature_selection
        use_mutual_information = True
        scoring_metric = root_mean_squared_error
        generate_feature_learning_curve = True
        model_to_use_for_learning_curve = randomforest_model_regressor
        number_of_features_to_keep = 10

* **feature_selection_algorithm** can be **univariate_feature_selection**, **recursive_feature_elimination**, **sequential_forward_selection**, **basic_forward_selection**

* **scoring_metric** determines which model performance parameter to use in generating the learning curve. It can be **mean_squared_error**, **mean_absolute_error**, **root_mean_squared_error**, **r2_score**.

* **model_to_use_for_learning_curve** sets which model to use to generate the learning curve and can be any model type available in the ``Model Parameters`` section.

==========================
Models and Tests to Run
==========================
This section sets up which models and tests will actually run when the configuration file is read in.

Example::

    [Models and Tests to Run]
     
        models = gkrr_model_regressor
        test_cases = SingleFit_withy, SingleFit_withouty
        #test_cases = ParamOptGA, SingleFit_withy

* Many models and tests may be specified in the ``Test Parameters`` and ``Model Parameters`` sections, but only the ones listed here will actually be run. This behavior can be useful if you only want to rerun some individual tests from a large configuration file.

* Use a comma to separate multiple models or tests. 

* For multiple models, each test in ``test_cases`` will be run with each model specified in ``models``.

* Test and model names must correspond exactly to the names given in the ``Models`` and ``Test Parameters`` sections.

* Multiple test_cases of the same type can be run by adding and underscore and a unique identifier to the test name. Note that this also has to match the test names in the ``Test Parameters`` section.

================================
Test Parameters
================================
This section specifies the parameters for individual tests.
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

     
        [[SingleFit_withy]]
        training_dataset = Initial
        testing_dataset  = Initial
        xlabel = Measured target
        ylabel = Target prediction

        [[SingleFit_withouty]]
        training_dataset = Initial
        testing_dataset  = Dataset_2
        xlabel = Measured target
        ylabel = Target prediction

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
    *   ParamGridSearch
    *   ParamOptGA

Check the **Tests** section of the documentation for details on keywords for each test.

.. _model-parameters:

=========================
Model Parameters
=========================
This section sets up the model parameters. If ParamOptGA is used, model parameters will be optimized, and numeric model parameters set here will not be used.

Example::

    [Model Parameters]
     
        [[gkrr_model_regressor]]
        alpha = 0.003019951720
        gamma = 3.467368504525
        coef0 = 1
        degree = 3
        kernel = rbf

* Model names should correspond to the model names in the Models and Tests to Run section

* Available model types are included in the /MASTML_input_file_template/MASTML_input_file_template.conf file

* For explanations of individual model parameters you can also check the sklearn documentation at http://scikit-learn.org/stable/

