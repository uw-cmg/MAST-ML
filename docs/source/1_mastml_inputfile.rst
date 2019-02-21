.. _mastml_input_file:

#####################
MAST-ML Input File
#####################

This document provides an overview of the various sections and fields of the MAST-ML input file.

A full template input file can be downloaded here: :download:`MASTML_InputFile <MASTML_fullinputfile.conf>`

*******************
Input file sections
*******************

=============
General Setup
=============
The "GeneralSetup" section of the input file allows the user to specify an assortment of basic MAST-ML parameters, ranging
from which column names in the CSV file to use as features for fitting (i.e. X data) or to fit to (i.e. y data), as well
as which metrics to employ in fitting a model, among other things.

Example::

    [GeneralSetup]
        input_features = feature_1, feature_2, etc. or "Auto"
        target_feature = target_feature
        randomizer = False
        metrics = root_mean_squared_error, mean_absolute_error, etc. or "Auto"
        not_input_features = additional_feature_1, additional_feature_2
        grouping_feature = grouping_feature_1
        validation_columns = validation_feature_1

* **input_features** List of input X features
* **target_feature** Target y feature
* **randomizer** Whether or not to randomize y feature data
* **metrics** Which metrics to evaluate model fits
* **not_input_features** Additional features that are not to be fitted on (i.e. not X features)
* **grouping_feature** Feature names that provide information on data grouping
* **validation_columns** Feature name that designates whether data will be used for validation (set rows as 1 or 0 in csv file)

=============
Data Cleaning
=============
The "DataCleaning" section of the input file allows the user to clean their data to remove rows or columns that contain
empty or NaN fields, or fill in these fields using imputation or principal component analysis methods.

Example::

    [DataCleaning]
        cleaning_method = remove, imputation, ppca
        imputation_strategy = mean, median

* **cleaning_method**  Method of data cleaning. "remove" simply removes columns with missing data. "imputation" uses basic operation to fill in missing values. "ppca" uses principal component analysis to fill in missing values.
* **imputation_strategy** Only valid field if doing imputation, selects method to impute missing data by using mean, median, etc. of the column

==========
Clustering
==========
Optional section to perform clustering of data using well-known clustering algorithms available in scikit-learn.
Note that the subsection names must match the corresponding name of the routine in scikit-learn. More information on
clustering routines and the parameters to set for each routine can be found here:
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
For the purpose of this full input file, we use the scikit-learn default parameter values. Note that not all parameters are listed.

Example::

    [Clustering]
        [[AffinityPropagation]]
            damping = 0.5
            max_iter = 200
            convergence_iter = 15
            affinity = euclidean
        [[AgglomerativeClustering]]
            n_clusters = 2
            affinity = euclidean
            compute_full_tree = auto
            linkage = ward
        [[Birch]]
            threshold = 0.5
            branching_factor = 50
            n_clusters = 3
        [[DBSCAN]]
            eps = 0.5
            min_samples = 5
            metric = euclidean
            algorithm = auto
            leaf_size = 30
        [[KMeans]]
            n_clusters = 8
            n_init = 10
            max_iter = 300
            tol = 0.0001
        [[MiniBatchKMeans]]
            n_clusters = 8
            max_iter = 100
            batch_size = 100
        [[MeanShift]]
        [[SpectralClustering]]
            n_clusters = 8
            n_init = 10
            gamma = 1.0
            affinity = rbf

==================
Feature Generation
==================
Optional section to perform feature generation based on properties of the constituent elements. These routines were
custom written for MAST-ML, except for PolynomialFeatures. For more information on the MAST-ML custom routines, consult
the MAST-ML online documentation. For more information on PolynomialFeatures, see:
http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

Example::

    [FeatureGeneration]
        [[Magpie]]
            composition_feature = Material Compositions
        [[MaterialsProject]]
            composition_feature = Material Compositions
            api_key = my_api_key
        [[Citrine]]
            composition_feature = Material Compositions
            api_key = my_api_key
        [[ContainsElement]]
            composition_feature = Host element
            all_elements = False
            element = Al
            new_name = has_Al
        [[PolynomialFeatures]]
            degree=2
            interaction_only=False
            include_bias=True

* **composition_feature** Name of column in csv file containing material compositions
* **api_key** Your API key to access the Materials Project or Citrine. Register for your account at Materials Project: https://materialsproject.org or at Citrine: https://citrination.com
* **all_elements** For ContainsElement, whether or not to scan all data rows to assess all elements present in data set
* **element** For ContainsElement, name of element of interest. Ignored if all_elements = True
* **new_name** For ContainsElement, name of new feature column to generate. Ignored if all_elements = True

=====================
Feature Normalization
=====================
Optional section to perform feature normalization of the input or generated features using well-known
feature normalization algorithms available in scikit-learn. Note that the subsection names must match the corresponding
name of the routine in scikit-learn. More information on normalization routines and the parameters to set for each
routine can be found here: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing .
For the purpose of this full input file, we use the scikit-learn default parameter values. Note that not all parameters are listed,
and only the currently listed normalization routines are supported. In addition, MeanStdevScaler is a custom written normalization
routine for MAST-ML. Additional information on MeanStdevScaler can be found in the online MAST-ML documentation.

Example::

    [FeatureNormalization]
        [[Binarizer]]
            threshold = 0.0
        [[MaxAbsScaler]]
        [[MinMaxScaler]]
        [[Normalizer]]
            norm = l2
        [[QuantileTransformer]]
            n_quantiles = 1000
            output_distribution = uniform
        [[RobustScaler]]
            with_centering = True
            with_scaling = True
        [[StandardScaler]]
        [[MeanStdevScaler]]
            mean = 0
            stdev = 1


=====================
Learning Curve
=====================
Optional section to perform learning curve analysis on a dataset. Two types of learning curves will be generated: a
data learning curve (score vs. amount of training data) and a feature learning curve (score vs. number of features).

Example::

    [LearningCurve]
        estimator = KernelRidge_learn
        cv = RepeatedKFold_learn
        scoring = root_mean_squared_error
        n_features_to_select = 5
        selector_name = MASTMLFeatureSelector

* **estimator** A scikit-learn model/estimator. The name needs to match an entry in the [Models] section. Note this model will be removed from the [Models] list after the learning curve is generated.
* **cv** A scikit-learn cross validation generator. The name needs to match an entry in the [DataSplits] section. Note this method will be removed from the [DataSplits] list after the learning curve is generated.
* **scoring** A scikit-learn scoring method compatible with MAST-ML. See the MAST-ML online documentation at https://htmlpreview.github.io/?https://raw.githubusercontent.com/uw-cmg/MAST-ML/dev_Ryan_2018-10-29/docs/build/html/3_metrics.html for more information.
* **n_features_to_select** The max number of features to use for the feature learning curve.
* **selector_name** Method to conduct feature selection for the feature learning curve. The name needs to match an entry in the [FeatureSelection] section. Note this method will be removed from the [FeatureSelection] section after the learning curve is generated.

=====================
Feature Selection
=====================
Optional section to perform feature selection using routines in scikit-learn, mlxtend and custom-written for MAST-ML.
Note that the subsection names must match the corresponding name of the routine in scikit-learn. More information on
selection routines and the parameters to set for each routine can be found here:
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection . For the purpose of this full
input file, we use the scikit-learn default parameter values. Note that not all parameters are listed, and only the
currently listed selection routines are supported. In addition, MASTMLFeatureSelector is a custom written selection
routine for MAST-ML. Additional information on MASTMLFeatureSelector can be found in the online MAST-ML documentation.
Finally, SequentialFeatureSelector is a routine available from the mlxtend package, which documention can be found
here: http://rasbt.github.io/mlxtend/

Example::

    [FeatureSelection]
        [[GenericUnivariateSelect]]
        [[SelectPercentile]]
        [[SelectKBest]]
        [[SelectFpr]]
        [[SelectFdr]]
        [[SelectFwe]]
        [[RFE]]
            estimator = RandomForestRegressor_selectRFE
            n_features_to_select = 5
            step = 1
        [[SequentialFeatureSelector]]
            estimator = RandomForestRegressor_selectSFS
            k_features = 5
        [[RFECV]]
            estimator = RandomForestRegressor_selectRFECV
            step = 1
            cv = LeaveOneGroupOut_selectRFECV
            min_features_to_select = 1
        [[SelectFromModel]]
            estimator = KernelRidge_selectfrommodel
            max_features = 5
        [[VarianceThreshold]]
            threshold = 0.0
        [[PCA]]
            n_components = 5
        [[MASTMLFeatureSelector]]
            estimator = KernelRidge_selectMASTML
            n_features_to_select = 5
            cv = LeaveOneGroupOut_selectMASTML

* **estimator**  A scikit-learn model/estimator. The name needs to match an entry in the [Models] section. Note this model will be removed from the [Models] list after the learning curve is generated.
* **n_features_to_select** The max number of features to select
* **step** For RFE and RFECV, the number of features to remove in each step
* **k_features** For SequentialFeatureSelector, the max number of features to select.
* **cv** A scikit-learn cross validation generator. The name needs to match an entry in the [DataSplits] section. Note this method will be removed from the [DataSplits] list after the learning curve is generated.

=====================
Data Splits
=====================
Optional section to perform data splits using cross validation routines in scikit-learn, and custom-written for MAST-ML.
Note that the subsection names must match the corresponding name of the routine in scikit-learn. More information on
selection routines and the parameters to set for each routine can be found here:
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection . For the purpose of this full
input file, we use the scikit-learn default parameter values. Note that not all parameters are listed, and only the
currently listed data split routines are supported. In addition, NoSplit is a custom written selection routine for
MAST-ML, which simply produces a full data fit with no cross validation. Additional information on NoSplit can be found
in the online MAST-ML documentation.

Example::

    [DataSplits]
        [[NoSplit]]
        [[KFold]]
            shuffle = True
            n_splits = 10
        [[RepeatedKFold]]
            n_splits = 5
            n_repeats = 10
        # Here, an example of another instance of RepeatedKFold, this one being used in the [LearningCurve] section above.
        [[RepeatedKFold_learn]]
            n_splits = 5
            n_repeats = 10
        [[GroupKFold]]
            n_splits = 3
        [[LeaveOneOut]]
        [[LeavePOut]]
            p = 10
        [[RepeatedStratifiedKFold]]
            n_splits = 5
            n_repeats = 10
        [[StratifiedKFold]]
            n_splits = 3
        [[ShuffleSplit]]
            n_splits = 10
        [[StratifiedShuffleSplit]]
            n_splits = 10
        [[LeaveOneGroupOut]]
            # The column name in the input csv file containing the group labels
            grouping_column = Host element
        # Here, an example of another instance of LeaveOneGroupOut, this one being used in the [FeatureSelection] section above.
        [[LeaveOneGroupOut_selectMASTML]]
            # The column name in the input csv file containing the group labels
            grouping_column = Host element
        # Here, an example of another instance of LeaveOneGroupOut, this one being used based on the creation of the "has_Al"
        # group from the [[ContainsElement]] routine present in the [FeatureGeneration] section.
        [[LeaveOneGroupOut_Al]]
            grouping_column = has_Al
        # Here, an example of another instance of LeaveOneGroupOut, this one being used based on the creation of clusters
        # from the [[KMeans]] routine present in the [Clustering] section.
        [[LeaveOneGroupOut_kmeans]]
            grouping_column = KMeans

=========
Models
=========
Optional section to denote different models/estimators for model fitting from scikit-learn. Note that the subsection
names must match the corresponding name of the routine in scikit-learn. More information on different model routines
and the parameters to set for each routine can be found here for ensemble methods:
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble and here for kernel ridge and linear methods:
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_ridge and here for neural network methods:
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neural_network and here for support vector machine
and decision tree methods: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm . For the purpose of
this full input file, we use the scikit-learn default parameter values. Note that not all parameters are listed, and only
the currently listed data split routines are supported.

Example::

    [Models]
        # Ensemble methods

        [[AdaBoostClassifier]]
            n_estimators = 50
            learning_rate = 1.0
        [[AdaBoostRegressor]]
            n_estimators = 50
            learning_rate = 1.0
        [[BaggingClassifier]]
            n_estimators = 50
            max_samples = 1.0
            max_features = 1.0
        [[BaggingRegressor]]
            n_estimators = 50
            max_samples = 1.0
            max_features = 1.0
        [[ExtraTreesClassifier]]
            n_estimators = 10
            criterion = gini
            min_samples_split = 2
            min_samples_leaf = 1
        [[ExtraTreesRegressor]]
            n_estimators = 10
            criterion = mse
            min_samples_split = 2
            min_samples_leaf = 1
        [[GradientBoostingClassifier]]
            loss = deviance
            learning_rate = 1.0
            n_estimators = 100
            subsample = 1.0
            criterion = friedman_mse
            min_samples_split = 2
            min_samples_leaf = 1
        [[GradientBoostingRegressor]]
            loss = ls
            learning_rate = 0.1
            n_estimators = 100
            subsample = 1.0
            criterion = friedman_mse
            min_samples_split = 2
            min_samples_leaf = 1
        [[RandomForestClassifier]]
            n_estimators = 10
            criterion = gini
            min_samples_leaf = 1
            min_samples_split = 2
        [[RandomForestRegressor]]
            n_estimators = 10
            criterion = mse
            min_samples_leaf = 1
            min_samples_split = 2

        # Kernel ridge and linear methods

        [[KernelRidge]]
            alpha = 1
            kernel = linear
        # Here, an example of another instance of KernelRidge, this one being used based by the [[MASTMLFeatureSelector]]
        # method from the [FeatureSelection] section.
        [[KernelRidge_selectMASTML]]
            alpha = 1
            kernel = linear
        # Here, an example of another instance of KernelRidge, this one being used based in the [LearningCurve] section.
        [[KernelRidge_learn]]
            alpha = 1
            kernel = linear

        [[ARDRegression]]
            n_iter = 300
        [[BayesianRidge]]
            n_iter = 300
        [[ElasticNet]]
            alpha = 1.0
        [[HuberRegressor]]
            epsilon = 1.35
            max_iter = 100
        [[Lars]]
        [[Lasso]]
            alpha = 1.0
        [[LassoLars]]
            alpha = 1.0
            max_iter = 500
        [[LassoLarsIC]]
            criterion = aic
            max_iter = 500
        [[LinearRegression]]
        [[LogisticRegression]]
            penalty = l2
            C = 1.0
        [[Perceptron]]
            alpha = 0.0001
        [[Ridge]]
            alpha = 1.0
        [[RidgeClassifier]]
            alpha = 1.0
        [[SGDClassifier]]
            loss = hinge
            penalty = l2
            alpha = 0.0001
        [[SGDRegressor]]
            loss = squared_loss
            penalty = l2
            alpha = 0.0001

        # Neural networks

        [[MLPClassifier]]
            hidden_layer_sizes = 100,
            activation = relu
            solver = adam
            alpha = 0.0001
            batch_size = auto
            learning_rate = constant
        [[MLPRegressor]]
            hidden_layer_sizes = 100,
            activation = relu
            solver = adam
            alpha = 0.0001
            batch_size = auto
            learning_rate = constant

        # Support vector machine methods

        [[LinearSVC]]
            penalty = l2
            loss = squared_hinge
            tol = 0.0001
            C = 1.0
        [[LinearSVR]]
            epsilon = 0.1
            loss = epsilon_insensitive
            tol = 0.0001
            C = 1.0
        [[NuSVC]]
            nu = 0.5
            kernel = rbf
            degree = 3
        [[NuSVR]]
            nu = 0.5
            C = 1.0
            kernel = rbf
            degree = 3
        [[SVC]]
            C = 1.0
            kernel = rbf
            degree = 3
        [[SVR]]
            C = 1.0
            kernel = rbf
            degree = 3

        # Decision tree methods

        [[DecisionTreeClassifier]]
            criterion = gini
            splitter = best
            min_samples_split = 2
            min_samples_leaf = 1
        [[DecisionTreeRegressor]]
            criterion = mse
            splitter = best
            min_samples_split = 2
            min_samples_leaf = 1
        [[ExtraTreeClassifier]]
            criterion = gini
            splitter = random
            min_samples_split = 2
            min_samples_leaf = 1
        [[ExtraTreeRegressor]]
            criterion = mse
            splitter = random
            min_samples_split = 2
            min_samples_leaf = 1

=================
Plot Settings
=================
This section controls which types of plots MAST-ML will write to the results directory.

Example::

    [PlotSettings]
        target_histogram = True
        train_test_plots = True
        predicted_vs_true = True
        predicted_vs_true_bars = True
        best_worst_per_point = True
        feature_vs_target = False
        average_normalized_errors = True

* **target_histogram** Whether or not to output target data histograms
* **train_test_plots** Whether or not to output parity plots within each CV split
* **predicted_vs_true** Whether or not to output summarized parity plots
* **predicted_vs_true_bars** Whether or not to output averaged parity plots
* **best_worst_per_point** Whether or not to output parity plot showing best and worst split per point
* **feature_vs_target** Whether or not to show plots of target feature as a function of each individual input feature
* **average_normalized_errors** Whether or not to show the average plots of the normalized errors