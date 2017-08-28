===================
ParamGridSearch
===================

Perform a grid search over hyperparameter values. 

-----------------
Input keywords
-----------------

See :doc:`4_1_singlefit` for the following keywords:

* training_dataset
* testing_dataset: Should be the same as training_dataset

Additional keywords:

* param_1, param_2, param_3, etc.: hyperparameter optimization instructions, made up of semicolon-delimited pieces as follows:

    * Piece 1: the word 'model' or a custom feature class.method string,
        e.g. DBTT.calculate_EffectiveFluence, for a class and method 
            * The custom module file (.py file) must reside in the ``custom_features`` folder
            * The custom module file must have the same name as the custom class (e.g. DBTT.py for class DBTT)
    * Piece 2: the parameter name
    * Piece 3: the parameter type.
            * Use only 'int', 'float', 'bool', or 'str'.
            * Some sklearn model hyperparameters may be type-sensitive
    * Piece 4: the series type:
            * 'discrete': Grid will use values given in Piece 5
            * 'continuous': Grid will be defined by a range given in Piece 5
            * 'continuous-log': Grid will be defined by a log10 range given in Piece 5
    * Piece 5: grid values, given by:
            * A colon-delimited list of values, if Piece 4 is 'discrete'
                * A parameter with only one discrete value will be considered
                    a non-optimized parameter, and can be used to define
                    parameters of custom features.
            * Colon-delimited values for the starting value, ending value, and number of points in the range, if Piece 4 is 'continuous' or 'continuous-log'
                * The range will be given using numpy's linspace for 'continous' and numpy's logspace for 'continuous-log'.
                * The start and end value are inclusive.
    * Example::

        param_1= model;alpha;float;continuous-log;-10:10:21
        param_2= model;gamma;float;continuous-log;-5:5:21
        param_3= DBTT.calculate_EffectiveFluence;pvalue;float;discrete;0.2
        param_4= DBTT.calculate_EffectiveFluence;flux_feature;str;discrete;flux_n_cm2_sec
        param_5= DBTT.calculate_EffectiveFluence;fluence_feature;str;discrete;fluence_n_cm2

* fix_random_for_testing: 
        * 0 - use random numbers (default)
        * 1 - fix the randomizer for testing purposes
* num_cvtests: The number of cross-validation (CV) tests to use for evaluating every grid point
* num_folds: The number of folds for K-fold CV; leave blank to use leave-out-percent CV
* percent_leave_out: Percent to leave out for leave-out-percent CV; leave blank to use K-fold CV
* num_bests: Number of best individuals to track (default 10)
* processors: Number of processors to use. 
    * 1 - use one processor (default)
    * 2 or higher - use this many processors, all on a SINGLE node (using the multiprocessing module)
* pop_upper_limit: Upper limit for population size. Raises an error if the
    grid contains more than this many individuals. (default 1000000)

----------------
Code
----------------

.. autoclass:: ParamGridSearch.ParamGridSearch

