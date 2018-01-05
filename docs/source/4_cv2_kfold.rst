=====================================
KFoldCV
=====================================

Perform K-fold cross validation.

In plots, predictions for all K folds are plotted.

-----------------
Input keywords
-----------------

See :doc:`4_cv1_loperc`:

* training_dataset (Should be the same as testing_dataset)
* testing_dataset (Should be the same as training_dataset)
* xlabel 
* ylabel
* mark_outlying_points
* num_cvtests
* fix_random_for_testing: Whether to fix the random seed for testing

Additional keywords: 

* num_folds: Number of folds to use
    * Data is split into this many folds. A new split is used for each CV test.
    * Each CV test consists of this many subtests, where each fold in turn is left out, the model is trained on all of the remaining data, and then the model makes predictions for the observations in the left-out fold. 
    * Typical values are 2 and 5.

----------------
Code
----------------

.. autoclass:: KFoldCV.KFoldCV

