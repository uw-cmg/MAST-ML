=====================================
Leave Out Percent Cross Validation
=====================================

Perform leave-out-percent cross validation.

Percentages are determined randomly for each test. If it is necessary to ensure that all data is used, :doc:`4_cv2_kfold` may be more appropriate. 

-----------------
Input keywords
-----------------

See :doc:`4_1_singlefit`:

* training_dataset (Should be the same as testing_dataset)
* testing_dataset (Should be the same as training_dataset)
* xlabel 
* ylabel

Additional keywords:

* mark_outlying_points: Comma-separated pair of the number of outlying points to mark in the best and worst tests, respectively, e.g. ``0,3``
* percent_leave_out: Percentage of data to leave out in each CV test
* num_cvtests: Number of CV tests to perform
* fix_random_for_testing: Whether to fix the random seed for testing so that repeated test case runs will be identical
    * 1 - fix the random seed
    * 0 (default) - do not fix the random seed

----------------
Code
----------------

.. autoclass:: LeaveOutPercentCV.LeaveOutPercentCV

