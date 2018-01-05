====================
SingleFitGrouped
====================

Like :doc:`4_1_singlefit`, but also splits data by a grouping feature,
and reports statistics and plots data by groups.

A grouping feature must be set in the :ref:`General Setup <general-setup>` section of the input file. 

-----------------
Input keywords
-----------------

See :doc:`4_1_singlefit`:

* training_dataset
* testing_dataset
* xlabel
* ylabel
* plot_filter_out

Additional keywords:

* mark_outlying_groups: Number of outlying groups (groups with the largest predictive RMSEs) to plot separately
    * This number should typically be 1, 2, or 3.
* fit_only_on_matched_groups: 
    * 0: Fit on all data in the training dataset (default)
    * 1: Fit only on data in the training dataset where the testing dataset also contains some data with the same value of the grouping feature (that is, if the grouping feature is Alloy, and there are no Alloy X's in the testing dataset, then do not use any Alloy X data when training)

----------------
Code
----------------

.. autoclass:: SingleFitGrouped.SingleFitGrouped

