====================
PlotNoAnalysis
====================

Plots target data for each testing dataset versus a selected feature.
Data is also split out by a grouping feature set in the :ref:`General Setup <general-setup>` section of the input file. 

This test does no analysis or model fitting and is meant for double-checking data.

-----------------
Input keywords
-----------------

See :doc:`4_1_singlefit`:

* training_dataset: for compatibility; only testing_dataset is used
* testing_dataset: multiple testing datasets are allowed
* plot_filter_out

Additional keywords:

* xlabel: x-axis label for selected feature
* ylabel: y-axis label for target feature
* feature_plot_feature: Feature to plot on x-axis, against target feature data
* data_labels: List of dataset labels, one for each dataset in testing_dataset

----------------
Code
----------------

.. autoclass:: PlotNoAnalysis.PlotNoAnalysis

