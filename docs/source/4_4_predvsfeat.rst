====================
PredictionVsFeature
====================

Performs :doc:`4_1_singlefit` for each testing dataset, and also plots target data and target predictions for each testing dataset, versus a selected feature. Data is also split out by a grouping feature set in the :ref:`General Setup <general-setup>` section of the input file. 

-----------------
Input keywords
-----------------

See :doc:`4_1_singlefit`:

* training_dataset
* testing_dataset: multiple testing datasets are allowed
* xlabel
* ylabel
* plot_filter_out

Additional keywords: 

* feature_plot_feature: Feature to plot on the x-axis
* feature_plot_xlabel: x-axis label for per-group plots of target data and target predictions versus the designated feature. Groups are determined via ``grouping_feature`` in the :ref:`General Setup <general-setup>` section of the input file.
* feature_plot_ylabel: y-axis label for per-group plots
* markers: Comma-delimited marker list for different series (matplotlib)
* outlines: Comma-delimited color list for marker outlines
* linestyles: Comma-delimited list for line styles connecting markers; None is allowed as a line style
* sizes: Comma-delimited list for marker sizes
* data_labels: Comma-delimited list of testing dataset labels; if there is only one testing dataset, use a trailing comma

----------------
Code
----------------

.. autoclass:: PredictionVsFeature.PredictionVsFeature

