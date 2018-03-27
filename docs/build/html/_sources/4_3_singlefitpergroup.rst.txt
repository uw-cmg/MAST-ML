====================
SingleFitPerGroup
====================

Performs :doc:`4_1_singlefit` for each group of data. Data is split out by a grouping feature set in the :ref:`General Setup <general-setup>` section of the input file. 

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

* mark_outlying_groups: Number of outlying groups to mark

----------------
Code
----------------

.. autoclass:: SingleFitPerGroup.SingleFitPerGroup

