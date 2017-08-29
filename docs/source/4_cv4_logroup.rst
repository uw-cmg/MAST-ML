=====================================
Leave Out Group Cross Validation
=====================================

Perform leave-out-group cross validation.

Data points are assigned to groups based on the ``grouping_feature`` in the :ref:`General Setup <general-setup>` section of the input file.

Each group is left out in turn, the model is trained on the remaining data, and then the model predicts data for the left-out group.

-----------------
Input keywords
-----------------

See :doc:`4_1_singlefit`:

* training_dataset (Should be the same as testing_dataset)
* testing_dataset (Should be the same as training_dataset)

Additional keywords:

* xlabel: Label for the groups along the x-axis
* ylabel: Label for the average RMSE of each group along the y-axis
* mark_outlying_points: Number of outlying points (one point for each group) to label

----------------
Code
----------------

.. autoclass:: LeaveOutGroupCV.LeaveOutGroupCV

