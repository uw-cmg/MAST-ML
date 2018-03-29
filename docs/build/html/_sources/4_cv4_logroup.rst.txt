=====================================
LeaveOutGroupCV
=====================================

Perform leave-out-group cross validation.

Data points are assigned to groups based on the ``grouping_feature`` in the :ref:`General Setup <general-setup>` section of the input file.

Each group is left out in turn, the model is trained on the remaining data, and then the model predicts data for the left-out group.

----------------
Code
----------------

.. autoclass:: LeaveOutGroupCV.LeaveOutGroupCV

