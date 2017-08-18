=============
SingleFit
=============

Perform a single fit using all input feature data and target feature data
from the training dataset.

Create predictions for all observations in the testing dataset using 
input feature data from the testing dataset.

If target data is available in the testing dataset, plot target predictions
versus target data.

-----------------
Input keywords
-----------------

* training_dataset: Training dataset. Use a dataset name from the :ref:`Data Setup <data-setup>` section of the input file.
* testing_dataset: Testing dataset. Use a dataset name from the :ref:`Data Setup <data-setup>` section of the input file.
* xlabel: Label for x-axis in a plot of target prediction vs. target data (default "Measured")
* ylabel: Label for y-axis in a plot of target prediction vs. target data (default "Predicted")
* plot_filter_out: Comma-separated list of feature;operator;value for filtering out values in the plot.
    * Example: ``temperature_C;<>;290,time;=;0``

----------------
Code
----------------

.. autoclass:: SingleFit.SingleFit

