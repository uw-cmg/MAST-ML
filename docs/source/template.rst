######
Part
######
********
Chapter
********
========
Section
========
------------
Subsection
------------
^^^^^^^^^^^^^^
Subsubsection
^^^^^^^^^^^^^^
""""""""""""
paragraph
""""""""""""

********************
Test Classes
********************

:ref:`4_1_singlefit`



.. _singlefit:

=============
SingleFit
=============

Perform a single fit to all input data and create predictions for all observations.

------------
Keywords
------------

* training_dataset 
    * <DataHandler object>
    * Training dataset handler
* testing_dataset 
    * <DataHandler object>
    * Testing dataset handler
    * If more than one datahandler is passed as a list, only the first one will be used.
* model
    * <sklearn model>
    * Machine-learning model
* save_path
    * <str>
    * Save path
* xlabel <str>: Label for full-fit x-axis (default "Measured")
* ylabel <str>: Label for full-fit y-axis (default "Predicted")
* stepsize <float>: Step size for plot grid (default None)
* plot_filter_out <list>: List of semicolon-delimited strings with
                            feature;operator;value for filtering out
                            values for plotting.
