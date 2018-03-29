#####################
Installing MASTML
#####################

*************
Requirements
*************

========
Hardware
========
Computer capable of running python 3.

=========
Data
=========

* Numeric data in a CSV file. There must be at least some target feature data, so that models can be fit.

* First row of each column should have a text name which is how columns will be referenced later in the input file.

* (Advanced: Data in a database with an associated python class that can extract the data to a CSV file.)

**************
Installation
**************

For Python and MAST-ml package installation, see:

:doc:`0_1_terminal_installation` OR

:doc:`0_2_windows_installation`

Note that in some situations such as running on a computer without a local display, you may also need to change the matplotlib backend in order for plotting to work
(see :ref:`Matplotlib backend <matplotlib-backend>`).

*******************
Startup
*******************

===========================
Locate the examples folder
===========================

In the MASTML code directory, navigate to the ``examples`` folder.

The file ``example_input.conf`` will use ``basic_test_data.csv`` to run an example.

========================
Run the MASTML command
========================

The format is ``python <path to MASTML>/MASTML.py <config file name>``

For example, from inside the ``examples`` folder::
    
    python ../MASTML.py example_input.conf

This is a terminal command. 
For Windows, assuming setup has been followed
as above, go to the Anaconda Navigator, Environments, select the environment,
click the green arrow button, and Open terminal.

================
Check output
================

``index.html`` should be created, linking to certain representative plots for each test

Output will be located in subfolders in the test folder given by save_path in the :ref:`General Setup <general-setup>` section of the input file.

