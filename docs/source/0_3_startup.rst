*******************
Startup
*******************

===========================
Locate the examples folder
===========================

In the installed MASTML directory, navigate to the ``tests`` folder.

Under tests/conf, The file ``example_input.conf`` will use the ``example_data.xlsx`` data file located in tests/csv to run an example.

========================
Run the MASTML command
========================

The format is ``python3 -m mastml.mastml_driver <path to config file> <path to data .xlsx file> -o <path to results folder>``

For example, to conduct the test run above, while in the MASTML install directory::

    python3 -m mastml.mastml_driver tests/conf/example_input.conf tests/csv/example_data.xlsx -o results/example_results

This is a terminal command.
For Windows, assuming setup has been followed
as above, go to the Anaconda Navigator, Environments, select the environment,
click the green arrow button, and Open terminal.

When you execute the above command, you'll know it's working if you begin to see output on your screen.

================
Check output
================

``index.html`` should be created, linking to certain representative plots for each test

For this example, output will be located in subfolders in the results/example_results folder.

Check the following to see if the run completed successfully::

    A log.log file is generated and the last line contains the phrase "Making html file of all run stats..."
    An index.html file that gives some summary plots from all the tests that were run
    A series of subfolders with names "StandardScaler"->"DoNothing"->"KernelRidge", with the following three directories
    within the "KernelRidge" directory: "LeaveOneGroupOut_host", "NoSplit", and "RepeatedKFold"

You can compare all of these files with those given in the /example_results directory which should match.