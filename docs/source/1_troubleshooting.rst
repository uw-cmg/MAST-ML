#################
Troubleshooting
#################

********
General
********

==============================
Check the MASTMLlog.log file
==============================

If you do not get a message like “Successfully read in and parsed your MASTML input file, test.conf” then something is wrong with the configuration file.

Note that in early development, your config file may be correct, but your mastmlvalidationnames.conf and/or mastmlvaliddationtypes.conf files may not be up to date; may not include all class names.

* Get those mastml*.conf files from test/main_test

* Put them in your working directory with your .conf file and data csv(s).

=====================================
Update your local copy of the code
=====================================

MASTML is changing quickly. Check master and pull from master before test runs.

If you are having issues with github, check status.github.com


***********
KeyError
***********

Generally KeyError’s come from not matching test_cases in the [models and tests to run] section with tests in the [Test Parameters} section

They can also be caused by not having updated the mastmlvalidationnames.conf and mastmlvalidationtypes.conf files.

.. _matplotlib-backend:

***********************************************
QXcbConnection: Could not connect to display 
***********************************************

If no plots can be created because of a display error,
check the MASTMLlog.log file near the beginning of 
the current MASTML run for the lines ::
    
    Using matplotlib backend <backend name>
    Matplotlib defaults from <file location>

Some backends only work on certain machines or setups:

    * Mac computers have a macosx backend

    * Clusters may need a graphical display to use the Qt5Agg backend

Try adding a matplotlibrc file into your current running directory
(e.g. where your .conf file is) with the line ::

    backend: Agg

or another backend to replace the previous backend. 
Rerun MASTML and see if the MASTMLlog.log file picks up the new backend and,
if so, whether the plots are now made correctly.

If the problem persists, please submit a github issue ticket.

*********************
MemoryError
*********************

Memory errors may be found when running ParamOptGA or other tests.
Note that ParamOptGA builds an initial large dictionary of all combinations,
so the number and grid spacing of parameters in the param_xxx 
will affect the initial memory that the test must use.

Try increasing the available memory or decreasing parameters.

To check memory when running on a cluster, try adding a line ::

    ulimit -a

to the submission script to see if there is any memory information.

On PBS/Torque, it may be helpful to try omitting the pvmem term, as in::

    #PBS -l nodes=1:ppn=12

(no pvmem term, normally #PBS -l nodes=1:ppn=12,pvmem=2000mb), or increasing it.



