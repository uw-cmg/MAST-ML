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

