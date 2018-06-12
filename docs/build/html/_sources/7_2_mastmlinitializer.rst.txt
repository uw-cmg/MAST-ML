===================
MASTML Initializer
===================

MASTML initializer classes:

*ConfigFileParser*: Class to read in contents of MASTML input files

*ConfigFileConstructor*: Constructor class to build input file template. Used for input file validation and data type casting.

*ConfigFileValidator*: Class to validate contents of user-specified MASTML input file and flag any errors

*ModelTestConstructor*: Class that takes parameters from configdict (configfile as dict) and performs calls to appropriate MASTML methods

----------------
Code
----------------

.. autoclass:: MASTMLInitializer.ConfigFileParser

.. autoclass:: MASTMLInitializer.ConfigFileConstructor

.. autoclass:: MASTMLInitializer.ConfigFileValidator

.. autoclass:: MASTMLInitializer.ModelTestConstructor

