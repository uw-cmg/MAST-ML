*************************
Introduction
*************************

This document provides step-by-step tutorials of conducting and analyzing different MAST-ML runs. For this tutorial,
we will be using the dataset example_data.csv in the tests/csv/ folder and input file example_input.conf in tests/conf/.

MAST-ML requires two files to run: The first is the text-based input file (.conf extension). This file contains all of
the key settings for MAST-ML, for example, which models to fit and how to normalize your input feature matrix. The
second file is the data file (.csv or .xlsx extension). This is the data file containing the input feature columns and
values (X values) and the corresponding y data to fit models to. The data file may contain other columns that are
dedicated to constructing groups of data for specific tests, or miscellaneous notes, which columns can be selectively
left out so they are not used in the fitting. This will be discussed in more detail below.

Throughout this tutorial, we will be modifying the input file to add and remove different sections and values. For a
complete and more in-depth discussion of the input file and its myriad settings, the reader is directed to the dedicated
input file section:

:ref:`mastml_input_file`

The data contained in the example_data.csv file consist of a previously selected matrix of X features created from
combinations of elemental properties, for example the average atomic radius of the elements in the material. The y data
values used for fitting are listed in the "Reduced barrier (eV)" column, and are DFT-calculated migration barriers of
dilute solute diffusion, referenced to the host system. For example, the value of Ag solute diffusing through a Ag host
is set to zero. The "Host element" and "Solute element" columns denote which species comprise the corresponding reduced
migration barrier.