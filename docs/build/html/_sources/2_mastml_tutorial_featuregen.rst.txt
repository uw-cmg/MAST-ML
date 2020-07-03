****************************************
Feature generation and normalization
****************************************

For this run, we are going to first generate a large X feature matrix based on a suite of elemental properties. Then,
we are going to normalize the feature matrix so that all values in a given feature column have a mean of zero and a
standard deviation equal to one.

To perform the feature generation and normalization steps, add these sections to your input file. Use the same file from
the previous run, which contains the GeneralSetup and DataCleaning sections, and use your data file with the values you
previously removed. (Note that you can use the pristine original data file too, and the data cleaning step will simply
do nothing). For the purpose of this example, we are going to generate elemental features using the MAGPIE approach,
using compositions as specified in the "Solute element" column of the data file. Note that if multiple elements are
present, features containing the average (both mean and composition-weighted averages) of the elements present will be
calculated. The value specified in the composition_feature parameter must be a column name in your data file which
contains the material compositions.

Example::

    [FeatureGeneration]
        [[Magpie]]
            composition_feature = Solute element
            feature_types = composition_avg, arithmetic_avg, max, min, difference, elements

    [FeatureNormalization]
        [[StandardScaler]]



After performing this run, we can see that the .csv files in the feature generation and normalization folders of the
results directory tree are now updated to reflect the generated and normalized X feature matrices. There are now many
more features in the generated_features.csv file:

.. image:: MASTMLtutorial_run3_1.png

Note that feature columns that are identical in all values are removed automatically. We can see that the normalized
feature set consists of each column having mean zero and standard deviation of one:

.. image:: MASTMLtutorial_run3_2.png