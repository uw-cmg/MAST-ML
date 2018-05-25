# MAterial Science Tools - Machine Learning 

## Project Standards:

* Wrap lines to 100 chars
* `Pylint` to check code style

## Large scale TODO items:

* Refactor code
* Add unit tests
* Write classification routines
* Improve rst and docstring documentation
* Speed up the grid search hyperparameter thing

## Random Ideas

* Serve the http documentation somewhere so users don't have to build it from .rst themseleves
* Seperate out the .conf file parsing/running stuff from the rest of the program.
  (Two modules: 1. a sklearn/keras wrapper 2. a .conf file parser and executer)
* Standardize/abstract the image generation, so user only needs to specify the image type and
  location once.

## Documentation and install instructions

Full documentation starts in `docs/build/html/index.html`.

Clone the repository and open `index.html` with your browser.

## Contributors 
DBTT Skunkworks undergraduates (~2015-2017, alphabetical order):
*   Ben Anderson
*   Josh Cordell
*   Fenil Doshi
*   Jerit George
*   Aren Lorenson
*   Matthew Parker
*   Josh Perry
*   Liam Witteman
*   Haotian (Will) Wu
*   Jinyu Xia
*   Hao Yuan

CMG group members:
*   Ben Afflerbach
*   Ryan Jacobs
*   Tam Mayeshiba
*   Henry Wu

Advisor:
*   Professor Dane Morgan

UKY Summer Employees:
*   Luke Harold Miles
*   Robert Max Williams

## Summary of modules in MASTML

### DataHandler.py
 
This module serves as a constructor to organize aspects of a pandas dataframe, such as which fields
are input vs. target data, list of features, etc. This is used to organize data mainly for easy
passing of relevant data to sklearn models for prediction and subsequent plotting for SingleFit and
cross-validation routines (e.g. KFoldCV)

### DataOperations.py

This module handles parsing of input data files (e.g. in csv or xlsx format) to pandas dataframes,
and splits the dataframe by which features are used in fitting and which data are being fit to. In
addition, there is a collection of miscellaneous functions that are useful for various operations on
dataframes and numpy arrays, and methods to map between the two data types.

### FeatureGeneration.py

This module handles the creation of materials science-centric features to fit data to. These are
generally consisted of constructing features based on the composition of the material of interest,
such as averages of elemental properties of the elements comprising the material, and pulling data
from online materials databases. This is often used in lieu of, or in addition to, custom
user-specified features.

### FeatureOperations.py

This module contains a broad assortment of pandas dataframe helper utilities related to adding,
removing and grouping features for specific kinds of fits, as well as handling feature normalization
input. Normalizing features (e.g., rescaling all values so that the mean is 0 and standard deviation
is 1) can be important for correct usage of certain regression models.

### FeatureSelection.py

This module contains routines to effectively down-select the most important features from a large
feature space that are found to best describe the data the user is fitting to. For example, after
MAST-ML uses the FeatureGeneration module to construct a feature matrix, there may be 600 features
to fit a model to. Realistically, many of these features are either redundant or contain no useful
information. Different algorithms to select the most important features are thus useful to obtain
the most accurate fits and minimize overfitting.

### KFoldCV.py

This module is used to conduct k-fold cross validation tests for a specific model. In k-fold cross
validation, the full data set is split into k groups, and each of the k groups is held out of the
set, and the model is trained on the remaining data and the model accuracy is assessed on the data
that was left out. This is done for each of the k folds and the results are averaged. The data
comprising each fold is random, so many iterations of k-fold cross validation are typically
performed and the fitting results are averaged over the number of k-fold cross validation runs.

### LeaveOneOutCV.py

This module is the same as KFoldCV, except that each data point is left out once. This is the
equivalent of KFoldCV where k = number of data points.

### LeaveOutGroupCV.py

This module conducts cross validation by leaving out user-specified groups of data. This is useful
if your data naturally clusters by some physical variable that the user is aware of. For example, if
the Boston Housing data was grouped into houses that were contained in 3 distinct neighborhoods, it
is reasonable to suspect that the house price may vary by neighborhood. Instead of doing KFoldCV,
which randomly splits the data and will thus contain testing and training data from all
neighborhoods at once, this method allows the user to selectively leave out entire neighborhoods at
a time. This is a good way to test the extensibility of your model to groups of data that are truly
outside of the training set.

### LeaveOutPercentCV.py

This module is the same as KFoldCV, except instead of specifying groups of data by number of folds,
the groups are denoted by a percent of the total dataset.  For example, leave out 10% cross
validation is the same as 10-fold cross validation.

### MASTML.py

This is the main driver function to conduct a MAST-ML run. Generally, nothing in this module is
directly called, except when executing a run by typing “python MASTML.py”. This module thus performs
a full workflow and calls the other relevant modules and methods therein based on the parameters the
user specifies in the input file.

### MASTMLInitializer.py

This module handles the initialization of a MAST-ML run by parsing the input file and checking that
the user-specified values and test/model names in the input file are all valid. This module is also
used for creating instances of the specific models to be used later in fitting to the dataset of
interest.

### ParamGridSearch.py

This module is used in the optimization of hyperparameters. It conducts a grid search of up to 4
different numerical parameters. The range and increment size of possible values for each parameter
are specified by the user. The grid search then evaluates the model of interest at each point on the
grid, and suggests the best parameter values based on which values yield the best model fit (e.g.
least error in a regression task).

### ParamOptGA.py

This module is also used for hyperparameter optimization. Instead of conducting a grid search, a
genetic algorithm is used to perform a global minimization of the model error based on values of the
user-specified hyperparameters.

### PlotNoAnalysis.py

This is a module that is used to construct data plots (e.g. parity plots of predicted vs. true data
values) without having to perform the various statistical analyses that usually accompany tests like
KFoldCV and SingleFit.

### PredictionVsFeature.py

This module is a special case of SingleFit, where the predicted values are plotted against the true
values for one specific feature (SingleFit uses all features). This can be useful if the user wants
to directly see how well individual features affect the model fit.

### SingleFit.py

This module conducts a “full fit” of the data, and generates a parity plot of predicted data vs.
actual data values. A full fit is one where all the data and all the features are used in the fit.
For more accurate model assessment, some sort of cross validation is required. The single fit error
represents the absolute best case scenario of how your model will perform, but it isn’t meant to
assess the predictive accuracy of a given model.

### SingleFitGrouped.py

This module is the same as SingleFit, except that it creates parity plots based on data that is
organized into specific groups. All data is put on one plot and the data points are colored
differently based on which group they are in (e.g.  houses in different neighborhoods based on the
example above).

### SingleFitPerGroup.py

This module is the same as SingleFit, except that it creates parity plots based on data that is
organized into specific groups. The data from each group is put in a separate plot to more easily
see how the fit varies between groups. This is in contrast to SingleFitGrouped, which puts data of
all groups on one plot and just colors them differently.


## Directories

+ The docs folder has all the files that build the html documentation.
+ The examples folder has the example run to do to ensure installation was done correctly.
+ mongo_data was also from a previous research project that initially launched this whole endeavor.
+ plot_data contains various plotting utilities used to make the plots for SingleFit and KFoldCV
  tests.
+ test and unit_tests are collections of separate unit tests.
+ under_development are some other code pieces that are stale and haven't been integrated to the
  main package.
+ the input file template folder just has a full example input file for reference.
+ The config file path contains the data needed to make some custom features for feature generation.
+ the Jupyternotebook part is meant to be a collection of tutorials and examples written in notebook
  style. Those are being written separately by a grad student in our group

_Original code 4/12/17 pulled from the DBTT Skunkworks project._

