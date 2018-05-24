__author__ = 'Tam Mayeshiba'
__maintainer__ = 'Ryan Jacobs'
__version__ = '1.0'
__email__ = 'rjacobs3@wisc.edu'
__date__ = 'October 14th, 2017'

import os
import numpy as np
from sklearn.model_selection import LeaveOneOut
from plot_data.PlotHelper import PlotHelper
from KFoldCV import KFoldCV
from SingleFit import timeit

class LeaveOneOutCV(KFoldCV):
    """Class to conduct leave-one-out cross-validation analysis
   
    Args:
        training_dataset (DataHandler object): Training dataset handler
        testing_dataset (DataHandler object): Testing dataset handler
        model (sklearn model object): sklearn model
        save_path (str): Save path
        xlabel (str): Label for full-fit x-axis (default "Measured")
        ylabel (str): Label for full-fit y-axis (default "Predicted")
        mark_outlying_points (int): Number of outlying points to mark in best and worst tests, e.g. 3
 
    Returns:
        Analysis in the save_path folder
        Plots results in a predicted vs. measured square plot.

    Raises:
        ValueError: if testing target data is None; CV must have testing target data

    """
    def __init__(self, 
        training_dataset=None,
        testing_dataset=None,
        model=None,
        save_path=None,
        xlabel="Measured",
        ylabel="Predicted",
        mark_outlying_points=None,
        *args, **kwargs):
        """
        Additional class attributes to parent class:
            Set by keyword:
            Set in code:
            self.all_pred_array <numpy array>: array of all predictions, since each data point has a separate prediction
        """
        if not(training_dataset == testing_dataset):
            raise ValueError("Only testing_dataset will be used. Use the same values for training_dataset and testing_dataset")
        KFoldCV.__init__(self, 
            training_dataset=training_dataset, #only testing_dataset is used
            testing_dataset=testing_dataset,
            model=model, 
            save_path = save_path,
            xlabel=xlabel,
            ylabel=ylabel,
            mark_outlying_points = mark_outlying_points,
            num_cvtests=1, #single test
            num_folds = -1, #set in code
            fix_random_for_testing = 0, #no randomization in this test
            )
        self.all_pred_array=None

    def set_up_cv(self):
        self.num_folds = len(self.testing_dataset.target_data) 
        KFoldCV.set_up_cv(self)
        self.readme_list.append("Equivalent to %i leave-one-out CV tests\n" % self.num_folds)

    @timeit
    def predict(self):
        #Predictions themselves are covered in self.fit()
        self.get_statistics()
        self.print_statistics()
        self.readme_list.append("----- Output data -----\n")
        self.print_output_csv(label="loo", cvtest_entry=self.cvtest_dict[0])

    def print_statistics(self):
        self.readme_list.append("----- Statistics -----\n")
        statlist=['avg_rmse','std_rmse','avg_mean_error','std_mean_error']
        for key in statlist:
            self.readme_list.append("%s: %3.3f\n" % (key,self.cvtest_dict[0][key]))
    
    def print_output_csv(self, label="", cvtest_entry=None):
        """
        """
        olabel = "%s_test_data.csv" % label
        ocsvname = os.path.join(self.save_path, olabel)
        self.testing_dataset.add_feature("LOO Predictions", 
                    cvtest_entry['prediction_array'])
        cols = self.testing_dataset.print_data(ocsvname, ["LOO Predictions"])
        self.readme_list.append("%s file created with columns:\n" % olabel)
        for col in cols:
            self.readme_list.append("    %s\n" % col)

    @timeit
    def plot(self):
        self.readme_list.append("----- Plotting -----\n")
        notelist=list()
        notelist.append("Mean over %i LOO tests:" % (self.num_folds))
        notelist.append("RMSE:")
        notelist.append("    {:.2f} $\pm$ {:.2f}".format(self.cvtest_dict[0]['avg_rmse'], self.cvtest_dict[0]['std_rmse']))
        notelist.append("Mean error:")
        notelist.append("    {:.2f} $\pm$ {:.2f}".format(self.cvtest_dict[0]['avg_mean_error'], self.cvtest_dict[0]['std_mean_error']))
        self.plot_results(notelist=list(notelist))
        self.plot_residuals_histogram()
    
    def plot_results(self, notelist=list()):
        kwargs2 = dict()
        kwargs2['xlabel'] = self.xlabel
        kwargs2['ylabel'] = self.ylabel
        kwargs2['labellist'] = list(["loo_prediction"]) 
        kwargs2['xdatalist'] = list([self.testing_dataset.target_data])
        kwargs2['ydatalist'] = list(
                [self.cvtest_dict[0]['prediction_array']]) #only one cvtest, with number of folds equal to number of data points
        kwargs2['xerrlist'] = list([self.testing_dataset.target_error_data])
        kwargs2['yerrlist'] = list([None])
        kwargs2['notelist'] = list(notelist)
        kwargs2['guideline'] = 1
        kwargs2['plotlabel'] = "loo_results"
        kwargs2['save_path'] = self.save_path
        if not (self.mark_outlying_points is None):
            kwargs2['marklargest'] = self.mark_outlying_points
            if (self.testing_dataset.labeling_features is None):
                raise ValueError("Must specify some labeling features if you want to mark the largest outlying points")
            labels = self.testing_dataset.data[self.testing_dataset.labeling_features[0]]
            kwargs2['mlabellist'] = list([labels])
        myph = PlotHelper(**kwargs2)
        myph.multiple_overlay()
        self.readme_list.append("Plot loo_results.png created,\n")
        self.readme_list.append("    showing results of all LOO tests.\n")

