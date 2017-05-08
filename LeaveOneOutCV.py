import os
import matplotlib
import numpy as np
import data_parser
import matplotlib.pyplot as plt
from mean_error import mean_error
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import data_analysis.printout_tools as ptools
import plot_data.plot_predicted_vs_measured as plotpm
import plot_data.plot_xy as plotxy
from SingleFit import SingleFit
from KFoldCV import KFoldCV
from LeaveOutPercentCV import LeaveOutPercentCV
from SingleFit import timeit
from sklearn.metrics import r2_score

class LeaveOneOutCV(KFoldCV):
    """leave-one-out cross validation
   
    Args:
        training_dataset, (Should be the same as testing_dataset)
        testing_dataset, (Should be the same as training_dataset)
        model,
        save_path,
        input_features,
        target_feature,
        target_error_feature,
        labeling_features, 
        xlabel, 
        ylabel,
        stepsize,
        mark_outlying_points (Use only 1 number), see parent class.
 
    Returns:
        Analysis in the save_path folder
        Plots results in a predicted vs. measured square plot.
    Raises:
        ValueError if testing target data is None; CV must have
                testing target data
    """
    def __init__(self, 
        training_dataset=None,
        testing_dataset=None,
        model=None,
        save_path=None,
        input_features=None,
        target_feature=None,
        target_error_feature=None,
        labeling_features=None,
        xlabel="Measured",
        ylabel="Predicted",
        stepsize=1,
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
            input_features = input_features, 
            target_feature = target_feature,
            target_error_feature = target_error_feature,
            labeling_features = labeling_features,
            xlabel=xlabel,
            ylabel=ylabel,
            stepsize=stepsize,
            mark_outlying_points = mark_outlying_points,
            num_cvtests=1, #single test
            num_folds = -1, #set in code
            fix_random_for_testing = 0, #no randomization in this test
            )
        self.all_pred_array=None
        return 

    def set_up_cv(self):
        self.num_folds = len(self.testing_target_data) 
        KFoldCV.set_up_cv(self)
        self.readme_list.append("Equivalent to %i leave-one-out CV tests\n" % self.num_folds)
        return

    @timeit
    def predict(self):
        #Predictions themselves are covered in self.fit()
        self.get_statistics()
        self.print_statistics()
        self.readme_list.append("----- Output data -----\n")
        self.print_output_csv(label="loo", cvtest_entry=self.cvtest_dict[0])
        return

    @timeit
    def plot(self):
        self.readme_list.append("----- Plotting -----\n")
        notelist=list()
        notelist.append("Mean over %i LOO tests:" % (self.num_folds))
        notelist.append("RMSE:")
        notelist.append("    {:.2f} $\pm$ {:.2f}".format(self.statistics['avg_fold_avg_rmses'], self.statistics['std_fold_avg_rmses']))
        notelist.append("Mean error:")
        notelist.append("    {:.2f} $\pm$ {:.2f}".format(self.statistics['avg_fold_avg_mean_errors'], self.statistics['std_fold_avg_mean_errors']))
        self.plot_results(notelist=list(notelist))
        return
    
    def plot_results(self, notelist=list()):
        kwargs2 = dict()
        kwargs2['xlabel'] = self.xlabel
        kwargs2['ylabel'] = self.ylabel
        kwargs2['labellist'] = list(["_prediction"]) #underscore prevents label
        kwargs2['xdatalist'] = list([self.testing_target_data])
        kwargs2['ydatalist'] = list(
                [self.cvtest_dict[0]['prediction_array']]) #only one cvtest, with number of folds equal to number of data points
        kwargs2['xerrlist'] = list([self.testing_target_data_error])
        kwargs2['yerrlist'] = list([None])
        kwargs2['notelist'] = list(notelist)
        kwargs2['guideline'] = 1
        kwargs2['plotlabel'] = "loo_results"
        kwargs2['save_path'] = self.save_path
        kwargs2['stepsize'] = self.stepsize
        if not (self.mark_outlying_points is None):
            kwargs2['marklargest'] = self.mark_outlying_points
            if (self.labeling_features is None) or (len(self.labeling_features) == 0):
                raise ValueError("Must specify some labeling features if you want to mark the largest outlying points")
            labels = np.asarray(self.testing_dataset.get_data(self.labeling_features[0])).ravel()
            kwargs2['mlabellist'] = list([labels])
        plotxy.multiple_overlay(**kwargs2)
        self.readme_list.append("Plot loo_results.png created,\n")
        self.readme_list.append("    showing results of all LOO tests.\n")
        return

