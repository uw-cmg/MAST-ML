__author__ = 'Tam Mayeshiba'
__maintainer__ = 'Ryan Jacobs'
__version__ = '1.0'
__email__ = 'rjacobs3@wisc.edu'
__date__ = 'October 14th, 2017'

import numpy as np
from plot_data.PlotHelper import PlotHelper
import os
from LeaveOutPercentCV import LeaveOutPercentCV
from SingleFit import timeit
from sklearn.model_selection import LeaveOneGroupOut

class LeaveOutGroupCV(LeaveOutPercentCV):
    """Class to conduct leave-out-group cross-validation analysis
   
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
        ValueError: if grouping feature is None
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
        """
        if not(training_dataset == testing_dataset):
            raise ValueError("Only testing_dataset will be used. Use the same values for training_dataset and testing_dataset")
        LeaveOutPercentCV.__init__(self, 
            training_dataset=training_dataset, #only testing_dataset is used
            testing_dataset=testing_dataset,
            model=model, 
            save_path = save_path,
            xlabel=xlabel,
            ylabel=ylabel,
            mark_outlying_points = mark_outlying_points,
            percent_leave_out = -1, #not using this field
            num_cvtests = -1, #not using this field; num_cvtests is set by number of groups
            fix_random_for_testing = 0, #no randomization in this test
            )
        if self.testing_dataset.grouping_feature is None:
            raise ValueError("grouping feature must be set.")
        return 

    def predict(self):
        #Predictions themselves are covered in self.fit()
        self.get_statistics()
        self.print_statistics()
        self.readme_list.append("----- Output data -----\n")
        for cvtest in self.cvtest_dict.keys():
            self.print_output_csv(label=self.cvtest_dict[cvtest]['group'], cvtest_entry=self.cvtest_dict[cvtest])
        return

    @timeit
    def plot(self):
        self.readme_list.append("----- Plotting -----\n")
        notelist=list()
        notelist.append("Mean RMSE: {:.2f}".format(self.statistics['avg_rmse']))
        kwargs=dict()
        kwargs['xlabel'] = self.xlabel
        kwargs['ylabel'] = self.ylabel
        kwargs['notelist'] = notelist
        kwargs['save_path'] = self.save_path
        kwargs['marklargest'] = self.mark_outlying_points
        group_label_list=list()
        rms_list=list()
        group_rms_list=list()
        for cvtest in self.cvtest_dict.keys():
            group_rms_list.append((self.cvtest_dict[cvtest]['group'],
                                    self.cvtest_dict[cvtest]['rmse']))
        group_rms_list.sort() #sorts by group
        group_rms_array = np.array(group_rms_list)
        kwargs['xdatalist'] = [group_rms_array[:,0]]
        kwargs['ydatalist'] = [np.array(group_rms_array[:,1],'float')]
        kwargs['xerrlist'] = [None]
        kwargs['yerrlist'] = [None]
        kwargs['labellist'] = ['predicted_rmse']
        kwargs['plotlabel'] = "leave_out_group"
        myph = PlotHelper(**kwargs)
        myph.plot_rmse_vs_text()
        self.readme_list.append("Plot leave_out_group.png created.\n")
        return

    def set_up_cv(self):
        if self.testing_dataset.target_data is None:
            raise ValueError("Testing target data cannot be none for cross validation.")
        indices = np.arange(0, len(self.testing_dataset.target_data))
        self.num_cvtests = len(self.testing_dataset.groups)
        self.readme_list.append("----- CV setup -----\n")
        self.readme_list.append("%i CV tests,\n" % self.num_cvtests)
        self.readme_list.append("leaving out %s groups.\n" % self.testing_dataset.grouping_feature)
        self.cvmodel = LeaveOneGroupOut()
        cvtest=0
        for train, test in self.cvmodel.split(self.testing_dataset.input_data,
                        self.testing_dataset.target_data,
                        self.testing_dataset.group_data):
            group = self.testing_dataset.groups[cvtest]
            self.cvtest_dict[cvtest] = dict()
            self.cvtest_dict[cvtest]['train_index'] = train
            self.cvtest_dict[cvtest]['test_index'] = test
            self.cvtest_dict[cvtest]['group'] = group #keep track of group
            cvtest=cvtest + 1
        return

    def print_statistics(self):
        LeaveOutPercentCV.print_statistics(self)
        self.readme_list.append("Statistics by group:\n")
        statlist=list()
        for cvtest in range(0, self.num_cvtests):
            for key in ['rmse','mean_error']:
                statlist.append("  %s: %s: %3.3f\n" % (self.cvtest_dict[cvtest]['group'], key, self.cvtest_dict[cvtest][key]))
        statlist.sort() #sorts by group
        for stat in statlist:
            self.readme_list.append(stat)
        return

    def print_output_csv(self, label="", cvtest_entry=None):
        """
        """
        olabel = "%s_test_data.csv" % label
        ocsvname = os.path.join(self.save_path, olabel)
        self.testing_dataset.add_feature("Group Prediction", 
                    cvtest_entry['prediction_array'])
        addl_cols = list()
        addl_cols.append("Group Prediction")
        cols = self.testing_dataset.print_data(ocsvname, addl_cols)
        self.readme_list.append("%s file created with columns:\n" % olabel)
        for col in cols:
            self.readme_list.append("    %s\n" % col)
        return

