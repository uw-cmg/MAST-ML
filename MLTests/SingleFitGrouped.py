__author__ = 'Tam Mayeshiba'
__maintainer__ = 'Ryan Jacobs'
__version__ = '1.0'
__email__ = 'rjacobs3@wisc.edu'
__date__ = 'October 14th, 2017'

import numpy as np
from sklearn.metrics import mean_squared_error
from plot_data.PlotHelper import PlotHelper
import os
from MLTests.SingleFit import SingleFit
from MLTests.SingleFit import timeit
import logging

class SingleFitGrouped(SingleFit):
    """
    Class to perform a single full fit and split out group contributions to
     overall scoring metric used.

    Args:
        training_dataset (DataHandler object): Training dataset handler
        testing_dataset (DataHandler object): Testing dataset handler
        model (sklearn model object): sklearn model
        save_path (str): Save path
        xlabel (str): Label for full-fit x-axis (default "Measured")
        ylabel (str): Label for full-fit y-axis (default "Predicted")
        plot_filter_out (list): List of semicolon-delimited strings with feature;operator;value for
          leaving out specific values for plotting.
        mark_outlying_groups (int): Number of outlying groups to mark
        fit_only_on_matched_groups (int): 0 - fit on all data in the training dataset (default)
                                          1 - fit only on groups in the training dataset that are also in the testing dataset

    Returns:
        Analysis in the save_path folder
        Plots results in a predicted vs. measured square plot.

    Raises:
        ValueError: if grouping_feature is not set in testing dataset
        ValueError: if testing target data is None; has to have at least some testing target data to
                     plot
    """
    def __init__(self, training_dataset=None, testing_dataset=None, model=None, save_path=None,
                 xlabel="Measured", ylabel="Predicted", plot_filter_out=None,
                 mark_outlying_groups=2, fit_only_on_matched_groups=0):
        """
        Additional class attributes to parent class:
            Set by keyword:
            self.mark_outlying_groups <int>: Number of outlying groups to mark.
                                If greater than the number of groups,
                                all groups will be marked separately.
            self.fit_only_on_matched_groups <int>: If 1, fit only on
                                groups in training that are also in testing.
            Set in code:
            self.per_group_statistics <dict>: Dictionary of per-group RMSEs
            self.outlying_groups <list>: List of groups with highest RMSE
            self.plotting_dict <dict>: Dictionary of data to plot
        """
        SingleFit.__init__(
            self,
            training_dataset=training_dataset,
            testing_dataset=testing_dataset,
            model=model,
            save_path = save_path,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_filter_out=plot_filter_out)
        if self.testing_dataset.grouping_feature is None:
            raise ValueError("grouping_feature is not set.")
        self.mark_outlying_groups = int(mark_outlying_groups)
        self.fit_only_on_matched_groups = int(fit_only_on_matched_groups)
        # Sets later in code
        self.per_group_statistics = dict()
        self.outlying_groups = list()
        self.plotting_dict = dict()

    @timeit
    def set_up(self):
        if self.testing_dataset.target_data is None:
            raise ValueError("testing target data cannot be None")
        SingleFit.set_up(self)
        if self.fit_only_on_matched_groups != 1:
            return
        out_training_groups = np.setdiff1d(self.training_dataset.groups,
                                           self.testing_dataset.groups)
        filter_list = [(self.training_dataset.grouping_feature, "=", group)
                       for group in out_training_groups]
        for group in out_training_groups:
            filter_list.append((self.training_dataset.grouping_feature, "=" ,group))
        self.training_dataset.add_filters(filter_list)
        self.training_dataset.set_up_data_from_features() #reset all data

    def get_statistics(self):
        self.get_per_group_statistics(preface="")
        SingleFit.get_statistics(self) #also adds plot filters
        if not self.plot_filter_out is not None:
            self.get_per_group_statistics(preface="filtered_")
        self.get_outlying_groups()

    def print_statistics(self):
        SingleFit.print_statistics(self)
        self.readme_list.append("Per-group RMSEs from overall fit:\n")
        groups = list(self.per_group_statistics.keys())
        groups.sort()
        for group in groups:
            skeys = list(self.per_group_statistics[group].keys())
            skeys.sort()
            for skey in skeys:
                self.readme_list.append("    %s: %s: %3.3f\n" %
                                        (group, skey, self.per_group_statistics[group][skey]))

    def plot_results(self): # TODO: this has different args from super
        SingleFit.plot_results(self)
        self.get_plotting_dict()
        group_notelist = list()
        if self.plot_filter_out is not None:
            group_notelist.append("Data not shown:")
            for feature, value, threshold in self.plot_filter_out:
                group_notelist.append("  %s %s %s" % (feature, value, threshold))
        if self.plot_filter_out is None:
            group_notelist.append("RMSEs for overall fit:")
            group_notelist.append("Overall: %3.3f" % self.statistics['rmse'])
        else:
            group_notelist.append("RMSEs for shown data:")
            group_notelist.append("Overall: %3.3f" % self.statistics['filtered_rmse'])
        self.plot_group_splits_with_outliers(group_dict=dict(self.plotting_dict),
                                             outlying_groups=list(self.outlying_groups),
                                             label="per_group_info",
                                             group_notelist=list(group_notelist))
        self.readme_list.append("Plot in subfolder per_group_info created\n")
        self.readme_list.append("    labeling outlying groups and their RMSEs.\n")

    def get_per_group_statistics(self, preface=""):
        gfeat = self.testing_dataset.grouping_feature
        for group in self.testing_dataset.groups:
            g_ypredict = self.testing_dataset.target_prediction[self.testing_dataset.data[gfeat] == group]
            g_ydata = self.testing_dataset.target_data[self.testing_dataset.data[gfeat] == group]
            #g_mean_error = np.mean(g_ypredict - g_ydata)
            g_rmse = np.sqrt(mean_squared_error(g_ypredict, g_ydata))
            if group not in self.per_group_statistics:
                self.per_group_statistics[group] = dict()
            self.per_group_statistics[group]['%srmse' % preface] = g_rmse

    def get_outlying_groups(self):
        self.outlying_groups = list()
        highest_rmses = list()
        num_mark = min(self.mark_outlying_groups, len(self.per_group_statistics.keys()))
        for _ in range(num_mark):
            highest_rmses.append((0, "nogroup"))
        if self.plot_filter_out is None:
            criterion = 'rmse'
        else:
            criterion = 'filtered_rmse'
        for group in self.per_group_statistics:
            min_entry = min(highest_rmses)
            min_rmse = min_entry[0]
            if criterion in self.per_group_statistics[group]:
                g_rmse = self.per_group_statistics[group][criterion]
                if g_rmse > min_rmse:
                    highest_rmses[highest_rmses.index(min_entry)] = (g_rmse, group)
        logging.debug("Highest %s list: %s" % (criterion, highest_rmses))
        for high_rmse in highest_rmses:
            self.outlying_groups.append(high_rmse[1])

    def get_plotting_dict(self):
        plot_dict = dict()
        if self.plot_filter_out is None:
            criterion = 'rmse'
        else:
            criterion = 'filtered_rmse'
        gfeat = self.testing_dataset.grouping_feature
        for group in self.testing_dataset.groups:
            g_ypredict = self.testing_dataset.target_prediction[self.testing_dataset.data[gfeat] == group]
            g_ydata = self.testing_dataset.target_data[self.testing_dataset.data[gfeat] == group]
            if self.testing_dataset.target_error_data is None:
                g_ydata_err = np.zeros(len(g_ydata))
            else:
                g_ydata_err = self.testing_dataset.target_error_data[self.testing_dataset.data[gfeat] == group]
            plot_dict[group] = dict()
            plot_dict[group]['xdata'] = g_ydata
            plot_dict[group]['xerrdata'] = g_ydata_err
            plot_dict[group]['ydata'] = g_ypredict
            if criterion in self.per_group_statistics[group]:
                plot_dict[group]['rmse'] = self.per_group_statistics[group][criterion]
            else:
                plot_dict[group]['rmse'] = None
        self.plotting_dict = dict(plot_dict)

    @timeit
    def plot_group_splits_with_outliers(self, group_dict=None, outlying_groups=None,
                                        label="group_splits", group_notelist=None):
        myph = PlotHelper(
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            save_path=os.path.join(self.save_path, label),
            guideline=1,
            group_dict=group_dict,
            outlying_groups=(outlying_groups or []),
            plotlabel=label,
            notelist=(group_notelist or []))
        myph.plot_group_splits_with_outliers()
