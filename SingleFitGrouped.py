import numpy as np
from sklearn.metrics import mean_squared_error
from plot_data.PlotHelper import PlotHelper
import os
from SingleFit import SingleFit
from SingleFit import timeit
import logging

class SingleFitGrouped(SingleFit):
    """Do a single full fit and split out group contributions to RMSE.

    Args:
        training_dataset,
        testing_dataset,
        model,
        save_path,
        xlabel, 
        ylabel,
        plot_filter_out, see parent class
        mark_outlying_groups (int): Number of outlying groups to mark
        fit_only_on_matched_groups (int): 0 - fit on all data in the training
                                                dataset (default)
                                          1 - fit only on groups in the training
                                                dataset that are also in the
                                                testing dataset

    Returns:
        Analysis in the save_path folder
        Plots results in a predicted vs. measured square plot.

    Raises:
        ValueError: if grouping_feature is not set in testing dataset
        ValueError: if testing target data is None; has to have at least
                    some testing target data to plot
    """
    def __init__(self, 
        training_dataset=None,
        testing_dataset=None,
        model=None,
        save_path=None,
        xlabel="Measured",
        ylabel="Predicted",
        plot_filter_out = None,
        mark_outlying_groups = 2,
        fit_only_on_matched_groups = 0,
        *args, **kwargs):
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
        SingleFit.__init__(self, 
            training_dataset=training_dataset, 
            testing_dataset=testing_dataset,
            model=model, 
            save_path = save_path,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_filter_out = plot_filter_out)
        if self.testing_dataset.grouping_feature is None:
            raise ValueError("grouping_feature is not set.")
        self.mark_outlying_groups = int(mark_outlying_groups)
        self.fit_only_on_matched_groups = int(fit_only_on_matched_groups)
        # Sets later in code
        self.per_group_statistics = dict()
        self.outlying_groups = list()
        self.plotting_dict = dict()
        return

    @timeit 
    def set_up(self):
        if self.testing_dataset.target_data is None:
            raise ValueError("testing target data cannot be None")
        SingleFit.set_up(self)
        if not (self.fit_only_on_matched_groups == 1):
            return
        out_training_groups = np.setdiff1d(self.training_dataset.groups, self.testing_dataset.groups)
        filter_list = list()
        for group in out_training_groups:
            filter_list.append((self.training_dataset.grouping_feature,"=",group))
        self.training_dataset.add_filters(filter_list)
        self.training_dataset.set_up_data_from_features() #reset all data
        return

    def get_statistics(self):
        self.get_per_group_statistics(preface="")
        SingleFit.get_statistics(self) #also adds plot filters
        if not(self.plot_filter_out is None):
            self.get_per_group_statistics(preface="filtered_")
        self.get_outlying_groups()
        return

    def print_statistics(self):
        SingleFit.print_statistics(self)
        self.readme_list.append("Per-group RMSEs from overall fit:\n")
        groups = list(self.per_group_statistics.keys())
        groups.sort()
        for group in groups:
            skeys = list(self.per_group_statistics[group].keys())
            skeys.sort()
            for skey in skeys:
                self.readme_list.append("    %s: %s: %3.3f\n" % (group, skey, self.per_group_statistics[group][skey]))
        return

    def plot_results(self):
        SingleFit.plot_results(self)
        self.get_plotting_dict()
        group_notelist=list()
        if not(self.plot_filter_out is None):
            group_notelist.append("Data not shown:")
            for (feature,value,threshold) in self.plot_filter_out:
                group_notelist.append("  %s %s %s" % (feature,value,threshold))
        if self.plot_filter_out is None:
            group_notelist.append("RMSEs for overall fit:")
            group_notelist.append("Overall: %3.3f" % self.statistics['rmse'])
        else:
            group_notelist.append("RMSEs for shown data:")
            group_notelist.append("Overall: %3.3f" % self.statistics['filtered_rmse'])
        self.plot_group_splits_with_outliers(group_dict=dict(self.plotting_dict), outlying_groups=list(self.outlying_groups), label="per_group_info", group_notelist=list(group_notelist))
        self.readme_list.append("Plot in subfolder per_group_info created\n")
        self.readme_list.append("    labeling outlying groups and their RMSEs.\n")
        return
    
    def get_per_group_statistics(self, preface=""):
        gfeat = self.testing_dataset.grouping_feature
        for group in self.testing_dataset.groups:
            g_ypredict= self.testing_dataset.target_prediction[self.testing_dataset.data[gfeat] == group]
            g_ydata = self.testing_dataset.target_data[self.testing_dataset.data[gfeat] == group]
            #g_mean_error = np.mean(g_ypredict - g_ydata)
            g_rmse = np.sqrt(mean_squared_error(g_ypredict, g_ydata))
            if not (group in self.per_group_statistics.keys()):
                self.per_group_statistics[group] = dict()
            self.per_group_statistics[group]['%srmse' % preface] = g_rmse
        return

    def get_outlying_groups(self):
        self.outlying_groups = list()
        highest_rmses = list()
        num_mark = min(self.mark_outlying_groups, len(self.per_group_statistics.keys()))
        for oidx in range(0, num_mark):
            highest_rmses.append((0, "nogroup"))
        if self.plot_filter_out is None:
            criterion = 'rmse'
        else:
            criterion = 'filtered_rmse'
        for group in self.per_group_statistics.keys():
            min_entry = min(highest_rmses)
            min_rmse = min_entry[0]
            if criterion in self.per_group_statistics[group].keys():
                g_rmse = self.per_group_statistics[group][criterion]
                if g_rmse > min_rmse:
                    highest_rmses[highest_rmses.index(min_entry)]= (g_rmse, group)
        logging.debug("Highest %s list: %s" % (criterion, highest_rmses))
        for high_rmse in highest_rmses:
            self.outlying_groups.append(high_rmse[1])
        return

    def get_plotting_dict(self):
        plot_dict=dict()
        if self.plot_filter_out is None:
            criterion = 'rmse'
        else:
            criterion = 'filtered_rmse'
        gfeat = self.testing_dataset.grouping_feature
        for group in self.testing_dataset.groups:
            g_ypredict= self.testing_dataset.target_prediction[self.testing_dataset.data[gfeat] == group]
            g_ydata = self.testing_dataset.target_data[self.testing_dataset.data[gfeat] == group]
            if self.testing_dataset.target_error_data is None:
                g_ydata_err = np.zeros(len(g_ydata))
            else:
                g_ydata_err = self.testing_dataset.target_error_data[self.testing_dataset.data[gfeat] == group]
            plot_dict[group] = dict()
            plot_dict[group]['xdata'] = g_ydata
            plot_dict[group]['xerrdata'] = g_ydata_err
            plot_dict[group]['ydata'] = g_ypredict
            if criterion in self.per_group_statistics[group].keys():
                plot_dict[group]['rmse'] = self.per_group_statistics[group][criterion]
            else:
                plot_dict[group]['rmse'] = None
        self.plotting_dict=dict(plot_dict)
        return

    @timeit
    def plot_group_splits_with_outliers(self, group_dict=None, outlying_groups=list(), label="group_splits", group_notelist=list()):
        addl_kwargs=dict()
        addl_kwargs['xlabel'] = self.xlabel
        addl_kwargs['ylabel'] = self.ylabel
        addl_kwargs['save_path'] = os.path.join(self.save_path, label)
        addl_kwargs['guideline'] = 1
        addl_kwargs['group_dict'] = group_dict
        addl_kwargs['outlying_groups'] = list(outlying_groups)
        addl_kwargs['plotlabel'] = label
        addl_kwargs['notelist'] = list(group_notelist)
        myph = PlotHelper(**addl_kwargs)
        myph.plot_group_splits_with_outliers()
        return
    
