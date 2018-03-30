__author__ = 'Tam Mayeshiba'
__maintainer__ = 'Ryan Jacobs'
__version__ = '1.0'
__email__ = 'rjacobs3@wisc.edu'
__date__ = 'October 14th, 2017'

import numpy as np
import os
from SingleFit import SingleFit
from SingleFitGrouped import SingleFitGrouped
from SingleFit import timeit
import logging
import copy

class SingleFitPerGroup(SingleFitGrouped):
    """Class used to split out the data by groups and then do single fits on each group.

    Args:
        training_dataset (DataHandler object): Training dataset handler
        testing_dataset (DataHandler object): Testing dataset handler
        model (sklearn model object): sklearn model
        save_path (str): Save path
        xlabel (str): Label for full-fit x-axis (default "Measured")
        ylabel (str): Label for full-fit y-axis (default "Predicted")
        plot_filter_out (list): List of semicolon-delimited strings with feature;operator;value for leaving out specific values for plotting.

        mark_outlying_groups (int): Number of outlying groups to mark

    Returns:
        Analysis in the save_path folder
        Single fits for each group in group name folders
        Plots results in a predicted vs. measured square plot.

    Raises:
        ValueError if testing dataset grouping_feature is not set
        ValueError if testing target data is None; has to have at least some testing target data to plot

    """
    def __init__(self, training_dataset=None, testing_dataset=None, model=None, save_path=None, xlabel="Measured",
        ylabel="Predicted", plot_filter_out = None, mark_outlying_groups = 2, *args, **kwargs):
        """
        Additional class attributes to parent class:
        self.all_groups = list()
        self.plot_groups = list()
        self.per_group_singlefits = dict()
        """
        SingleFitGrouped.__init__(self, 
            training_dataset=training_dataset, 
            testing_dataset=testing_dataset,
            model=model, 
            save_path = save_path,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_filter_out = plot_filter_out,
            mark_outlying_groups = mark_outlying_groups,
            fit_only_on_matched_groups = 0)
        # Sets later in code
        self.per_group_singlefits = dict()
        return
    
    @timeit
    def predict(self):
        self.get_group_predictions()
        self.get_statistics()
        self.print_statistics()
        return

    @timeit
    def plot(self):
        self.plot_results()
        return

    def get_group_predictions(self):
        for group in self.testing_dataset.groups:
            if not group in self.training_dataset.groups:
                logging.warning("Skipping group %s not in training groups." % group)
                continue #must have training data to do a fit
            gfeat = self.testing_dataset.grouping_feature
            group_training_dataset = copy.deepcopy(self.training_dataset)
            group_testing_dataset = copy.deepcopy(self.testing_dataset)
            group_training_dataset.data = group_training_dataset.data[group_training_dataset.data[gfeat] == group]
            group_testing_dataset.data = group_testing_dataset.data[group_testing_dataset.data[gfeat] == group]
            group_training_dataset.set_up_data_from_features()
            group_testing_dataset.set_up_data_from_features()
            self.per_group_singlefits[group]=SingleFit(
                    training_dataset = group_training_dataset,
                    testing_dataset = group_testing_dataset,
                    model = self.model,
                    save_path = os.path.join(self.save_path, str(group)),
                    xlabel = self.xlabel,
                    ylabel = self.ylabel,
                    plot_filter_out = self.plot_filter_out)
            self.per_group_singlefits[group].run()
        return


    def get_statistics(self):
        self.get_per_group_statistics()
        self.get_outlying_groups()
        return

    def print_statistics(self):
        self.readme_list.append("----- Statistics -----\n")
        self.readme_list.append("RMSEs from individual group fits:\n")
        groups = list(self.per_group_statistics.keys())
        groups.sort()
        for group in groups:
            skeys = list(self.per_group_statistics[group].keys())
            skeys.sort()
            for skey in skeys:
                val = self.per_group_statistics[group][skey]
                if type(val) is None:
                    self.readme_list.append("    %s: %s: None\n" % (group, skey))
                if type(val) is float():
                    self.readme_list.append("    %s: %s: %3.3f\n" % (group, skey, val))
                if type(val) is str():
                    self.readme_list.append("    %s: %s: %s\n" % (group, skey, val))
        return

    def plot_results(self):
        self.get_plotting_dict()
        group_notelist=list()
        if not(self.plot_filter_out is None):
            group_notelist.append("Data not shown:")
            for (feature, symbol, threshold) in self.plot_filter_out:
                group_notelist.append("  %s %s %s" % (feature, symbol, threshold))
        if self.plot_filter_out is None:
            group_notelist.append("RMSEs for individual fits:")
        else:
            group_notelist.append("Shown-data RMSEs for individual fits:")
        self.plot_group_splits_with_outliers(group_dict=dict(self.plotting_dict), 
                outlying_groups=list(self.outlying_groups), 
                label="per_group_fits_overlay", 
                group_notelist=list(group_notelist))
        self.readme_list.append("----- Plotting -----\n")
        self.readme_list.append("Plot in subfolder per_group_fits_overlay created,\n")
        self.readme_list.append("    labeling worst-fitting groups and their RMSEs.\n")
        return
    
    def get_per_group_statistics(self):
        for group in self.per_group_singlefits.keys(): 
            self.per_group_statistics[group] = dict(self.per_group_singlefits[group].statistics)
        return

    def get_plotting_dict(self):
        plot_dict=dict()
        if self.plot_filter_out is None:
            criterion = 'rmse'
        else:
            criterion = 'filtered_rmse'
        for group in self.per_group_singlefits.keys():
            g_singlefit = self.per_group_singlefits[group]
            g_ypredict= g_singlefit.testing_dataset.target_prediction
            g_ydata = g_singlefit.testing_dataset.target_data
            if g_singlefit.testing_dataset.target_error_data is None:
                g_ydata_err = np.zeros(len(g_ydata))
            else:
                g_ydata_err = g_singlefit.testing_dataset.target_error_data
            plot_dict[group] = dict()
            plot_dict[group]['xdata'] = g_ydata
            plot_dict[group]['xerrdata'] = g_ydata_err
            plot_dict[group]['ydata'] = g_ypredict
            plot_dict[group]['rmse'] = g_singlefit.statistics[criterion]
        self.plotting_dict=dict(plot_dict)
        return
