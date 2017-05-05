import matplotlib
import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import data_analysis.printout_tools as ptools
import plot_data.plot_predicted_vs_measured as plotpm
import plot_data.plot_xy as plotxy
import plot_data.plot_from_dict as plotdict
import portion_data.get_test_train_data as gttd
import os
from SingleFit import SingleFit
from SingleFitGrouped import SingleFitGrouped
from SingleFit import timeit
import logging
import copy

class SingleFitPerGroup(SingleFitGrouped):
    """Split out the data by groups and then do single fits on each group.

    Args:
        training_dataset,
        testing_dataset,
        model,
        save_path,
        input_features,
        target_feature,
        target_error_feature,
        labeling_features, 
        xlabel, 
        ylabel,
        stepsize, 
        plot_filter_out, 
        grouping_feature,
        mark_outlying_groups, see parent class

    Returns:
        Analysis in the save_path folder
        Single fits for each group in group name folders
        Plots results in a predicted vs. measured square plot.

    Raises:
        ValueError if grouping_feature is not set
        ValueError if testing target data is None; has to have at least
                    some testing target data to plot
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
        plot_filter_out = None,
        grouping_feature = None,
        mark_outlying_groups = 2,
        *args, **kwargs):
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
            input_features = input_features, 
            target_feature = target_feature,
            target_error_feature = target_error_feature,
            labeling_features = labeling_features,
            xlabel=xlabel,
            ylabel=ylabel,
            stepsize=stepsize,
            plot_filter_out = plot_filter_out,
            grouping_feature = grouping_feature,
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
        for group in self.test_groups:
            group_training_dataset = copy.deepcopy(self.training_dataset)
            group_testing_dataset = copy.deepcopy(self.testing_dataset)
            group_training_dataset.add_exclusive_filter(self.grouping_feature, "<>", group)
            group_testing_dataset.add_exclusive_filter(self.grouping_feature, "<>", group)
            self.per_group_singlefits[group]=SingleFit(
                    training_dataset = group_training_dataset,
                    testing_dataset = group_testing_dataset,
                    model = self.model,
                    save_path = os.path.join(self.save_path, str(group)),
                    input_features = list(self.input_features),
                    target_feature = self.target_feature,
                    target_error_feature = self.target_error_feature,
                    labeling_features = list(self.labeling_features),
                    xlabel = self.xlabel,
                    ylabel = self.ylabel,
                    stepsize = self.stepsize,
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
        for group in self.test_groups:
            skeys = list(self.per_group_statistics[group].keys())
            skeys.sort()
            for skey in skeys:
                val = self.per_group_statistics[group][skey]
                if type(val) is None:
                    self.readme_list.append("    %s: %s: None\n" % (group, skey))
                else:
                    self.readme_list.append("    %s: %s: %3.3f\n" % (group, skey, val))
        return

    def plot_results(self):
        self.get_plotting_dict()
        group_notelist=list()
        if not(self.plot_filter_out is None):
            group_notelist.append("Data not shown:")
            for pfstr in self.plot_filter_out:
                group_notelist.append("  %s" % pfstr.replace(";"," "))
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
    
    def set_group_info(self):
        SingleFitGrouped.set_group_info(self)
        all_groups = np.union1d(self.train_groups, self.test_groups)
        plot_groups = np.intersect1d(self.train_groups, self.test_groups) #have both training and testing data
        for group in all_groups:
            if group not in plot_groups:
                logging.info("Skipping group %s for futher analysis because training or testing data is missing." % group)
        self.train_groups = plot_groups #Can only train if have training data
        self.test_groups = plot_groups #Can only test if have testing data
        return

    def get_per_group_statistics(self):
        for group in self.test_groups: 
            self.per_group_statistics[group] = dict(self.per_group_singlefits[group].statistics)
        return

    def get_plotting_dict(self):
        plot_dict=dict()
        if self.plot_filter_out is None:
            criterion = 'rmse'
        else:
            criterion = 'rmse_plot_filter_out'
        for group in self.test_groups:
            g_singlefit = self.per_group_singlefits[group]
            if not(g_singlefit.plotting_index is None):
                plot_index = g_singlefit.plotting_index
            else:
                plot_index = np.arange(0, len(g_singlefit.testing_target_prediction))
            g_ypredict= g_singlefit.testing_target_prediction[plot_index]
            g_ydata = g_singlefit.testing_target_data[plot_index]
            if g_singlefit.testing_target_data_error is None:
                g_ydata_err = np.zeros(len(g_ydata))
            else:
                g_ydata_err = g_singlefit.testing_target_data_error[plot_index]
            plot_dict[group] = dict()
            plot_dict[group]['xdata'] = g_ydata
            plot_dict[group]['xerrdata'] = g_ydata_err
            plot_dict[group]['ydata'] = g_ypredict
            plot_dict[group]['rmse'] = g_singlefit.statistics[criterion]
        self.plotting_dict=dict(plot_dict)
        return
