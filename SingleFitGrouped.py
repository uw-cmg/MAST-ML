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
from SingleFit import timeit
import logging

class SingleFitGrouped(SingleFit):
    """Do a single full fit and split out group contributions to RMSE.

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
        stepsize, see parent class
        grouping_feature <str>: feature name for grouping data
        mark_outlying_groups <int>: Number of outlying groups to mark

    Returns:
        Analysis in the save_path folder
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
        grouping_feature = None,
        mark_outlying_groups = 2,
        *args, **kwargs):
        """
        Additional class attributes to parent class:
            self.grouping_feature <str>: Grouping feature
            self.mark_outlying_groups <int>: Number of outlying groups to mark.
                                If greater than the number of groups,
                                all groups will be marked separately.
        """
        SingleFit.__init__(self, 
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
            stepsize=stepsize)
        if grouping_feature is None:
            raise ValueError("grouping_feature is not set.")
        self.grouping_feature = grouping_feature
        self.mark_outlying_groups = int(mark_outlying_groups)
        # Sets later in code
        self.test_group_data = None
        self.test_group_indices = None
        self.test_groups =None
        self.per_group_statistics = dict()
        self.outlying_groups = list()
        self.plotting_dict = dict()
        return
    
    def set_data(self):
        SingleFit.set_data(self)
        if self.testing_target_data is None:
            raise ValueError("testing target data cannot be None")
        return

    def get_statistics(self):
        SingleFit.get_statistics(self)
        self.set_group_info()
        self.get_per_group_statistics()
        self.get_outlying_groups()
        return

    def print_statistics(self):
        SingleFit.print_statistics(self)
        self.readme_list.append("Per-group RMSEs from overall fit:\n")
        for group in self.test_groups:
            self.readme_list.append("    %s: %3.3f\n" % (group, self.per_group_statistics[group]))
        return

    def plot_results(self):
        SingleFit.plot_results(self)
        self.get_plotting_dict()
        self.plot_group_splits_with_outliers(group_dict=dict(self.plotting_dict), outlying_groups=list(self.outlying_groups), label="per_group_info", group_notelist=["RMSEs for overall fit:"])
        self.readme_list.append("Plot in subfolder per_group_info created\n")
        self.readme_list.append("    labeling outlying groups and their RMSEs.\n")
        return
    
    def set_group_info(self):
        self.test_group_data = np.asarray(self.testing_dataset.get_data(self.grouping_feature)).ravel()
        self.test_group_indices = gttd.get_logo_indices(self.test_group_data)
        self.test_groups = list(self.test_group_indices.keys())
        return

    def get_per_group_statistics(self):
        for group in self.test_groups:
            g_index = self.test_group_indices[group]["test_index"]
            g_ypredict= self.testing_target_prediction[g_index]
            g_ydata = self.testing_target_data[g_index]
            #g_mean_error = np.mean(g_ypredict - g_ydata)
            g_rmse = np.sqrt(mean_squared_error(g_ypredict, g_ydata))
            self.per_group_statistics[group] = g_rmse
        return

    def get_outlying_groups(self):
        self.outlying_groups = list()
        highest_rmses = list()
        num_mark = min(self.mark_outlying_groups, len(self.test_groups))
        for oidx in range(0, num_mark):
            highest_rmses.append((0, "nogroup"))
        for group in self.test_groups:
            min_entry = min(highest_rmses)
            min_rmse = min_entry[0]
            g_rmse = self.per_group_statistics[group]
            if g_rmse > min_rmse:
                highest_rmses[highest_rmses.index(min_entry)]= (g_rmse, group)
        logging.debug("Highest RMSEs: %s" % highest_rmses)
        for high_rmse in highest_rmses:
            self.outlying_groups.append(high_rmse[1])
        return

    def get_plotting_dict(self):
        plot_dict=dict()
        for group in self.test_groups:
            g_index = self.test_group_indices[group]["test_index"]
            g_ypredict= self.testing_target_prediction[g_index]
            g_ydata = self.testing_target_data[g_index]
            if self.testing_target_data_error is None:
                g_ydata_err = np.zeros(len(g_index))
            else:
                g_ydata_err = self.testing_target_data_error[g_index]
            plot_dict[group] = dict()
            plot_dict[group]['xdata'] = g_ydata
            plot_dict[group]['xerrdata'] = g_ydata_err
            plot_dict[group]['ydata'] = g_ypredict
            plot_dict[group]['rmse'] = self.per_group_statistics[group]
        self.plotting_dict=dict(plot_dict)
        return

    @timeit
    def plot_group_splits_with_outliers(self, group_dict=None, outlying_groups=list(), label="group_splits", group_notelist=list()):
        addl_kwargs=dict()
        addl_kwargs['xlabel'] = self.xlabel
        addl_kwargs['ylabel'] = self.ylabel
        addl_kwargs['save_path'] = os.path.join(self.save_path, label)
        addl_kwargs['stepsize'] = self.stepsize
        addl_kwargs['guideline'] = 1
        plotdict.plot_group_splits_with_outliers(group_dict = dict(group_dict),
            outlying_groups = list(outlying_groups),
            label=label, 
            group_notelist=list(group_notelist),
            addl_kwargs = dict(addl_kwargs))
        self.print_overlay_readme(label=label)
        return
    
    @timeit
    def print_overlay_readme(self, label=""):
        rlist=list()
        rlist.append("----- Folder contents -----\n")
        rlist.append("%s.png\n" % label)
        rlist.append("    Overlayed plot.\n")
        rlist.append("    See containing folder's README for more detail.\n")
        rlist.append("data_{series}.csv\n")
        rlist.append("    Data for each series in the plot:\n")
        rlist.append("    x data column\n")
        rlist.append("    x error column (all zeros is no error)\n")
        rlist.append("    y data column, labeled as the series in the plot\n")
        with open(os.path.join(self.save_path,label,"README"), 'w') as rfile:
            rfile.writelines(rlist)
        return

    @timeit
    def get_group_analysis_dict(self):
        all_groups = list(self.train_groups)
        all_groups.extend(list(self.test_groups))
        for group in all_groups:
            if not(group in self.train_group_indices.keys()):
                logging.info("Skipping group %s not in training data" % group)
                continue
            if not(group in self.test_group_indices.keys()):
                logging.info("Skipping group %s not in testing data" % group)
                continue
            self.group_analysis_dict[group] = self.do_single_fit(label=group,
                                                group=group)
        return

    @timeit
    def get_split_group_dict(self):
        osg_dict=dict()
        highest_rmses=list()
        num_mark = min(self.mark_outlying_groups, len(self.test_groups))
        for oidx in range(0, num_mark):
            highest_rmses.append((0, "nogroup"))
        sgdict=dict()
        for group in self.group_analysis_dict.keys():
            g_analysis = self.group_analysis_dict[group] #SingleFit obj.
            sgdict[group]=dict()
            sgdict[group]['xdata'] = g_analysis.testing_target_data
            if self.target_error_feature is None:
                sgdict[group]['xerrdata'] = np.zeros(len(g_analysis.testing_target_data))
            else:
                sgdict[group]['xerrdata'] = g_analysis.testing_target_data_error
            sgdict[group]['ydata'] = g_analysis.testing_target_prediction
            g_rmse = g_analysis.statistics['rmse']
            sgdict[group]['rmse'] = g_rmse
            min_entry = min(highest_rmses)
            min_rmse = min_entry[0]
            if g_rmse > min_rmse:
                highest_rmses[highest_rmses.index(min_entry)]= (g_rmse, group)
        logging.debug("Highest RMSEs: %s" % highest_rmses)
        self.split_group_dict=dict(sgdict)
        for high_rmse in highest_rmses:
            self.split_outlying_groups.append(high_rmse[1])
        return

    @timeit
    def old_print_readme(self):
        rlist=list()
        rlist.append("----- Folder contents -----\n")
        rlist.append("single_fit\n")
        rlist.append("    Folder containing a single overall fit.\n")
        if self.grouping_feature is None:
            rlist.append("Other folders would appear if\n")
            rlist.append("    grouping_feature were to be set.\n")
        else:
            if self.overall_analysis.testing_target_data is None:
                rlist.append("<group> folders\n")
                rlist.append("    Individual group fits, training on\n")
                rlist.append("        a single group and testing on\n")
                rlist.append("        a single group.\n")
                rlist.append("    Only groups that appear in both the\n")
                rlist.append("        training data and the testing data\n")
                rlist.append("        have their own folder.\n")
                rlist.append("Overlayed plot folders would have appeared,\n")
                rlist.append("    but there is no testing data to plot.\n")
            else:
                rlist.append("overall_overlay\n")
                rlist.append("    The single overall fit, but with RMSE\n")
                rlist.append("        data split out by group contributions,\n")
                rlist.append("        comparing predicted and measured data\n")
                rlist.append("        within each group.\n")
                rlist.append("    Predictions are made using a model fitted\n")
                rlist.append("        to all of the data at once.\n")
                rlist.append("split_overlay\n")
                rlist.append("    Predicted vs. measured data and RMSEs\n")
                rlist.append("        for fits to each individual group.\n")
                rlist.append("    The plot overlays each of these fits.\n")
                rlist.append("<group> folders\n")
                rlist.append("    Individual group fits, which are overlayed\n")
                rlist.append("        in the split_overlay folder.\n")
        with open(os.path.join(self.save_path,"README"), 'w') as rfile:
            rfile.writelines(rlist)
        return

    @timeit
    def old_run(self):
        self.overall_analysis = self.do_single_fit()
        if self.grouping_feature is None: #no additional analysis to do
            return
        self.set_group_info()
        self.get_overall_group_dict()
        if self.overall_analysis.testing_target_data is None:
            pass  #can't make overlay plots because there is no predicted data
        else:
            self.plot_group_splits_with_outliers(group_dict=self.overall_group_dict,
                outlying_groups = self.overall_outlying_groups,
                label="overall_overlay", 
                group_notelist=["RMSEs for overall fit:",
                    "Overall: %3.2f" % self.overall_analysis.statistics['rmse']])
        self.get_group_analysis_dict()
        if self.overall_analysis.testing_target_data is None:
            pass #can't make overlay plots because there is no predicted data
        else:
            self.get_split_group_dict()
            self.plot_group_splits_with_outliers(group_dict=self.split_group_dict,
                outlying_groups = self.split_outlying_groups,
                label="split_overlay",
                group_notelist=["RMSEs for per-group fits:"])
        self.print_readme()
        return


