import matplotlib
import matplotlib.pyplot as plt
import data_parser
import numpy as np
import data_analysis.printout_tools as ptools
import plot_data.plot_xy as plotxy
import os
import portion_data.get_test_train_data as gttd
from SingleFit import SingleFit
from SingleFit import timeit
###########
# Just plotting, no analysis
# Tam Mayeshiba 2017-03-24
###########

class PlotNoAnalysis(SingleFit):
    """Plotting without analysis
    Args:
        training_dataset, (same as testing_dataset; only testing_dataset used)
        testing_dataset, (same as training_dataset; only testing_dataset used)
        model, (not used)
        save_path,
        xlabel,
        ylabel,
        plot_filter_out, see parent class.
        feature_plot_feature <str>: Feature to plot on x-axis, against target feature
        timex <str>: Time format string, if x-axis is a time
    Returns:
        Plots target data against feature data, by group if a grouping feature
            is set
    Raises:
        ValueError if feature_plot_feature is None or there is no testing
            target data
    """
    def __init__(self,
            training_dataset=None,
            testing_dataset=None,
            model=None,
            save_path=None,
            xlabel="X",
            ylabel="Y",
            plot_filter_out=None,
            feature_plot_feature=None,
            timex=None):
        """
        Additional attributes to parent class:
            self.feature_plot_feature <str>: Feature to plot on x-axis
            self.timex <str>: Time format string, if x-axis is a time
        """
        SingleFit.__init__(self,
                training_dataset = training_dataset,
                testing_dataset = testing_dataset,
                model = model,
                save_path = save_path,
                xlabel = xlabel,
                ylabel = ylabel,
                plot_filter_out = plot_filter_out)
        if feature_plot_feature is None:
            raise ValueError("feature_plot_feature is not set")
        if self.testing_dataset.target_data is None:
            raise ValueError("No testing target data.")
        self.feature_plot_feature = feature_plot_feature
        self.timex = timex
        return

    @timeit
    def run(self):
        self.testing_dataset.add_filters(self.plot_filter_out)
        if not(self.plot_filter_out is None):
            self.readme_list.append("Plot filtering out:\n")
            for (feature, symbol, threshold) in self.plot_filter_out:
                self.readme_list.append("  %s %s %s\n" % (feature, symbol, threshold))
        self.plot_single(group=None)
        if self.testing_dataset.grouping_feature is None:
            return
        for group in self.testing_dataset.groups:
            self.plot_single(group=group)
        return

    def plot_single(self, group=None):
        xdata = self.testing_dataset.data[self.feature_plot_feature]
        ydata = self.testing_dataset.target_data
        kwargs=dict()
        if group is None:
            kwargs['plotlabel'] = "all"
        if not (group is None):
            xdata = xdata[self.testing_dataset.group_data == group]
            ydata = ydata[self.testing_dataset.group_data == group]
            kwargs['plotlabel'] = group
        kwargs['save_path'] = self.save_path
        kwargs['timex'] = self.timex
        kwargs['xlabel'] = self.xlabel
        kwargs['ylabel'] = self.ylabel
        notelist=list()
        if not(self.plot_filter_out is None):
            notelist.append("Data not shown:")
            for (feature, symbol, threshold) in self.plot_filter_out:
                notelist.append("  %s %s %s" % (feature, symbol, threshold))
        kwargs['notelist'] = notelist
        plotxy.single(xdata, ydata, **kwargs)
        #if do_fft == 1:
        #    plotxy.single(xdata, fft(ydata), **kwargs)
        return
