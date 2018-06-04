__author__ = 'Tam Mayeshiba'
__maintainer__ = 'Ryan Jacobs'
__version__ = '1.0'
__email__ = 'rjacobs3@wisc.edu'
__date__ = 'October 14th, 2017'

from plot_data.PlotHelper import PlotHelper
import os
from MLTests.SingleFit import SingleFit
from MLTests.SingleFit import timeit

class PlotNoAnalysis(SingleFit):
    """Class to conduct data plotting, but no data analysis

    Args:
        training_dataset (DataHandler object): Training dataset handler
        testing_dataset (DataHandler object): Testing dataset handler
        model (sklearn model object): sklearn model
        save_path (str): Save path
        xlabel (str): Label for full-fit x-axis (default "Measured")
        ylabel (str): Label for full-fit y-axis (default "Predicted")
        plot_filter_out (list): List of semicolon-delimited strings with feature;operator;value for leaving out specific values for plotting.
        feature_plot_feature (str): Feature to plot on x-axis, against target feature
        data_labels (list of str): Dataset labels

    Returns:
        Plots target data against feature data, by group if a grouping feature is set

    Raises:
        ValueError: if
            feature_plot_feature is None
            there is no testing target data
            data_labels is None

    """
    def __init__(self, training_dataset=None, testing_dataset=None, model=None, save_path=None, xlabel="X",
            ylabel="Y", plot_filter_out=None, feature_plot_feature=None, data_labels=None):
        """
        Additional attributes to parent class:
            self.testing_datasets <list>: List of testing datasets which
                        will be plotted simultaneously
            self.feature_plot_feature <str>: Feature to plot on x-axis
            self.data_labels <list of str>: Labels for datasets
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
        self.testing_datasets = testing_dataset #full list
        for tidx in range(0, len(self.testing_datasets)):
            testing_dataset = self.testing_datasets[tidx]
            if testing_dataset.target_data is None:
                raise ValueError("No testing target data for dataset %i" % tidx)
        self.feature_plot_feature = feature_plot_feature
        if data_labels is None:
            raise ValueError("data_labels is not set")
        self.data_labels = None
        if type(data_labels) is str:
            self.data_labels = [data_labels]
        elif type(data_labels) is list:
            self.data_labels = data_labels

    @timeit
    def run(self):
        self.set_up()
        self.plot()
        self.print_readme()
        self.print_output_csv()

    @timeit
    def print_output_csv(self):
        self.readme_list.append("----- Output data -----\n")
        for tidx in range(0, len(self.data_labels)):
            csvname = "output_%s.csv" % self.data_labels[tidx]
            testing_dataset = self.testing_datasets[tidx]
            ocsvname = os.path.join(self.save_path, csvname)
            cols_printed = testing_dataset.print_data(ocsvname)
            self.readme_list.append("%s file created with columns:\n" % csvname)
            for col in cols_printed:
                self.readme_list.append("    %s\n" % col)

    @timeit
    def plot(self):
        if not(self.plot_filter_out is None):
            self.readme_list.append("Plot filtering out:\n")
            for (feature, symbol, threshold) in self.plot_filter_out:
                self.readme_list.append("  %s %s %s\n" % (feature, symbol, threshold))
        self.one_plot(group=None)
        if self.testing_dataset.grouping_feature is None: #same for all datasets
            return
        for group in self.testing_dataset.groups:
            self.one_plot(group=group)

    @timeit
    def set_up(self):
        for testing_dataset in self.testing_datasets:
            testing_dataset.add_filters(self.plot_filter_out)

    def one_plot(self, group=None):
        xdatalist=list()
        xerrlist=list()
        ydatalist=list()
        yerrlist=list()
        for testing_dataset in self.testing_datasets:
            xdata = testing_dataset.data[self.feature_plot_feature]
            ydata = testing_dataset.target_data
            yerrdata = testing_dataset.target_error_data
            xerrdata = None
            if not (group is None):
                xdata = xdata[testing_dataset.group_data == group]
                ydata = ydata[testing_dataset.group_data == group]
                if len(xdata) == 0:
                    xdata = list()
                if len(ydata) == 0:
                    ydata = list()
                if not (yerrdata is None):
                    yerrdata = yerrdata[testing_dataset.group_data == group]
                    if len(yerrdata) == 0:
                        yerrdata = None
            xdatalist.append(xdata)
            xerrlist.append(xerrdata)
            ydatalist.append(ydata)
            yerrlist.append(yerrdata)
        kwargs=dict()
        if group is None:
            kwargs['plotlabel'] = "all"
        if not (group is None):
            kwargs['plotlabel'] = group
        kwargs['xdatalist'] = xdatalist
        kwargs['ydatalist'] = ydatalist
        kwargs['xerrlist'] = xerrlist
        kwargs['yerrlist'] = yerrlist
        kwargs['labellist'] = self.data_labels
        #kwargs['faces'] = ["None"] * len(self.data_labels)
        kwargs['save_path'] = self.save_path
        kwargs['xlabel'] = self.xlabel
        kwargs['ylabel'] = self.ylabel
        notelist=list()
        #if not(self.plot_filter_out is None):
        #    notelist.append("Data not shown:")
        #    for (feature, symbol, threshold) in self.plot_filter_out:
        #        notelist.append("  %s %s %s" % (feature, symbol, threshold))
        kwargs['notelist'] = notelist
        myph = PlotHelper(**kwargs)
        myph.multiple_overlay()
        #if do_fft == 1:
        #    plotxy.single(xdata, fft(ydata), **kwargs)
