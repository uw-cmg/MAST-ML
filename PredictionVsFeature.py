__author__ = 'Tam Mayeshiba'
__maintainer__ = 'Ryan Jacobs'
__version__ = '1.0'
__email__ = 'rjacobs3@wisc.edu'
__date__ = 'October 14th, 2017'

import numpy as np
import os
from SingleFit import timeit
from SingleFit import SingleFit
from plot_data.PlotHelper import PlotHelper
import copy
import logging

class PredictionVsFeature(SingleFit):
    """Class used to make plots of predicted values versus a specific feature, using a single fit.

    Args:
        training_dataset (DataHandler object): Training dataset handler
        testing_dataset (DataHandler object): Testing dataset handler
        model (sklearn model object): sklearn model
        save_path (str): Save path
        xlabel (str): Label for full-fit x-axis (default "Measured")
        ylabel (str): Label for full-fit y-axis (default "Predicted")
        plot_filter_out (list): List of semicolon-delimited strings with feature;operator;value for leaving out specific values for plotting.

        feature_plot_feature (str): feature for per-group feature plots
        feature_plot_xlabel (str): x-axis label for per-group plots of predicted and measured y data versus data from field of numeric_field_name
        feature_plot_ylabel (str): y-axis label for per-group plots
        markers (str): comma-delimited marker list for split plots
        outlines (str): comma-delimited color list for split plots
        linestyles (str): comma-delimited list of line styles for split plots
        sizes (str): comma-delimited list of sizes for split plots
        data_labels (str): comma-delimited list of testing dataset labels for split plots

    Returns:
        Analysis in save_path folder

    Raises:
        ValueError: if feature_plot_feature is not set

    """
    def __init__(self, training_dataset=None, testing_dataset=None, model=None, save_path=None, xlabel="Measured",
        ylabel="Predicted", plot_filter_out = "", feature_plot_xlabel = "X", feature_plot_ylabel = "Prediction",
        feature_plot_feature = None, markers="", outlines="", data_labels="", linestyles="", legendloc=None,
        sizes="", *args, **kwargs):
        """
            Additional class attributes not in parent class:

            Set by keyword:
            self.testing_datasets <list of data objects>: testing datasets
            self.feature_plot_xlabel <str>: x-axis label for feature plots
            self.feature_plot_ylabel <str>: y-axis label for feature plots
            self.feature_plot_feature <str>: feature for feature plots
            self.markers <list of str>: list of markers for plotting
            self.outlines <list of str>: list of edge colors for plotting
            self.linestyles <list of str>: list of linestyles for plotting
            self.data_labels <list of str>: list of data labels for plotting
            self.legendloc <str>: legend location (optional)
            self.sizes <list of str>: list of sizes for plotting

            Set by code:
            self.sf_dict <dict of SingleFit objects>: dict of SingleFit objects,
                                    with keys as testing dataset names
        """
        SingleFit.__init__(self,
            training_dataset=training_dataset,
            testing_dataset=testing_dataset,
            model=model,
            save_path = save_path,
            xlabel=xlabel,
            ylabel=ylabel,
            plot_filter_out = plot_filter_out)
        #Sets by keyword
        self.testing_datasets = testing_dataset #expect a list of one or more testing datasets
        self.feature_plot_xlabel = feature_plot_xlabel
        self.feature_plot_ylabel = feature_plot_ylabel
        if feature_plot_feature is None:
            raise ValueError("feature plot's x feature is not set in feature_plot_feature")
        self.feature_plot_feature = feature_plot_feature
        self.markers = markers
        self.outlines = outlines
        self.data_labels = data_labels
        self.linestyles = linestyles
        self.legendloc = legendloc
        self.sizes = sizes
        #Sets in code
        self.sf_dict = dict()

    @timeit
    def set_up(self):
        SingleFit.set_up(self)

    @timeit
    def predict(self):
        self.get_singlefits()

    def get_singlefits(self):
        """Get SingleFit for each testing dataset.
        """
        self.readme_list.append("----- Fitting and prediction -----\n")
        self.readme_list.append("See results per dataset in subfolders\n")
        tidxs = np.arange(0, len(self.testing_datasets))
        for tidx in tidxs:
            testset = self.data_labels[tidx]
            testdata = copy.deepcopy(self.testing_datasets[tidx])
            self.readme_list.append("  %s\n" % testset)
            self.sf_dict[testset] = SingleFit(
                training_dataset = self.training_dataset,
                testing_dataset = testdata,
                model = self.model,
                save_path = os.path.join(self.save_path, str(testset)),
                xlabel=self.xlabel,
                ylabel=self.ylabel,
                plot_filter_out = self.plot_filter_out)
            self.sf_dict[testset].run()

    @timeit
    def plot(self):
        self.readme_list.append("----- Plotting -----\n")
        self.readme_list.append("See plots in subfolders:\n")
        self.make_series_feature_plot(group=None)
        if self.training_dataset.grouping_feature is None:
            self.readme_list.append("Grouping feature is not set. No group plots to do.\n")
            return
        allgroups=list()
        for testset in self.sf_dict:
            for group in self.sf_dict[testset].testing_dataset.groups:
                allgroups.append(group)
        allgroups = np.unique(allgroups)
        for testset in self.sf_dict:
            if self.sf_dict[testset].testing_dataset.target_data is None:
                self.sf_dict[testset].testing_dataset.add_filters(self.plot_filter_out) #filters will not have been added by SingleFit yet
        for group in allgroups:
            self.make_series_feature_plot(group=group)

    @timeit
    def make_series_feature_plot(self, group=None):
        """Make series feature plot
        """
        plabel = "feature_plot_group_%s" % str(group)
        self.readme_list.append("  %s\n" % plabel)
        pdict=dict()
        group_notelist=list()
        if not(self.plot_filter_out is None):
            group_notelist.append("Data not displayed:")
            for (feature, symbol, threshold) in self.plot_filter_out:
                group_notelist.append("  %s %s %s" % (feature, symbol, threshold))
        testsets = list(self.sf_dict.keys())
        testsets.sort()
        for testset in testsets:
            ts_sf_td = self.sf_dict[testset].testing_dataset
            gfeat = self.training_dataset.grouping_feature
            if group is None:
                if ts_sf_td.target_data is None:
                    measured = None
                else:
                    measured = ts_sf_td.target_data
                feature_data = ts_sf_td.data[self.feature_plot_feature]
                predicted = ts_sf_td.target_prediction
            else:
                if ts_sf_td.target_data is None:
                    measured = None
                else:
                    measured = ts_sf_td.target_data[ts_sf_td.data[gfeat] == group]
                feature_data = ts_sf_td.data[self.feature_plot_feature][ts_sf_td.data[gfeat] == group]
                predicted = ts_sf_td.target_prediction[ts_sf_td.data[gfeat] == group]
            if measured is None:
                pass
            elif len(measured) == 0:
                pass
            else:
                series_label = "%s measured" % testset
                pdict[series_label] = dict()
                pdict[series_label]['xdata'] = feature_data
                pdict[series_label]['xerrdata'] = None
                pdict[series_label]['ydata'] = measured
            if len(predicted) == 0:
                pass
            else:
                series_label = "%s predicted" % testset
                pdict[series_label] = dict()
                pdict[series_label]['xdata'] = feature_data
                pdict[series_label]['xerrdata'] = None
                pdict[series_label]['ydata'] = predicted
        series_list = list(pdict.keys())
        if len(series_list) == 0:
            logging.info("No series for plot.")
            return
        faces = ["None"] * len(self.markers)
        addl_kwargs = {
            'guideline': 0,
            'save_path': os.path.join(self.save_path, plabel),
            'xlabel': self.feature_plot_xlabel,
            'ylabel': self.feature_plot_ylabel,
            'markers': self.markers,
            'outlines': self.outlines,
            'linestyles': self.linestyles,
            'legendloc': self.legendloc,
            'sizes': self.sizes,
            'faces': faces,
            'group_dict': pdict,
            'plotlabel': plabel,
            'outlying_groups': series_list,
            'notelist': list(group_notelist),
        }
        myph = PlotHelper(**addl_kwargs)
        myph.plot_group_splits_with_outliers()

