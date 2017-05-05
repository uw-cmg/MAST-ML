import matplotlib.pyplot as plt
import matplotlib
import data_parser
import numpy as np
import os
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import data_analysis.printout_tools as ptools
import portion_data.get_test_train_data as gttd
from SingleFit import timeit
from SingleFit import SingleFit
import plot_data.plot_xy as plotxy
import plot_data.plot_from_dict as plotdict
import copy
import time
import logging

class PredictionVsFeature(SingleFit):
    """Make prediction vs. feature plots from a single fit.

    Args:
        training_dataset,
        testing_dataset, (Multiple testing datasets are allowed as a list.)
        model,
        save_path,
        input_features,
        target_feature,
        target_error_feature,
        labeling_features,
        xlabel,
        ylabel,
        stepsize,
        plot_filter_out, see parent class.
        grouping_feature <str>: feature for grouping (optional)
        feature_plot_xlabel <str>: x-axis label for per-group plots of predicted and
                                measured y data versus data from field of
                                numeric_field_name
        feature_plot_ylabel <str>: y-axis label for per-group plots
        feature_plot_feature <str>: feature for per-group feature plots
        markers <str>: comma-delimited marker list for split plots
        outlines <str>: comma-delimited color list for split plots
        linestyles <str>: comma-delimited list of line styles for split plots
        data_labels <str>: comma-delimited list of testing dataset labels for split plots

    Returns:
        Analysis in save_path folder

    Raises:
        ValueError if feature_plot_feature is not set
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
        plot_filter_out = "",
        grouping_feature = None,
        feature_plot_xlabel = "X",
        feature_plot_ylabel = "Prediction",
        feature_plot_feature = None,
        markers="",
        outlines="",
        data_labels="",
        linestyles="",
        legendloc=None,
        sizes="",
        *args, **kwargs):
        """
            Additional class attributes not in parent class:
           
            Set by keyword:
            self.grouping_feature <str>: feature for grouping data (optional)
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
            self.testing_dataset_dict <dict of data objects>: testing datasets and their information:
                dataset
                feature_data
                group_indices (if self.grouping_feature is set)
                SingleFit (SingleFit object)
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
            stepsize=stepsize,
            plot_filter_out = plot_filter_out)
        #Sets by keyword
        self.grouping_feature = grouping_feature
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
        self.testing_dataset_dict = dict() 
        return
   
    @timeit
    def set_up(self):
        SingleFit.set_up(self)
        tidxs = np.arange(0, len(self.testing_datasets)) 
        for tidx in tidxs:
            td_entry = dict()
            td_entry['dataset'] = copy.deepcopy(self.testing_datasets[tidx])
            td_entry['feature_data'] = np.asarray(td_entry['dataset'].get_data(self.feature_plot_feature)).ravel()
            if not (self.grouping_feature is None):
                td_group_data = np.asarray(td_entry['dataset'].get_data(self.grouping_feature)).ravel()
                td_entry['group_indices'] = gttd.get_logo_indices(td_group_data)
            test_data_label = self.data_labels[tidx]
            self.testing_dataset_dict[test_data_label] = dict(td_entry)
        return

    @timeit
    def predict(self):
        self.get_singlefits()
        return

    def get_singlefits(self):
        """Get SingleFit for each testing dataset.
        """
        self.readme_list.append("----- Fitting and prediction -----\n")
        self.readme_list.append("See results per dataset in subfolders\n")
        for testset in self.testing_dataset_dict.keys():
            self.readme_list.append("  %s\n" % testset)
            self.testing_dataset_dict[testset]['SingleFit'] = SingleFit( 
                training_dataset = self.training_dataset, 
                testing_dataset = self.testing_dataset_dict[testset]['dataset'],
                model = self.model, 
                save_path = os.path.join(self.save_path, str(testset)),
                input_features = self.input_features, 
                target_feature = self.target_feature,
                target_error_feature = self.target_error_feature,
                labeling_features = self.labeling_features,
                xlabel=self.xlabel,
                ylabel=self.ylabel,
                stepsize=self.stepsize,
                plot_filter_out = self.plot_filter_out)
            self.testing_dataset_dict[testset]['SingleFit'].run()
        return

    @timeit
    def plot(self):
        self.readme_list.append("----- Plotting -----\n")
        self.readme_list.append("See plots in subfolders:\n")
        self.make_series_feature_plot(group=None)
        if self.grouping_feature is None:
            self.readme_list.append("Grouping feature is not set. No group plots to do.\n")
            return
        allgroups=list()
        for testset in self.testing_dataset_dict.keys():
            for group in self.testing_dataset_dict[testset]['group_indices'].keys():
                allgroups.append(group)
        allgroups = np.unique(allgroups)
        for group in allgroups:
            self.make_series_feature_plot(group=group)
        return

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
            for pfstr in self.plot_filter_out:
                group_notelist.append(pfstr.replace(";"," "))
        testsets = list(self.testing_dataset_dict.keys())
        testsets.sort()
        for testset in testsets:
            ts_dict = self.testing_dataset_dict[testset]
            ts_sf = ts_dict['SingleFit']
            if ts_sf.plot_filter_out is None:
                plotting_index = np.arange(0, len(ts_sf.testing_target_prediction))
            else:
                plotting_index = ts_sf.plotting_index
            if plotting_index is None:
                print("NO PLOTTING INDEX: set %s, group %s" % (testset, group))
            if group is None:
                group_index = np.arange(0, len(ts_sf.testing_target_prediction))
            else:
                if not group in ts_dict['group_indices'].keys():
                    logging.info("No group %s for test set %s. Skipping." % (group, testset))
                    continue
                group_index = ts_dict['group_indices'][group]['test_index']
            display_index = np.intersect1d(group_index, plotting_index)
            feature_data = ts_dict['feature_data'][display_index]
            if ts_sf.testing_target_data is None:
                measured = None
            else:
                measured = ts_sf.testing_target_data[display_index]
            predicted = ts_sf.testing_target_prediction[display_index]
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
        addl_kwargs=dict()
        addl_kwargs['guideline'] = 0
        addl_kwargs['save_path'] = os.path.join(self.save_path, plabel)
        addl_kwargs['xlabel'] = self.feature_plot_xlabel
        addl_kwargs['ylabel'] = self.feature_plot_ylabel
        addl_kwargs['markers'] = self.markers
        addl_kwargs['outlines'] = self.outlines
        addl_kwargs['linestyles'] = self.linestyles
        addl_kwargs['legendloc'] = self.legendloc
        addl_kwargs['sizes'] = self.sizes
        faces = list()
        for fidx in range(0, len(self.markers)):
            faces.append("None")
        addl_kwargs['faces'] = faces
        plotdict.plot_group_splits_with_outliers(group_dict=pdict,
            outlying_groups = series_list,
            label=plabel,
            group_notelist = list(group_notelist),
            addl_kwargs = dict(addl_kwargs))
        return

