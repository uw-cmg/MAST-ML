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
from AnalysisTemplate import AnalysisTemplate
from AnalysisTemplate import timeit
import logging

class FullFit(AnalysisTemplate):
    """Do full fit.
        training_dataset,
        testing_dataset,
        model,
        save_path,
        train_index,
        test_index,
        input_features,
        target_feature,
        labeling_features, see AnalysisTemplate.
        xlabel <str>: Label for full-fit x-axis.
        ylabel <str>: Label for full-fit y-axis
        stepsize <float>: Step size for plot grid
        group_field_name <str>: (optional) field name for grouping data
                                        field may be numeric
        measured_error_field_name <str>: field name for measured y-data error (optional)
        mark_outlying_groups <int>: Number of outlying groups to mark
        Plots results in a Predicted vs. Measured square plot.
    """
    def __init__(self, 
        training_dataset=None,
        testing_dataset=None,
        model=None,
        save_path=None,
        train_index=None,
        test_index=None,
        input_features=None,
        target_feature=None,
        labeling_features=None,
        xlabel="Measured",
        ylabel="Predicted",
        stepsize=1,
        group_field_name = None,
        measured_error_field_name = None,
        mark_outlying_groups = 2,
        *args, **kwargs):
        AnalysisTemplate.__init__(self, 
            training_dataset=training_dataset, 
            testing_dataset=testing_dataset,
            model=model, 
            save_path = save_path,
            train_index = train_index, 
            test_index = test_index,
            input_features = input_features, 
            target_feature = target_feature,
            labeling_features = labeling_features)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.stepsize = float(stepsize)
        self.group_field_name = group_field_name
        self.measured_error_field_name = measured_error_field_name
        self.mark_outlying_groups = int(mark_outlying_groups)
        self.overall_analysis=None
        self.train_group_data = None
        self.test_group_data = None
        self.train_group_indices = None
        self.test_group_indices = None
        self.train_groups=None
        self.test_groups =None
        self.overall_group_dict=dict()
        self.overall_outlying_groups=list()
        self.measured_error_data = None
        self.group_analysis_dict=dict()
        self.split_outlying_groups=list()
        self.split_group_dict=dict()
        return
    
    @timeit
    def do_single_fit(self, label="single_fit", train_index=None, test_index=None):
        """Do a single fit.
            Args:
                label <str>: subfolder label
                train_index <None or list>: training index:
                                list of data index numbers to use,
                                for example, to filter data by groups
                test_index <None or list>: testing index
            Returns:
                AnalysisTemplate object
                Plots and saves information to the 'label' subfolder
        """
        single_save = os.path.join(self.save_path, label)
        single_analysis = AnalysisTemplate(
                training_dataset = self.training_dataset,
                testing_dataset= self.testing_dataset,
                model=self.model, 
                save_path = single_save,
                train_index = train_index, 
                test_index = test_index,
                input_features = self.input_features, 
                target_feature = self.target_feature,
                labeling_features = self.labeling_features)
        if not os.path.isdir(single_save):
            os.mkdir(single_save)
        single_analysis.get_unfiltered_data()
        single_analysis.get_train_test_indices()
        single_analysis.get_data()
        single_analysis.get_model()
        single_analysis.get_trained_model()
        single_analysis.get_prediction()
        single_analysis.get_statistics()
        single_analysis.print_statistics()
        single_analysis.print_output_csv()
        single_analysis.print_readme()
        addl_plot_kwargs = dict()
        addl_plot_kwargs['xlabel'] = self.xlabel
        addl_plot_kwargs['ylabel'] = self.ylabel
        addl_plot_kwargs['stepsize'] = self.stepsize
        addl_plot_kwargs['label'] = label
        addl_plot_kwargs['save_path'] = single_save
        if self.measured_error_field_name is None:
            pass
        else:
            self.measured_error_data = np.asarray(self.testing_dataset.get_data(self.measured_error_field_name)).ravel()
            if test_index is None:
                addl_plot_kwargs['xerr'] = self.measured_error_data
            else:
                addl_plot_kwargs['xerr'] = self.measured_error_data[test_index]
        single_analysis.plot_results(addl_plot_kwargs)
        return single_analysis

    @timeit
    def set_group_info(self):
        self.test_group_data = np.asarray(self.testing_dataset.get_data(self.group_field_name)).ravel()
        self.train_group_data = np.asarray(self.training_dataset.get_data(self.group_field_name)).ravel()
        self.train_group_indices = gttd.get_logo_indices(self.train_group_data)
        self.test_group_indices = gttd.get_logo_indices(self.test_group_data)
        self.train_groups = list(self.train_group_indices.keys())
        self.test_groups = list(self.test_group_indices.keys())
        return

    @timeit
    def get_overall_group_dict(self):
        osg_dict=dict()
        highest_rmses=list()
        num_mark = min(self.mark_outlying_groups, len(self.test_groups))
        for oidx in range(0, num_mark):
            highest_rmses.append((0, "nogroup"))
        for group in self.test_groups:
            g_index = self.test_group_indices[group]["test_index"]
            g_ypredict= self.overall_analysis.testing_target_prediction[g_index]
            if self.overall_analysis.testing_target_data is None:
                continue #no target data; cannot add to plotting
            g_ydata = self.overall_analysis.testing_target_data[g_index]
            #g_mean_error = np.mean(g_ypredict - g_ydata)
            g_rmse = np.sqrt(mean_squared_error(g_ypredict, g_ydata))
            if self.measured_error_field_name is None:
                g_ydata_err = None
            else:
                g_ydata_err = self.measured_error_data[g_index]
            osg_dict[group] = dict()
            osg_dict[group]['xdata'] = g_ydata
            osg_dict[group]['xerrdata'] = g_ydata_err
            osg_dict[group]['ydata'] = g_ypredict
            osg_dict[group]['rmse'] = g_rmse
            min_entry = min(highest_rmses)
            min_rmse = min_entry[0]
            if g_rmse > min_rmse:
                highest_rmses[highest_rmses.index(min_entry)]= (g_rmse, group)
        logging.debug("Highest RMSEs: %s" % highest_rmses)
        self.overall_group_dict=dict(osg_dict)
        for high_rmse in highest_rmses:
            self.overall_outlying_groups.append(high_rmse[1])
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
            train_index = self.train_group_indices[group]['test_index']
            test_index = self.test_group_indices[group]['test_index']
            self.group_analysis_dict[group] = self.do_single_fit(label=group,
                                                train_index=train_index,
                                                test_index=test_index)
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
            g_analysis = self.group_analysis_dict[group] #AnalysisTemplate obj.
            sgdict[group]=dict()
            sgdict[group]['xdata'] = g_analysis.testing_target_data
            if self.measured_error_field_name is None:
                sgdict[group]['xerrdata'] = None
            else:
                sgdict[group]['xerrdata'] = self.measured_error_data[g_analysis.test_index]
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
    def print_readme(self):
        rlist=list()
        rlist.append("----- Folder contents -----\n")
        rlist.append("single_fit\n")
        rlist.append("    Folder containing a single overall fit.\n")
        if self.group_field_name is None:
            rlist.append("Other folders would appear if\n")
            rlist.append("    group_field_name were to be set.\n")
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
    def run(self):
        self.overall_analysis = self.do_single_fit()
        if self.group_field_name is None: #no additional analysis to do
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


