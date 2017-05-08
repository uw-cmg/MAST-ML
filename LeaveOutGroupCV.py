import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from mean_error import mean_error
import csv
import matplotlib
import portion_data.get_test_train_data as gttd
import data_analysis.printout_tools as ptools
import plot_data.plot_rmse as plotrmse
import os
from SingleFit import SingleFit
from LeaveOutPercentCV import LeaveOutPercentCV
from SingleFit import timeit
from sklearn.metrics import r2_score

class LeaveOutGroupCV(LeaveOutPercentCV):
    """leave-out-group cross validation
   
    Args:
        training_dataset, (Should be the same as testing_dataset)
        testing_dataset, (Should be the same as training_dataset)
        model,
        save_path,
        input_features,
        target_feature,
        target_error_feature,
        labeling_features, 
        xlabel, 
        ylabel,
        stepsize,
        mark_outlying_points (Use only 1 number), see parent class.
        grouping_feature <str>: Feature name for grouping
 
    Returns:
        Analysis in the save_path folder
        Plots results in a predicted vs. measured square plot.
    Raises:
        ValueError if grouping feature is None
        ValueError if testing target data is None; CV must have
                testing target data
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
        mark_outlying_points=None,
        grouping_feature=None,
        *args, **kwargs):
        """
        Additional class attributes to parent class:
            Set by keyword:
            self.grouping_feature <str>: Grouping feature
            Set in code:
            self.group_data <numpy array>: Grouping data
            self.group_indices <dict>: Grouping indices
            self.groups <list>: List of groups
        """
        if not(training_dataset == testing_dataset):
            raise ValueError("Only testing_dataset will be used. Use the same values for training_dataset and testing_dataset")
        LeaveOutPercentCV.__init__(self, 
            training_dataset=training_dataset, #only testing_dataset is used
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
            mark_outlying_points = mark_outlying_points,
            percent_leave_out = -1, #not using this field
            num_cvtests = -1, #not using this field; num_cvtests is set by number of groups
            fix_random_for_testing = 0, #no randomization in this test
            )
        if grouping_feature is None:
            raise ValueError("grouping feature must be set.")
        self.grouping_feature = grouping_feature
        self.group_data = None
        return 

    def predict(self):
        #Predictions themselves are covered in self.fit()
        self.get_statistics()
        self.print_statistics()
        self.readme_list.append("----- Output data -----\n")
        for cvtest in self.cvtest_dict.keys():
            self.print_output_csv(label=self.cvtest_dict[cvtest]['group'], cvtest_entry=self.cvtest_dict[cvtest])
        return

    @timeit
    def plot(self):
        self.readme_list.append("----- Plotting -----\n")
        notelist=list()
        notelist.append("Mean RMSE: {:.2f}".format(self.statistics['avg_rmse']))
        kwargs=dict()
        kwargs['xlabel'] = self.xlabel
        kwargs['ylabel'] = self.ylabel
        kwargs['notelist'] = notelist
        kwargs['savepath'] = self.save_path
        kwargs['marklargest'] = self.mark_outlying_points
        group_label_list=list()
        rms_list=list()
        group_rms_list=list()
        for cvtest in self.cvtest_dict.keys():
            group_rms_list.append((self.cvtest_dict[cvtest]['group'],
                                    self.cvtest_dict[cvtest]['rmse']))
        group_rms_list.sort() #sorts by group
        group_rms_array = np.array(group_rms_list)
        kwargs['rms_list'] = np.array(group_rms_array[:,1],'float')
        kwargs['group_list'] = group_rms_array[:,0]
        plotrmse.vs_leftoutgroup(**kwargs)
        return

    def set_up_cv(self):
        if self.testing_target_data is None:
            raise ValueError("Testing target data cannot be none for cross validation.")
        indices = np.arange(0, len(self.testing_target_data))
        self.readme_list.append("----- CV setup -----\n")
        self.readme_list.append("%i CV tests,\n" % self.num_cvtests)
        self.readme_list.append("leaving out %s groups.\n" % self.grouping_feature)
        self.group_data = np.asarray(self.testing_dataset.get_data(self.grouping_feature)).ravel()
        self.group_indices = gttd.get_logo_indices(self.group_data)
        self.groups = list(self.group_indices.keys())
        self.num_cvtests = len(self.groups)
        self.cvmodel = self.model
        for cvtest in range(0, self.num_cvtests):
            group = self.groups[cvtest]
            self.cvtest_dict[cvtest] = dict()
            self.cvtest_dict[cvtest]['train_index'] = self.group_indices[group]['train_index']
            self.cvtest_dict[cvtest]['test_index'] = self.group_indices[group]['test_index']
            self.cvtest_dict[cvtest]['group'] = group #keep track of group
        return

    def print_statistics(self):
        LeaveOutPercentCV.print_statistics(self)
        self.readme_list.append("Statistics by group:\n")
        statlist=list()
        for cvtest in range(0, self.num_cvtests):
            for key in ['rmse','mean_error']:
                statlist.append("  %s: %s: %3.3f\n" % (self.cvtest_dict[cvtest]['group'], key, self.cvtest_dict[cvtest][key]))
        statlist.sort() #sorts by group
        for stat in statlist:
            self.readme_list.append(stat)
        return

    def print_output_csv(self, label="", cvtest_entry=None):
        """
            Modify once dataframe is in place
        """
        olabel = "%s_test_data.csv" % label
        ocsvname = os.path.join(self.save_path, olabel)
        self.readme_list.append("%s file created with columns:\n" % olabel)
        headerline = ""
        printarray = None
        if len(self.labeling_features) > 0:
            self.readme_list.append("   labeling features: %s\n" % self.labeling_features)
            print_features = list(self.labeling_features)
        else:
            print_features = list()
        print_features.extend(self.input_features)
        self.readme_list.append("   input features: %s\n" % self.input_features)
        if not (self.testing_target_data is None):
            print_features.append(self.target_feature)
            self.readme_list.append("   target feature: %s\n" % self.target_feature)
            if not (self.target_error_feature is None):
                print_features.append(self.target_error_feature)
                self.readme_list.append("   target error feature: %s\n" % self.target_error_feature)
        for feature_name in print_features:
            headerline = headerline + feature_name + ","
            feature_vector = np.asarray(self.testing_dataset.get_data(feature_name)).ravel()
            if printarray is None:
                printarray = feature_vector
            else:
                printarray = np.vstack((printarray, feature_vector))
        headerline = headerline + "Prediction,"
        self.readme_list.append("   prediction: Prediction\n")
        printarray = np.vstack((printarray, cvtest_entry['prediction_array']))
        printarray=printarray.transpose()
        ptools.mixed_array_to_csv(ocsvname, headerline, printarray)
        return

    def plot_best_worst_overlay(self, notelist=list()):
        kwargs2 = dict()
        kwargs2['xlabel'] = self.xlabel
        kwargs2['ylabel'] = self.ylabel
        kwargs2['labellist'] = ["Best test","Worst test"]
        kwargs2['xdatalist'] = list([self.testing_target_data, 
                            self.testing_target_data])
        kwargs2['ydatalist'] = list(
                [self.cvtest_dict[self.best_test_index]['prediction_array'],
                self.cvtest_dict[self.worst_test_index]['prediction_array']])
        kwargs2['xerrlist'] = list([None,None])
        kwargs2['yerrlist'] = list([None,None])
        kwargs2['notelist'] = list(notelist)
        kwargs2['guideline'] = 1
        kwargs2['plotlabel'] = "best_worst_overlay"
        kwargs2['save_path'] = self.save_path
        kwargs2['stepsize'] = self.stepsize
        if not (self.mark_outlying_points is None):
            kwargs2['marklargest'] = self.mark_outlying_points
            if (self.labeling_features is None) or (len(self.labeling_features) == 0):
                raise ValueError("Must specify some labeling features if you want to mark the largest outlying points")
            labels = np.asarray(self.testing_dataset.get_data(self.labeling_features[0])).ravel()
            kwargs2['mlabellist'] = list([labels,labels])
        plotxy.multiple_overlay(**kwargs2)
        self.readme_list.append("Plot best_worst_overlay.png created,\n")
        self.readme_list.append("    showing the best and worst of %i tests.\n" % self.num_cvtests)
        return


def execute(model, data, savepath, 
            group_field_name=None,
            label_field_name=None,
            xlabel="Group number",
            ylabel="RMSE",
            title="",
            *args, **kwargs):
    """
        model, data, savepath = see AllTests.py
        group_field_name <str>: field name of field over which to group
                                    for the successive left-out groups
        label_field_name <str>: field name of a label field which corresponds
                                    to the grouping field
        xlabel <str>: x-axis label for the rmse-vs- left-out group plot
        ylabel <str>: y-axis label
        title <str>: plot title (optional)
    """
    if group_field_name == None:
        raise ValueError("No group field name set.")

    rms_list = list()
    group_value_list = list()
    group_label_list=list()
    me_list = list()
    group_list = list()

    Xdata = np.asarray(data.get_x_data())
    ydata = np.asarray(data.get_y_data()).ravel()
    groupdata = np.asarray(data.get_data(group_field_name)).ravel()
    if label_field_name == None:
        label_field_name = group_field_name
    
    labeldata = np.asarray(data.get_data(label_field_name)).ravel()

    indices = gttd.get_field_logo_indices(data, group_field_name)
    
    groups = list(indices.keys())
    groups.sort()
    for group in groups:
        train_index = indices[group]["train_index"]
        test_index = indices[group]["test_index"]
        test_group_val = groupdata[test_index[0]] #left-out group value
        test_group_label = labeldata[test_index[0]]
        model.fit(Xdata[train_index], ydata[train_index])
        Ypredict = model.predict(Xdata[test_index])

        rms = np.sqrt(mean_squared_error(Ypredict, ydata[test_index]))
        me = mean_error(Ypredict, ydata[test_index])
        rms_list.append(rms)
        group_value_list.append(test_group_val)
        group_label_list.append(test_group_label)
        me_list.append(me)
        group_list.append(group) #should be able to just use groups

    print('Mean RMSE: ', np.mean(rms_list))
    print('Mean Mean Error: ', np.mean(me_list))

    
    kwargs=dict()
    kwargs['xlabel'] = xlabel
    kwargs['ylabel'] = ylabel
    kwargs['title'] = title
    kwargs['notelist'] = notelist
    kwargs['savepath'] = savepath
    if label_field_name == None:
        kwargs['group_label_list'] = None
    else:
        kwargs['group_label_list'] = group_label_list

    plotrmse.vs_leftoutgroup(group_value_list, rms_list, **kwargs)

    csvname = os.path.join(savepath, "leavegroupout.csv")
    headerline = "Group,%s,%s,RMSE,Mean error" % (group_field_name, label_field_name)
    save_array = np.asarray([group_list, group_value_list, group_label_list, rms_list, me_list]).transpose()
    ptools.mixed_array_to_csv(csvname, headerline, save_array) 
    return
