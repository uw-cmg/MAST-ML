import matplotlib
import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import data_analysis.printout_tools as ptools
import plot_data.plot_predicted_vs_measured as plotpm
import plot_data.plot_xy as plotxy
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
        label_field_name = None,
        numeric_field_name = None,
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
            self.measured_error_data = np.asarray(self.training_dataset.get_data(self.measured_error_field_name)).ravel()
            if train_index is None:
                addl_plot_kwargs['xerr'] = self.measured_error_data
            else:
                addl_plot_kwargs['xerr'] = self.measured_error_data[train_index]
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
        xdatalist=list()
        ydatalist=list()
        labellist=list()
        xerrlist=list()
        yerrlist=list()
        otherxdata=list()
        otherxerrdata=list()
        otherydata=list()
        test_groups = list(self.test_groups)
        test_groups.sort()
        for group in test_groups:
            if group in outlying_groups:
                rmse = group_dict[group]['rmse']
                xdatalist.append(group_dict[group]['xdata'])
                xerrlist.append(group_dict[group]['xerrdata'])
                ydatalist.append(group_dict[group]['ydata'])
                yerrlist.append(None)
                labellist.append(group)
                group_notelist.append('{:<1}: {:.2f}'.format(group, rmse))
            else:
                otherxdata.extend(group_dict[group]['xdata'])
                otherxerrdata.extend(group_dict[group]['xerrdata'])
                otherydata.extend(group_dict[group]['ydata'])
        if len(otherxdata) > 0:
            xdatalist.insert(0,otherxdata) #prepend
            xerrlist.insert(0,otherxerrdata)
            ydatalist.insert(0,otherydata)
            yerrlist.insert(0,None)
            labellist.insert(0,"All others")
            all_other_rmse = np.sqrt(mean_squared_error(otherydata, otherxdata))
            group_notelist.append('{:<1}: {:.2f}'.format("All others", all_other_rmse))
        kwargs=dict()
        kwargs['xlabel'] = self.xlabel
        kwargs['ylabel'] = self.ylabel
        kwargs['save_path'] = os.path.join(self.save_path, label)
        kwargs['xdatalist'] = xdatalist
        kwargs['ydatalist'] = ydatalist
        kwargs['stepsize'] = self.stepsize
        kwargs['xerrlist'] = xerrlist
        kwargs['yerrlist'] = yerrlist
        kwargs['labellist'] = labellist
        kwargs['notelist'] = group_notelist
        kwargs['plotlabel'] = label
        kwargs['guideline'] = 1
        plotxy.multiple_overlay(**kwargs) 
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
        self.plot_group_splits_with_outliers(group_dict=self.overall_group_dict,
            outlying_groups = self.overall_outlying_groups,
            label="overall_overlay", 
            group_notelist=["RMSEs for overall fit:",
                "Overall: %3.2f" % self.overall_analysis.statistics['rmse']])
        self.get_group_analysis_dict()
        self.get_split_group_dict()
        self.plot_group_splits_with_outliers(group_dict=self.split_group_dict,
            outlying_groups = self.split_outlying_groups,
            label="split_overlay",
            group_notelist=["RMSEs for per-group fits:"])
        self.print_readme()
        return


def do_single_fit(model, 
            trainx=None,
            trainy=None, 
            testx=None, 
            testy=None,
            testyerr=None,
            xlabel="", ylabel="", 
            stepsize=1,
            guideline=1,
            savepath="",
            plotlabel="single_fit",
            **kwargs):
    """Single fit with test/train data and plotting.
    """
    if testx is None:
        testx = np.copy(trainx)
    if testy is None:
        testy = np.copy(trainy)
    model.fit(trainx, trainy)
    ypredict = model.predict(testx)
    rmse = np.sqrt(mean_squared_error(ypredict, testy))
    y_abs_err = np.absolute(ypredict - testy)
    mean_error = np.mean(ypredict - testy) #mean error sees if fit is shifted in one direction or another; so, do not want absolute here

    notelist=list()
    notelist.append("RMSE: %3.6f" % rmse)
    notelist.append("Mean error: %3.6f" % mean_error)
    print("RMSE: %3.6f" % rmse)
    print("Mean error: %3.6f" % mean_error)

    kwargs=dict()
    kwargs['xerr'] = testyerr #measured error
    kwargs['xlabel'] = xlabel
    kwargs['ylabel'] = ylabel
    kwargs['stepsize'] = stepsize
    kwargs['notelist'] = notelist
    kwargs['guideline'] = guideline
    kwargs['savepath'] = savepath
    kwargs['plotlabel'] = plotlabel
    plotxy.single(testy, ypredict, **kwargs)
    return [ypredict, y_abs_err, rmse, mean_error]


def execute(model, data, savepath, 
        xlabel="Measured",
        ylabel="Predicted",
        stepsize=1,
        group_field_name=None,
        label_field_name=None,
        numeric_field_name=None,
        measured_error_field_name=None,
        *args, **kwargs):
    """Do full fit.
        Main function is split off from execute in order to allow external use.
        Results in a predicted-vs-measured square plot
        model, data, savepath = see AllTests.py
        xlabel <str>: Label for full-fit x-axis.
        ylabel <str>: Label for full-fit y-axis
        stepsize <float>: Step size for plot grid
        group_field_name <str>: (optional) field name for grouping data
                                        field may be numeric
        label_field_name <str>: (optional) field name for labeling the groups
                                    from group_field_name
        numeric_field_name <str>: (optional) field name for numeric data 
                                    field that may help identify individual
                                    points or outliers
        measured_error_field_name <str>: field name for measured y-data error (optional)
    """
    stepsize=float(stepsize)
    if numeric_field_name == None: #help identify each point
        numeric_field_name = data.x_features[0]
    Xdata = np.asarray(data.get_x_data())
    ydata = np.asarray(data.get_y_data()).ravel()
    #
    kwargs_f = dict()
    kwargs_f['xlabel'] = xlabel
    kwargs_f['ylabel'] = ylabel
    kwargs_f['stepsize'] = stepsize
    kwargs_f['numeric_field_name'] = numeric_field_name
    kwargs_f['group_field_name'] = group_field_name
    kwargs_f['label_field_name'] = label_field_name
    kwargs_f['savepath'] = savepath
    kwargs_f['do_pergroup_fits'] = 1
    kwargs_f['test_numericdata'] = np.asarray(data.get_data(numeric_field_name)).ravel()
    if not(group_field_name == None):
        kwargs_f['test_groupdata'] = np.asarray(data.get_data(group_field_name)).ravel()
        if label_field_name == None:
            label_field_name = group_field_name
        kwargs_f['test_labeldata'] = np.asarray(data.get_data(label_field_name)).ravel()
    if not(measured_error_field_name == None):
        kwargs_f['test_yerrdata'] = np.asarray(data.get_data(measured_error_field_name)).ravel()
    kwargs_f['full_xtrain'] = Xdata
    kwargs_f['full_ytrain'] = ydata
    myarray = do_full_fit(model, **kwargs_f)
    return myarray

def do_full_fit(model,
        full_xtrain = None,
        full_ytrain = None,
        full_xtest = None,
        full_ytest = None,
        savepath="",
        xlabel="",
        ylabel="",
        stepsize=1,
        do_pergroup_fits = 1,
        test_groupdata=None,
        train_groupdata=None,
        test_numericdata=None,
        test_labeldata=None,
        group_field_name=None,
        numeric_field_name = None,
        label_field_name=None,
        test_yerrdata=None,
        *args,**kwargs):
    """Full fit.
        full_xtrain <numpy array>: X training data
        full_ytrain <numpy array>: Y training data
        full_xtest <numpy array>: X testing data
        full_ytest <numpy array>: Y testing data
            If full_ytest is not given, full_ytest is the same as full_ytrain.
        savepath <str>: save path
        xlabel <str>: x label for predicted-vs-measured plot
        ylabel <str>: y label for predicted-vs-measured plot
        stepsize <float>: step size for plot grid
        do_pergroup_fits <int>: 1 - also fit separately on each group, grouped
                                    by group_field_name
                                0 - do not do pergroup fits
        test_groupdata <numpy array>: group_field_name data corresponding to full_ytest
        train_groupdata <numpy array>: group_field_name data corresponding to full_ytrain
        test_numericdata <numpy array>: numeric_field_name data corresponding to full_ytest
        test_labeldata <numpy array>: label_field_name data corresponding to full_ytest
        test_yerrdata <numpy array>: measured_error_field_name data corresponding to full_ytest
        group_field_name <str>: field name of grouping field (see execute)
        numeric_field_name <str>: field name of numeric field (see execute)
        label_field_name <str>: field name of labeling field (see execute)
        measured_error_field_name <str>: field name for measured y-data error (optional)
    """
    stepsize = float(stepsize)
    if full_xtest is None:
        full_xtest = np.copy(full_xtrain)
    if full_ytest is None:
        full_ytest = np.copy(full_ytrain)
    if train_groupdata is None:
        if test_groupdata is None:
            pass
        else:
            train_groupdata = np.copy(test_groupdata)
    kwargs=dict()
    kwargs['xlabel'] = xlabel
    kwargs['ylabel'] = ylabel
    kwargs['stepsize'] = stepsize
    kwargs['savepath'] = savepath
    kwargs['guideline'] = 1
    kwargs['trainx'] = full_xtrain
    kwargs['trainy'] = full_ytrain
    kwargs['testx'] = full_xtest
    kwargs['testy'] = full_ytest
    if not (test_yerrdata is None):
        kwargs['testyerr'] = test_yerrdata
    [ypredict, y_abs_err, rmse, mean_error] = do_single_fit(model, **kwargs)

    if numeric_field_name == None: #help identify each point
        raise ValueError("Numeric field name not set.")

    if group_field_name == None:
        headerline = "%s,Measured,Predicted,Absolute error" % numeric_field_name
        myarray = np.array([test_numericdata, full_ytest, ypredict, y_abs_err]).transpose()
    else:
        headerline = "%s,%s,%s,Measured,Predicted,Absolute error" % (numeric_field_name, group_field_name, label_field_name)
        myarray = np.array([test_numericdata, test_groupdata, test_labeldata,full_ytest, ypredict, y_abs_err]).transpose()
    
    csvname = os.path.join(savepath, "FullFit_data.csv")
    ptools.mixed_array_to_csv(csvname, headerline, myarray)
    
    if group_field_name == None:
        return myarray #numeric label data,  measured, predicted, abs error
    train_group_indices = gttd.get_logo_indices(train_groupdata)
    test_group_indices = gttd.get_logo_indices(test_groupdata)
    
    test_groups = list(test_group_indices.keys())
    test_groups.sort()

    train_groups = list(train_group_indices.keys())
    train_groups.sort()
   
    #EXTRACT EACH GROUP FITS FROM OVERALL FIT, and OVERLAY
    xdatalist=list()
    ydatalist=list()
    labellist=list()
    xerrlist=list()
    yerrlist=list()
    group_notelist=list()
    group_notelist.append("RMSE from overall fitting:")
    rmselist=list()
    for group in test_groups:
        g_index = test_group_indices[group]["test_index"]
        g_label = test_labeldata[g_index[0]]
        g_ypredict = ypredict[g_index]
        g_ydata = full_ytest[g_index]
        g_mean_error = np.mean(g_ypredict - g_ydata)
        g_rmse = np.sqrt(mean_squared_error(g_ypredict, g_ydata))
        xdatalist.append(g_ydata) #actual
        ydatalist.append(g_ypredict) #predicted
        labellist.append(g_label)
        if test_yerrdata is None:
            xerrlist.append(None)
        else: 
            xerrlist.append(test_yerrdata[g_index])
        yerrlist.append(None)
        rmselist.append(g_rmse)
        group_notelist.append('{:<1}: {:.2f}'.format(g_label, g_rmse))
    #allow largest RMSE to be plotted if too many lines
    numlines = len(xdatalist)
    if numlines > 6: #too many lines for multiple_overlay
        import heapq
        showmax = 3
        maxrmslist = heapq.nlargest(showmax, rmselist)
        maxidxlist = heapq.nlargest(showmax, range(len(rmselist)), 
                        key=lambda x: rmselist[x])
        temp_xdatalist=list()
        temp_ydatalist=list()
        temp_xerrlist=list()
        temp_yerrlist=list()
        temp_group_notelist=list()
        temp_group_notelist.append("RMSE from overall fitting:")
        temp_labellist=list()
        bigxdata = np.empty((0,1))
        bigydata = np.empty((0,1))
        biglabel = "All others" # Use "_" for no label
        bigxerr = np.empty((0,1))
        bigyerr = np.empty((0,1))
        for nidx in range(0, numlines):
            if not (nidx in maxidxlist):
                bigxdata = np.append(bigxdata, xdatalist[nidx])
                bigydata = np.append(bigydata, ydatalist[nidx])
                bxerr = xerrlist[nidx]
                if bxerr is None:
                    bxerr = np.zeros(len(xdatalist[nidx]))
                bigxerr = np.append(bigxerr, bxerr)
                byerr = yerrlist[nidx]
                if byerr is None:
                    byerr = np.zeros(len(ydatalist[nidx]))
                bigyerr = np.append(bigyerr, byerr)
        temp_xdatalist.append(bigxdata)
        temp_ydatalist.append(bigydata)
        temp_labellist.append(biglabel)
        temp_xerrlist.append(bigxerr)
        temp_yerrlist.append(bigyerr)
        temp_group_notelist.append("Overall: %3.2f" % rmse) #overall rmse
        for nidx in range(0, numlines): #loop repeats because want big array first
            if (nidx in maxidxlist):
                temp_xdatalist.append(xdatalist[nidx])
                temp_ydatalist.append(ydatalist[nidx])
                temp_labellist.append(labellist[nidx])
                temp_xerrlist.append(xerrlist[nidx])
                temp_yerrlist.append(yerrlist[nidx])
                temp_group_notelist.append(group_notelist[nidx+1])
        xdatalist = temp_xdatalist
        ydatalist = temp_ydatalist
        labellist = temp_labellist
        xerrlist = temp_xerrlist
        yerrlist = temp_yerrlist
        group_notelist = temp_group_notelist
    kwargs['xdatalist'] = xdatalist
    kwargs['ydatalist'] = ydatalist
    kwargs['stepsize'] = stepsize
    kwargs['xerrlist'] = xerrlist
    kwargs['yerrlist'] = yerrlist
    kwargs['labellist'] = labellist
    kwargs['notelist'] = group_notelist
    kwargs['plotlabel'] = "OverallFit_overlay"
    plotxy.multiple_overlay(**kwargs) 
    
    #GET PER-GROUP FITS, and overlay them
    if not(do_pergroup_fits == 1):
        return myarray
   
    xdatalist=list()
    ydatalist=list()
    labellist=list()
    xerrlist=list()
    yerrlist=list()
    group_notelist=list()
    group_notelist.append("RMSE for per-group fitting:")
    for test_group in test_groups:
        if not test_group in train_groups: #cannot be trained
            continue
        g_train_index = train_group_indices[test_group]["test_index"]
        g_test_index = test_group_indices[test_group]["test_index"]
        g_label = test_labeldata[g_test_index[0]]
        kwargs['plotlabel'] = "GroupFit_%s_%s" % (test_group, g_label)
        kwargs['trainx'] = full_xtrain[g_train_index]
        kwargs['trainy'] = full_ytrain[g_train_index]
        kwargs['testx'] = full_xtest[g_test_index]
        if not (test_yerrdata is None):
            kwargs['testyerr'] = test_yerrdata[g_test_index]
        g_ytest = full_ytest[g_test_index]
        kwargs['testy'] = g_ytest
        [g_ypredict, g_y_abs_err, g_rmse, g_mean_error] = do_single_fit(model, **kwargs)
        g_myarray = np.array([test_labeldata[g_test_index], 
                            test_groupdata[g_test_index], 
                            g_ytest, g_ypredict, g_y_abs_err]).transpose()
        csvname = os.path.join(savepath, "GroupFit_data_%s_%s.csv" % (test_group, g_label))
        ptools.mixed_array_to_csv(csvname, headerline, g_myarray)
        xdatalist.append(g_ytest) #actual
        ydatalist.append(g_ypredict) #predicted
        labellist.append(g_label)
        if test_yerrdata is None:
            xerrlist.append(None)
        else:
            xerrlist.append(test_yerrdata[g_test_index])
        yerrlist.append(None)
        group_notelist.append('{:<1}: {:.2f}'.format(g_label, g_rmse))
    kwargs['xdatalist'] = xdatalist
    kwargs['ydatalist'] = ydatalist
    kwargs['stepsize'] = stepsize
    kwargs['xerrlist'] = xerrlist
    kwargs['yerrlist'] = yerrlist
    kwargs['labellist'] = labellist
    kwargs['notelist'] = group_notelist
    kwargs['plotlabel'] = "GroupFit_overlay"
    plotxy.multiple_overlay(**kwargs) 
    return myarray #for total fit: numeric data, group data, measured, predicted, error
