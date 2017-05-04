import matplotlib
import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import data_analysis.printout_tools as ptools
import plot_data.plot_predicted_vs_measured as plotpm
import plot_data.plot_xy as plotxy
import portion_data.get_test_train_data as gttd
import os
import time
import logging
import copy
logger = logging.getLogger()

def timeit(method):
    """Timing function for logger.
        Taken from http://stackoverflow.com/questions/38314993/standard-logging-for-every-method-in-a-class.
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        delta = te - ts

        hours, remainder = divmod(delta, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info('%s.%s took %02d:%02d:%02.6f',
                     method.__module__,
                     method.__name__,
                     int(hours),
                     int(minutes),
                     seconds)

        return result

    return timed

class SingleFit():
    """This is the basic analysis class.

    This class provides a template for other classes. Combine many classes 
    to do meta-analysis.

    Args:
        training_dataset <Data object>: Training dataset
        testing_dataset <Data object>: Testing dataset
        model <sklearn model>: Machine-learning model
        save_path <str>: Save path
        input_features <list of str>: List of input feature names
        target_feature <str>: Target feature name
        target_error_feature <str>: Name of the feature describing error
                                    in the target feature (optional)
        labeling_features <list of str>: List of labeling feature names
                                         that can help identify specific
                                         points
        xlabel <str>: Label for full-fit x-axis (default "Measured")
        ylabel <str>: Label for full-fit y-axis (default "Predicted")
        stepsize <float>: Step size for plot grid (default None)
        plot_filter_out <list>: List of semicolon-delimited strings with
                            feature;operator;value for filtering out
                            values for plotting.
    Returns:
        Analysis in the path marked by save_path

    Raises:
        ValueError if the following are not set:
                    training_dataset, testing_dataset,
                    input_features, target_feature,
                    model
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
        stepsize=None,
        plot_filter_out=None,
        *args, **kwargs):
        """Initialize class.
            Attributes that can be set through keywords:
                self.training_dataset
                self.testing_dataset
                self.model
                self.input_features
                self.target_feature
                self.target_error_feature
                self.labeling_features
                self.save_path
                self.xlabel
                self.ylabel
                self.stepsize
                self.plot_filter_out
            Other attributes:
                self.analysis_name <str>
                self.training_input_data <numpy array>: training input feature data
                self.training_target_data <numpy array>: training target feature data
                self.testing_input_data <numpy array>: testing input feature data
                self.testing_target_data <numpy array>: testing target data
                self.testing_target_data_error <numpy array or None>: testing target data error
                self.trained_model <sklearn model>: trained model
                self.testing_target_prediction <numpy array>: testing target predicted data
                self.statistics <dict of float>: statistics dictionary
                self.readme_list <dict of str>: stores lines for readme
                self.plotting_index <list of int>: index of testing target data
                            and testing target prediction to plot (after
                            self.plot_filter_out has been applied)
        """
        # Keyword-set attributes
        # training csv
        if training_dataset is None:
            raise ValueError("training_dataset is not set")
        if type(training_dataset) is list: #allow inheriting classes to get multiple datasets; MASTML will pass list of data objects
            self.training_dataset= copy.deepcopy(training_dataset[0]) #first item
        elif type(training_dataset) is str:
            self.training_dataset = data_parser.parse(training_dataset)
        else:
            self.training_dataset = copy.deepcopy(training_dataset)
        # testing csv
        if testing_dataset is None:
            raise ValueError("testing_dataset is not set")
        if type(testing_dataset) is list:
            self.testing_dataset = copy.deepcopy(testing_dataset[0])
        elif type(testing_dataset) is str:
            self.testing_dataset = data_parser.parse(testing_dataset)
        else:
            self.testing_dataset=copy.deepcopy(testing_dataset)
        # model
        if model is None:
            raise ValueError("No model.")
        self.model=model
        # features
        if input_features is None:
            raise ValueError("input_features is not set")
        self.input_features=input_features
        if target_feature is None:
            raise ValueError("target_feature is not set")
        self.target_feature=target_feature
        self.target_error_feature = target_error_feature
        if labeling_features is None:
            self.labeling_features = list()
        else:
            self.labeling_features = labeling_features
        # paths
        if save_path is None:
            save_path = os.getcwd()
        save_path = os.path.abspath(save_path)
        self.save_path = save_path
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        # plot customization
        self.xlabel = xlabel
        self.ylabel = ylabel
        if stepsize is None:
            self.stepsize = stepsize
        else:
            self.stepsize = float(stepsize)
        if type(plot_filter_out) is list:
            self.plot_filter_out = plot_filter_out
        elif type(plot_filter_out) is str:
            self.plot_filter_out = plot_filter_out.split(",")
        else:
            self.plot_filter_out = plot_filter_out
        # Self-set attributes
        self.analysis_name = os.path.basename(self.save_path)
        self.training_input_data=None
        self.training_target_data=None
        self.testing_target_data_error=None
        self.testing_input_data=None
        self.testing_target_data=None
        self.trained_model=None
        self.testing_target_prediction=None
        self.statistics=dict()
        self.readme_list=list()
        self.plotting_index=None
        #
        logger.info("-------- %s --------" % self.analysis_name)
        logger.info("Starting analysis at %s" % time.asctime())
        return
   
    @timeit
    def run(self):
        self.set_up()
        self.fit()
        self.predict()
        self.plot()
        self.print_readme()
        return

    @timeit
    def set_up(self):
        self.readme_list.append("%s\n" % time.asctime())
        self.set_data()
        return

    @timeit
    def fit(self):
        self.get_trained_model()
        self.print_model()
        return
   
    @timeit
    def predict(self):
        self.get_prediction()
        self.print_output_csv()
        self.get_statistics()
        self.print_statistics()
        return

    @timeit
    def plot(self):
        self.get_plotting_index()
        self.plot_results()
        return

    def set_data(self):
        """Replace with pandas dataframes
        """
        self.training_dataset.set_y_feature(self.target_feature)
        self.training_dataset.set_x_features(self.input_features)
        self.training_input_data = np.asarray(self.training_dataset.get_x_data())
        self.training_target_data = np.asarray(self.training_dataset.get_y_data()).ravel()
        hasy = self.testing_dataset.set_y_feature(self.target_feature)
        if hasy:
            self.testing_target_data = np.asarray(self.testing_dataset.get_y_data()).ravel()
            if not(self.target_error_feature is None):
                self.testing_target_data_error = np.asarray(self.testing_dataset.get_data(self.target_error_feature)).ravel()
        else:
            self.testing_dataset.set_y_feature(self.input_features[0]) #dummy y feature
            #self.testing_target_data remains None
        self.testing_dataset.set_x_features(self.input_features)
        self.testing_input_data = np.asarray(self.testing_dataset.get_x_data())
        return


    def get_trained_model(self):
        trained_model = self.model.fit(self.training_input_data, self.training_target_data)
        self.trained_model = trained_model
        return
    
    def print_model(self):
        self.readme_list.append("----- Model parameters -----\n")
        for param, paramval in self.trained_model.get_params().items():
            self.readme_list.append("%s: %s\n" % (param, paramval))
        return

    def get_prediction(self):
        self.testing_target_prediction = self.trained_model.predict(self.testing_input_data)
        self.testing_dataset.add_feature("Prediction",self.testing_target_prediction)
        return

    def get_rmse(self):
        rmse = np.sqrt(mean_squared_error(self.testing_target_data, self.testing_target_prediction))        
        return rmse
    
    def get_mean_error(self):
        pred_minus_true =self.testing_target_prediction-self.testing_target_data
        mean_error = np.mean(pred_minus_true)
        return mean_error

    def get_mean_absolute_error(self):
        mean_abs_err = mean_absolute_error(self.testing_target_data, self.testing_target_prediction)
        return mean_abs_err

    def get_rsquared(self):
        rsquared = r2_score(self.testing_target_data, self.testing_target_prediction)
        return rsquared
    
    def get_statistics(self):
        if self.testing_target_data is None:
            logger.warning("No testing target data. Statistics will not be collected.")
            return
        self.statistics['rmse'] = self.get_rmse()
        self.statistics['mean_error'] = self.get_mean_error()
        self.statistics['mean_absolute_error'] = self.get_mean_absolute_error()
        self.statistics['rsquared'] = self.get_rsquared()
        return

    def print_statistics(self):
        self.readme_list.append("----- Statistics -----\n")
        if len(self.statistics) == 0:
            self.readme_list.append("No target data.\n")
            self.readme_list.append("No statistics comparing target data and prediction were collected.\n")
        else:
            for skey, svalue in self.statistics.items():
                self.readme_list.append("%s:%3.4f\n" % (skey, svalue))
        return

    def print_output_csv(self):
        """
            Modify once dataframe is in place
        """
        self.readme_list.append("----- Output data -----\n")
        ocsvname = os.path.join(self.save_path, "output_data.csv")
        self.readme_list.append("output_data.csv file created with columns:\n")
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
        headerline = headerline + "Prediction"
        self.readme_list.append("   prediction: Prediction\n")
        printarray = np.vstack((printarray, self.testing_target_prediction))
        printarray=printarray.transpose()
        ptools.mixed_array_to_csv(ocsvname, headerline, printarray)
        return


    def get_plotting_index(self):
        """Get plotting index of points to plot
        Update with dataframe
        Inefficient.
        """
        if self.plot_filter_out is None:
            return
        out_index = list()
        for pfstr in self.plot_filter_out:
            pflist = pfstr.split(";")
            feature = pflist[0].strip()
            symbol = pflist[1].strip()
            rawvalue = pflist[2].strip()
            try:
                value = float(rawvalue)
            except ValueError:
                value = rawvalue
            featuredata = np.asarray(self.testing_dataset.get_data(feature)).ravel()
            for fidx in range(0, len(featuredata)):
                fdata = featuredata[fidx]
                if fdata is None:
                    continue
                if symbol == "<":
                    if fdata < value:
                        out_index.append(fidx)
                elif symbol == ">":
                    if fdata > value:
                        out_index.append(fidx)
                elif symbol == "=":
                    if fdata == value:
                        out_index.append(fidx)
        all_index = np.arange(0, len(self.testing_target_prediction))
        out_index = np.unique(out_index)
        plotting_index = np.setdiff1d(all_index, out_index) 
        self.plotting_index = list(plotting_index)
        return
    
    def plot_results(self, addl_plot_kwargs=None):
        self.readme_list.append("----- Plotting -----\n")
        if self.testing_target_data is None:
            logger.warning("No testing target data. Predicted vs. measured plot will not be plotted.")
            self.readme_list.append("No target data.\n")
            self.readme_list.append("No plot comparing predicted vs. measured data was made.\n")
            return
        plot_kwargs=dict()
        plot_kwargs['xlabel'] = self.xlabel
        plot_kwargs['ylabel'] = self.ylabel
        plot_kwargs['plotlabel'] = "single_fit"
        plot_kwargs['stepsize'] = self.stepsize
        plot_kwargs['guideline'] = 1
        notelist=list()
        notelist.append("RMSE: %3.3f" % self.statistics['rmse'])
        notelist.append("R-squared: %3.3f" % self.statistics['rsquared'])
        plot_kwargs['notelist'] = notelist
        plot_kwargs['save_path'] = self.save_path
        plot_kwargs['xerr'] = self.testing_target_data_error
        if not (addl_plot_kwargs is None):
            for addl_plot_kwarg in addl_plot_kwargs:
                plot_kwargs[addl_plot_kwarg] = addl_plot_kwargs[addl_plot_kwarg]
        if self.plot_filter_out is None:
            plotxy.single(self.testing_target_data,
                    self.testing_target_prediction,
                    **plot_kwargs)
        else:
            notelist.append("Data not shown:")
            for pfstr in self.plot_filter_out:
                notelist.append("  %s" % pfstr.replace(";"," "))
            plot_kwargs['xerr'] = self.testing_target_data_error[self.plotting_index]
            plotxy.single(self.testing_target_data[self.plotting_index],
                    self.testing_target_prediction[self.plotting_index],
                    **plot_kwargs)
        self.readme_list.append("Plot single_fit.png created.\n")
        self.readme_list.append("    Plotted data is in the data_... csv file.\n")
        self.readme_list.append("    All zeros for error columns indicate no error.\n")
        return

    @timeit
    def print_readme(self):
        with open(os.path.join(self.save_path,"README"), 'w') as rfile:
            rfile.writelines(self.readme_list)
        return

