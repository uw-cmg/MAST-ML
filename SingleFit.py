import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from plot_data.PlotHelper import PlotHelper
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
        training_dataset <DataHandler object>: Training dataset handler
        testing_dataset <DataHandler object>: Testing dataset handler
        model <sklearn model>: Machine-learning model
        save_path <str>: Save path
        xlabel <str>: Label for full-fit x-axis (default "Measured")
        ylabel <str>: Label for full-fit y-axis (default "Predicted")
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
        xlabel="Measured",
        ylabel="Predicted",
        plot_filter_out=None,
        *args, **kwargs):
        """Initialize class.
            Attributes that can be set through keywords:
                self.training_dataset
                self.testing_dataset
                self.model
                self.save_path
                self.xlabel
                self.ylabel
                self.plot_filter_out
            Other attributes:
                self.analysis_name <str>
                self.trained_model <sklearn model>: trained model
                self.statistics <dict of float>: statistics dictionary
                self.readme_list <dict of str>: stores lines for readme
        """
        # Keyword-set attributes
        # training csv
        if training_dataset is None:
            raise ValueError("training_dataset is not set")
        if type(training_dataset) is list: #allow inheriting classes to get multiple datasets; MASTML will pass list of data objects
            self.training_dataset= copy.deepcopy(training_dataset[0]) #first item
        else:
            self.training_dataset = copy.deepcopy(training_dataset)
        # testing csv
        if testing_dataset is None:
            raise ValueError("testing_dataset is not set")
        if type(testing_dataset) is list:
            self.testing_dataset = copy.deepcopy(testing_dataset[0])
        else:
            self.testing_dataset=copy.deepcopy(testing_dataset)
        # model
        if model is None:
            raise ValueError("No model.")
        self.model=model
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
        self.plot_filter_out = self.get_plot_filter(plot_filter_out)
        # Self-set attributes
        self.analysis_name = os.path.basename(self.save_path)
        self.trained_model=None
        self.statistics=dict()
        self.readme_list=list()
        self.testing_dataset_filtered=None
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
        self.plot_results()
        return



    def get_trained_model(self):
        trained_model = self.model.fit(self.training_dataset.input_data, self.training_dataset.target_data)
        self.trained_model = trained_model
        return
    
    def print_model(self):
        self.readme_list.append("----- Model parameters -----\n")
        for param, paramval in self.trained_model.get_params().items():
            self.readme_list.append("%s: %s\n" % (param, paramval))
        return

    def get_prediction(self):
        target_prediction = self.trained_model.predict(self.testing_dataset.input_data)
        self.testing_dataset.add_prediction(target_prediction)
        return

    def get_rmse(self):
        rmse = np.sqrt(mean_squared_error(self.testing_dataset.target_data, self.testing_dataset.target_prediction))        
        return rmse
    
    def get_mean_error(self):
        pred_minus_true =self.testing_dataset.target_prediction-self.testing_dataset.target_data
        mean_error = np.mean(pred_minus_true)
        return mean_error

    def get_mean_absolute_error(self):
        mean_abs_err = mean_absolute_error(self.testing_dataset.target_data, self.testing_dataset.target_prediction)
        return mean_abs_err

    def get_rsquared(self):
        rsquared = r2_score(self.testing_dataset.target_data, self.testing_dataset.target_prediction)
        return rsquared
    
    def get_statistics(self):
        if self.testing_dataset.target_data is None:
            logger.warning("No testing target data. Statistics will not be collected.")
            return
        self.statistics['rmse'] = self.get_rmse()
        self.statistics['mean_error'] = self.get_mean_error()
        self.statistics['mean_absolute_error'] = self.get_mean_absolute_error()
        self.statistics['rsquared'] = self.get_rsquared()
        self.plot_filter_update_statistics()
        return

    def print_statistics(self):
        self.readme_list.append("----- Statistics -----\n")
        if len(self.statistics) == 0:
            self.readme_list.append("No target data.\n")
            self.readme_list.append("No statistics comparing target data and prediction were collected.\n")
        else:
            skeys = list(self.statistics.keys())
            skeys.sort()
            for skey in skeys:
                self.readme_list.append("%s: %3.4f\n" % (skey, self.statistics[skey]))
        return

    def print_output_csv(self, csvname="output_data.csv"):
        """
            Modify once dataframe is in place
        """
        self.readme_list.append("----- Output data -----\n")
        ocsvname = os.path.join(self.save_path, csvname)
        cols_printed = self.testing_dataset.print_data(ocsvname)
        self.readme_list.append("%s file created with columns:\n" % csvname)
        for col in cols_printed:
            self.readme_list.append("    %s\n" % col)
        return

    def get_plot_filter(self, plot_filter_out):
        if plot_filter_out is None:
            self.plot_filter_out = None
            return
        pf_tuple_list = list()
        if type(plot_filter_out) is list:
            pfo_list = plot_filter_out
        elif type(plot_filter_out) is str:
            pfo_list = plot_filter_out.split(",")
        for pfo_item in pfo_list:
            if type(pfo_item) is tuple:
                pf_tuple_list.append(pfo_item)
            else:
                pflist = pfo_item.split(";")
                feature = pflist[0].strip()
                symbol = pflist[1].strip()
                rawvalue = pflist[2].strip()
                try:
                    value = float(rawvalue)
                except ValueError:
                    value = rawvalue
                pf_tuple_list.append((feature, symbol, value))
        return pf_tuple_list
        
    def plot_filter_update_statistics(self):
        if self.plot_filter_out is None:
            return
        if self.testing_dataset.target_data is None:
            return
        self.testing_dataset.add_filters(self.plot_filter_out)
        self.print_output_csv("output_data_filtered.csv")
        rmse_pfo = np.sqrt(mean_squared_error(self.testing_dataset.target_prediction, self.testing_dataset.target_data)) 
        self.statistics['filtered_rmse'] = rmse_pfo
        return
    
    def plot_results(self, addl_plot_kwargs=None):
        self.readme_list.append("----- Plotting -----\n")
        if self.testing_dataset.target_data is None:
            logger.warning("No testing target data. Predicted vs. measured plot will not be plotted.")
            self.readme_list.append("No target data.\n")
            self.readme_list.append("No plot comparing predicted vs. measured data was made.\n")
            return
        plot_kwargs=dict()
        plot_kwargs['xlabel'] = self.xlabel
        plot_kwargs['ylabel'] = self.ylabel
        plot_kwargs['plotlabel'] = "single_fit"
        plot_kwargs['guideline'] = 1
        notelist=list()
        notelist.append("RMSE: %3.3f" % self.statistics['rmse'])
        notelist.append("R-squared: %3.3f" % self.statistics['rsquared'])
        notelist.append("Mean error: %3.3f" % self.statistics['mean_error'])
        notelist.append("Mean abs error: %3.3f" % self.statistics['mean_absolute_error'])
        plot_kwargs['notelist'] = notelist
        plot_kwargs['save_path'] = self.save_path
        plot_kwargs['xerr'] = self.testing_dataset.target_error_data
        if not (addl_plot_kwargs is None):
            for addl_plot_kwarg in addl_plot_kwargs:
                plot_kwargs[addl_plot_kwarg] = addl_plot_kwargs[addl_plot_kwarg]
        if not(self.plot_filter_out is None):
            self.readme_list.append("Plot filtering out:\n")
            for (feature, symbol, threshold) in self.plot_filter_out:
                self.readme_list.append("  %s %s %s\n" % (feature, symbol, threshold))
            notelist.append("Shown-only RMSE: %3.3f" % self.statistics['filtered_rmse'])
            notelist.append("Data not shown:")
            for (feature, symbol, threshold) in self.plot_filter_out:
                notelist.append("  %s %s %s" % (feature, symbol, threshold))
        #Data should already have been filtered by now
        plot_kwargs['xdatalist'] = [self.testing_dataset.target_data]
        plot_kwargs['ydatalist'] = [self.testing_dataset.target_prediction]
        plot_kwargs['xerrlist']=[None]
        plot_kwargs['yerrlist']=[None]
        plot_kwargs['labellist'] = ["predicted_vs_measured"]
        myph = PlotHelper(**plot_kwargs)
        myph.multiple_overlay()
        self.readme_list.append("Plot single_fit.png created.\n")
        self.readme_list.append("    Plotted data is in the data_... csv file.\n")
        self.readme_list.append("    Error column of all zeros indicates no error.\n")
        return

    @timeit
    def print_readme(self):
        with open(os.path.join(self.save_path,"README"), 'w') as rfile:
            rfile.writelines(self.readme_list)
        return

