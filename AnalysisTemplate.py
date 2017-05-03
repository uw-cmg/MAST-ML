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

class AnalysisTemplate():
    """Basic analysis class.
        Template for other classes.
        Combine many analysis classes to do meta-analysis
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
        *args, **kwargs):
        """Initialize class.
            Attributes that can be set through keywords:
                self.training_dataset=""
                self.testing_dataset=""
                self.model=""
                self.train_index=""
                self.test_index=""
                self.input_features=""
                self.target_feature=""
                self.labeling_features=""
                self.save_path=""
            Other attributes:
                self.analysis_name=""
                self.training_input_data_unfiltered=""
                self.training_target_data_unfiltered=""
                self.testing_input_data_unfiltered=""
                self.testing_target_data_unfiltered=""
                self.training_input_data=""
                self.training_target_data=""
                self.testing_input_data=""
                self.testing_target_data=""
                self.trained_model=""
                self.testing_target_prediction=""
                self.statistics=dict()
        """
        # Keyword-set attributes
        # training csv
        if training_dataset is None:
            raise ValueError("training_dataset is not set")
        if type(training_dataset) is list: #allow inheriting classes to get multiple datasets; MASTML will pass list of data objects
            self.training_dataset=training_dataset[0] #first item
        elif type(training_dataset) is str:
            self.training_dataset = data_parser.parse(training_dataset)
        else:
            self.training_dataset = training_dataset
        # testing csv
        if testing_dataset is None:
            raise ValueError("testing_dataset is not set")
        if type(testing_dataset) is list:
            self.testing_dataset = testing_dataset[0]
        elif type(testing_dataset) is str:
            self.testing_dataset = data_parser.parse(testing_dataset)
        else:
            self.testing_dataset=testing_dataset
        # model
        self.model=model
        # train/test index
        self.train_index=train_index
        self.test_index=test_index
        # features
        if input_features is None:
            raise ValueError("input_features is not set")
        self.input_features=input_features
        if target_feature is None:
            raise ValueError("target_feature is not set")
        self.target_feature=target_feature
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
        #if analysis_name is None:
        #    self.analysis_name = "Results_%s" % time.strftime("%Y%m%d_%H%M%S")
        #else:
        #    self.analysis_name = analysis_name
        # Self-set attributes
        self.analysis_name = os.path.basename(self.save_path)
        self.training_input_data_unfiltered=None
        self.training_target_data_unfiltered=None
        self.testing_input_data_unfiltered=None
        self.testing_target_data_unfiltered=None
        self.training_input_data=None
        self.training_target_data=None
        self.testing_input_data=None
        self.testing_target_data=None
        self.trained_model=None
        self.testing_target_prediction=None
        self.statistics=dict()
        #
        logger.info("-------- %s --------" % self.analysis_name)
        logger.info("Starting analysis at %s" % time.asctime())
        return
   
    @timeit
    def run(self):
        self.get_unfiltered_data()
        self.get_train_test_indices()
        self.get_data()
        self.get_model()
        self.get_trained_model()
        self.get_prediction()
        self.get_statistics()
        self.print_statistics()
        self.print_output_csv()
        self.plot_results()
        self.print_readme()
        return

    
    @timeit
    def get_unfiltered_data(self):
        """Replace with pandas dataframes
        """
        self.training_dataset.set_y_feature(self.target_feature)
        self.training_dataset.set_x_features(self.input_features)
        self.training_input_data_unfiltered = np.asarray(self.training_dataset.get_x_data())
        self.training_target_data_unfiltered = np.asarray(self.training_dataset.get_y_data()).ravel()
        hasy = self.testing_dataset.set_y_feature(self.target_feature)
        if hasy:
            self.testing_target_data_unfiltered = np.asarray(self.testing_dataset.get_y_data()).ravel()
        else:
            self.testing_dataset.set_y_feature(self.input_features[0]) #dummy y feature
            #self.testing_target_data_unfiltered remains None
        self.testing_dataset.set_x_features(self.input_features)
        self.testing_input_data_unfiltered = np.asarray(self.testing_dataset.get_x_data())
        return
    
    @timeit
    def get_train_test_indices(self):
        if (self.train_index is None):
            train_obs = self.training_input_data_unfiltered.shape[0]
            self.train_index = np.arange(0, train_obs)
        if (self.test_index is None):
            test_obs = self.testing_input_data_unfiltered.shape[0]
            self.test_index = np.arange(0, test_obs)
        return
    
    @timeit
    def get_data(self):
        self.training_input_data = self.training_input_data_unfiltered[self.train_index]
        self.training_target_data = self.training_target_data_unfiltered[self.train_index]
        self.testing_input_data = self.testing_input_data_unfiltered[self.test_index]
        if not (self.testing_target_data_unfiltered is None):
            self.testing_target_data = self.testing_target_data_unfiltered[self.test_index]
        return

    @timeit
    def get_model(self):
        if (self.model is None):
            raise ValueError("No model.")
        return

    @timeit
    def get_trained_model(self):
        trained_model = self.model.fit(self.training_input_data, self.training_target_data)
        self.trained_model = trained_model
        return

    @timeit
    def get_prediction(self):
        prediction_for_unfiltered = list()
        for pidx in range(0, self.testing_input_data_unfiltered.shape[0]):
            prediction_for_unfiltered.append(np.nan)
        prediction_for_unfiltered = np.array(prediction_for_unfiltered)
        self.testing_target_prediction = self.trained_model.predict(self.testing_input_data)
        prediction_for_unfiltered[self.test_index] = np.array(self.testing_target_prediction)
        self.testing_dataset.add_feature("Prediction",prediction_for_unfiltered)
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
    
    @timeit
    def get_statistics(self):
        if self.testing_target_data is None:
            logger.warning("No testing target data. Statistics will not be collected.")
            return
        self.statistics['rmse'] = self.get_rmse()
        self.statistics['mean_error'] = self.get_mean_error()
        self.statistics['mean_absolute_error'] = self.get_mean_absolute_error()
        self.statistics['rsquared'] = self.get_rsquared()
        return

    @timeit
    def print_statistics(self):
        statname = os.path.join(self.save_path, "statistics.txt")
        with open(statname, 'w') as statfile:
            statfile.write("Statistics\n")
            statfile.write("%s\n" % time.asctime())
            for skey, svalue in self.statistics.items():
                statfile.write("%s:%3.4f\n" % (skey, svalue))
        return

    @timeit
    def print_output_csv(self):
        """
            Modify once dataframe is in place
        """
        ocsvname = os.path.join(self.save_path, "output_data.csv")
        headerline = ""
        printarray = None
        print_features = list(self.labeling_features)
        print_features.extend(self.input_features)
        if not (self.testing_target_data is None):
            print_features.append(self.target_feature)
        for feature_name in print_features:
            headerline = headerline + feature_name + ","
            feature_vector = np.asarray(self.testing_dataset.get_data(feature_name)).ravel()[self.test_index]
            if printarray is None:
                printarray = feature_vector
            else:
                printarray = np.vstack((printarray, feature_vector))
        headerline = headerline + "Prediction"
        printarray = np.vstack((printarray, self.testing_target_prediction))
        printarray=printarray.transpose()
        ptools.mixed_array_to_csv(ocsvname, headerline, printarray)
        return
    
    @timeit
    def plot_results(self, addl_plot_kwargs=None):
        if self.testing_target_data is None:
            logger.warning("No testing target data. Predicted vs. measured plot will not be plotted.")
            return
        plot_kwargs=dict()
        plot_kwargs['xlabel'] = "Measured"
        plot_kwargs['ylabel'] = "Predicted"
        plot_kwargs['guideline'] = 1
        notelist=list()
        notelist.append("RMSE: %3.3f" % self.statistics['rmse'])
        notelist.append("R-squared: %3.3f" % self.statistics['rsquared'])
        plot_kwargs['notelist'] = notelist
        plot_kwargs['save_path'] = self.save_path
        if not (addl_plot_kwargs is None):
            for addl_plot_kwarg in addl_plot_kwargs:
                plot_kwargs[addl_plot_kwarg] = addl_plot_kwargs[addl_plot_kwarg]
        plotxy.single(self.testing_target_data,
                self.testing_target_prediction,
                **plot_kwargs)
        return

    @timeit
    def print_readme(self):
        rlist=list()
        rlist.append("----- Folder contents -----\n")
        rlist.append("output_data.csv:\n")
        rlist.append("    Output data, consisting of:\n")
        rlist.append("    Labeling features columns, if any\n")
        rlist.append("    Input features columns\n")
        if self.testing_target_data is None:
            pass
        else:
            rlist.append("    Target feature column\n")
        rlist.append("    Target prediction column\n")
        rlist.append("statistics.txt:\n")
        rlist.append("    Statistics for the fit.\n")
        if self.testing_target_data is None:
            rlist.append("    No statistics were collected because there\n")
            rlist.append("        was no measured data (target feature data)\n")
            rlist.append("        in the testing dataset.\n")
            rlist.append("    For this reason, there is also no plot of\n")
            rlist.append("        predicted vs. measured data.\n")
        else:
            rlist.append("{y} vs {x}.png:\n")
            rlist.append("    Plot of predicted vs. measured data\n")
            rlist.append("data_ .csv\n")
            rlist.append("    Plotted data, in csv format\n")
            rlist.append("    Measured data column\n")
            rlist.append("    Measured data error column (all zeros is no error)\n")
            rlist.append("    Predicted data column\n")
            rlist.append("    Predicted data error column (all zeros is no error)\n")
        with open(os.path.join(self.save_path,"README"), 'w') as rfile:
            rfile.writelines(rlist)
        return

