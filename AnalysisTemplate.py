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

class Analysis:
    """Basic analysis class.
        Template for other classes.
        Combine many analysis classes to do meta-analysis
    """
    def __init__(self, 
        training_csv=None,
        testing_csv=None,
        model=None,
        train_index=None,
        test_index=None,
        input_features=None,
        target_feature=None,
        labeling_features=None,
        save_path=None,
        analysis_name=None,
        *args, **kwargs):
        """Initialize class.
            Attributes that can be set through keywords:
                self.training_csv=""
                self.testing_csv=""
                self.model=""
                self.train_index=""
                self.test_index=""
                self.input_features=""
                self.target_feature=""
                self.labeling_features=""
                self.save_path=""
                self.analysis_name=""
            Other attributes:
                self.training_dataset=""
                self.testing_dataset=""
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
        print(kwargs)
        self.training_csv=None
        self.testing_csv=None
        self.model=None
        self.train_index=None
        self.test_index=None
        self.input_features=None
        self.target_feature=None
        self.labeling_features=None
        self.save_path=None
        self.analysis_name=None
        print(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
            print(key, value)
        print(self.input_features)
        #
        if self.save_path is None:
            self.save_path = os.getcwd()
        self.save_path = os.path.abspath(self.save_path)
        if self.analysis_name is None:
            self.analysis_name = "Results_%s" % time.strftime("%Y%m%d_%H%M%S")
        if not os.path.isdir(self.analysis_name):
            os.mkdir(os.path.join(self.save_path, self.analysis_name))
        if self.labeling_features is None:
            self.labeling_features = list(self.input_features)
        #
        self.training_dataset=""
        self.testing_dataset=""
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
        #
        self.do_analysis()
        return

    def do_analysis(self):
        self.get_datasets()
        self.get_unfiltered_data()
        self.get_test_train_indices()
        self.get_data()
        self.get_model()
        self.get_trained_model()
        self.get_prediction()
        self.get_statistics()
        self.print_statistics()
        self.print_output_csv()
        self.plot_results()
        return

    def get_datasets(self):
        """Get datasets. 
            Replace depending on configuration file modifications
        """
        self.training_dataset = data_parser.parse(self.training_csv)
        self.testing_dataset = data_parser.parse(self.testing_csv)
        return
    
    def get_unfiltered_data(self):
        """Replace with pandas dataframes
        """
        self.training_dataset.set_x_features(self.input_features)
        self.training_dataset.set_y_feature(self.target_feature)
        self.training_input_data_unfiltered = np.asarray(self.training_dataset.get_x_data())
        self.training_target_data_unfiltered = np.asarray(self.training_dataset.get_y_data()).ravel()
        self.testing_dataset.set_x_features(self.input_features)
        self.testing_input_data_unfiltered = np.asarray(self.testing_dataset.get_x_data())
        hasy = self.testing_dataset.set_y_feature(self.target_feature)
        if hasy:
            self.testing_target_data_unfiltered = np.asarray(self.testing_dataset.get_y_data()).ravel()
        else:
            pass #keep self.testing_target_data as None
        return
    
    def get_train_test_indices(self):
        if (self.train_index is None):
            train_obs = self.training_input_data_unfiltered.shape[0]
            self.train_index = np.arange(0, train_obs)
        if (self.test_index is None):
            test_obs = self.testing_input_data_unfiltered.shape[0]
            self.test_index = np.arange(0, test_obs)
        return

    def get_data(self):
        self.training_input_data = self.training_input_data_unfiltered[self.train_index]
        self.training_target_data = self.training_target_data_unfiltered[self.train_index]
        self.testing_input_data = self.testing_input_data_unfiltered[self.test_index]
        self.testing_target_data = self.testing_target_data_unfiltered[self.test_index]
        return


    def get_model(self):
        if (self.model is None):
            raise ValueError("No model.")
        return
    def get_trained_model(self):
        trained_model = model.fit(self.training_input_data, self.training_target_data)
        self.trained_model = trained_model
        return

    def get_prediction(self):
        self.testing_target_prediction = self.trained_model.predict(self.testing_input_data)
        return

    def get_rmse(self):
        rmse = np.sqrt(mean_squared_error(self.testing_target_data, self.testing_target_prediction))        
        return rmse
    
    def get_mean_error(self):
        pred_minus_true =self.testing_target_prediction-self.testing_target_data
        mean_error = np.mean(pred_minus_true)
        return mean_error

    def get_mean_absolute_error(self):
        mean_absolute_error = mean_absolute_error(self.testing_target_data, self.testing_target_prediction)
        return mean_absolute_error

    def get_rsquared(self):
        rsquared = r2_score(self.testing_target_data, self.testing_target_prediction)
        return rsquared

    def get_statistics(self):
        self.statistics['rmse'] = self.get_rmse()
        self.statistics['mean_error'] = self.get_mean_error()
        self.statistics['mean_absolute_error'] = self.get_mean_absolute_error()
        self.statistics['rsquared'] = self.get_rsquared()
        return

    def print_statistics(self):
        statname = os.path.join(self.save_path, self.analysis_name, "statistics.txt")
        with open(statname, 'w') as statfile:
            statfile.write("Statistics\n")
            statfile.write("%s\n" % time.asctime())
            for skey, svalue in self.statistics.keys():
                statfile.write("%s:%3.4f" % (skey, svalue))
        return

    def print_output_csv(self):
        """
            Modify once dataframe is in place
        """
        ocsvname = os.path.join(self.save_path, self.analysis_name, "output_data.csv")
        headerline = ""
        printarray = ""
        print_features = self.labeling_features
        print_features.extend(self.input_features)
        print_features.extend(self.target_feature)
        for feature_name in print_features:
            headerline = headerline + feature_name + ","
            feature_vector = np.asarray(self.testing_data.get_data(feature_name)).ravel()[self.test_index]
            if printarray is None:
                printarray = feature_vector
            else:
                printarray = np.vstack((printarray, feature_vector))
        headerline = headerline + "Prediction"
        printarray = np.vstack((printarray, self.testing_target_prediction))
        printarray=printarray.transpose()
        ptools.mixed_array_to_csv(ocsvname, headerline, printarray)
        return

    def plot_results(self):
        plot_kwargs=dict()
        plot_kwargs['xlabel'] = "Measured"
        plot_kwargs['ylabel'] = "Predicted"
        plot_kwargs['guideline'] = 1
        notelist=list()
        notelist.append("RMSE: %3.3f" % self.statistics['rmse'])
        notelist.append("R-squared: %3.3f" % self.statistics['rsquared'])
        plot_kwargs['notelist'] = notelist
        plot_kwargs['save_path'] = os.path.join(self.save_path, self.analysis_name)
        plotxy.single(self.testing_target_data,
                self.testing_target_prediction,
                **plot_kwargs)
        return

def execute(model="", data="", savepath="", lwr_data="",
        training_csv=None,
        testing_csv=None,
        train_index=None,
        test_index=None,
        input_features=None,
        target_feature=None,
        labeling_features=None,
        save_path=None,
        analysis_name=None,
        *args, **kwargs):
    """Remove once alltests is updated
    """
    kwargs=dict()
    kwargs['training_csv'] = training_csv
    kwargs['testing_csv'] = testing_csv
    kwargs['model'] = model
    kwargs['train_index'] = train_index
    kwargs['test_index'] = test_index
    kwargs['input_features'] = list(input_features.split(",")) #LIST
    kwargs['target_feature'] = target_feature
    kwargs['labeling_features'] = labeling_features
    kwargs['save_path'] = save_path
    kwargs['analysis_name'] = analysis_name
    print(kwargs)
    mya = Analysis(**kwargs)
    return
        

