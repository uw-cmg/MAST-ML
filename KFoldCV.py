__author__ = 'Tam Mayeshiba'
__maintainer__ = 'Ryan Jacobs'
__version__ = '1.0'
__email__ = 'rjacobs3@wisc.edu'
__date__ = 'October 14th, 2017'

import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from LeaveOutPercentCV import LeaveOutPercentCV
from SingleFit import timeit

class KFoldCV(LeaveOutPercentCV):
    """Class to conduct k-fold cross-validation analysis
   
    Args:
        training_dataset (DataHandler object): Training dataset handler
        testing_dataset (DataHandler object): Testing dataset handler
        model (sklearn model object): sklearn model
        save_path (str): Save path
        xlabel (str): Label for full-fit x-axis (default "Measured")
        ylabel (str): Label for full-fit y-axis (default "Predicted")
        mark_outlying_points (list of int): Number of outlying points to mark in best and worst tests, e.g. [0,3]

        num_cvtests (int): number of CV tests to perform. Each CV test contains K folds as set by num_folds
        fix_random_for_testing (int):
                1 - fix random shuffle for testing purposes
                0 (default) Use random shuffle.
        num_folds (int): Number of folds for KFold CV
 
    Returns:
        Analysis in the save_path folder
        Plots results in a predicted vs. measured square plot.

    Raises:
        ValueError: if testing target data is None; CV must have testing target data

    """
    def __init__(self, 
        training_dataset=None,
        testing_dataset=None,
        model=None,
        save_path=None,
        xlabel="Measured",
        ylabel="Predicted",
        mark_outlying_points=None,
        num_cvtests=10,
        fix_random_for_testing=0,
        num_folds=2,
        *args, **kwargs):
        """
        Additional class attributes to parent class:
            Set by keyword:
            self.num_folds <int>: Number of folds for KFold test
            Set in code:
        """
        if not(training_dataset == testing_dataset):
            raise ValueError("Only testing_dataset will be used. Use the same values for training_dataset and testing_dataset")
        LeaveOutPercentCV.__init__(self, 
            training_dataset=training_dataset, #only testing_dataset is used
            testing_dataset=testing_dataset,
            model=model, 
            save_path = save_path,
            xlabel=xlabel,
            ylabel=ylabel,
            mark_outlying_points = mark_outlying_points,
            percent_leave_out = -1, #not using this field
            num_cvtests = num_cvtests,
            fix_random_for_testing = fix_random_for_testing)
        self.num_folds = int(num_folds)
        return 

    @timeit
    def plot(self):
        self.readme_list.append("----- Plotting -----\n")
        notelist=list()
        notelist.append("Mean over %i tests of:" % self.num_cvtests)
        notelist.append("  {:d}-fold-average RMSE:".format(self.num_folds))
        notelist.append("    {:.2f} $\pm$ {:.2f}".format(self.statistics['avg_fold_avg_rmses'], self.statistics['std_fold_avg_rmses']))
        notelist.append("  {:d}-fold-average mean error:".format(self.num_folds))
        notelist.append("    {:.2f} $\pm$ {:.2f}".format(self.statistics['avg_fold_avg_mean_errors'], self.statistics['std_fold_avg_mean_errors']))
        notelist.append("R-squared:" "{:.2f}".format(self.statistics['r2_score']))
        notelist.append("R-squared (no int): " "{:.2f}".format(self.statistics['r2_score_noint']))
        self.plot_best_worst_overlay(notelist=list(notelist))
        self.plot_meancv_overlay(notelist=list(notelist))
        self.plot_residuals_histogram()
        return

    def set_up_cv(self):
        if self.testing_dataset.target_data is None:
            raise ValueError("Testing target data cannot be none for cross validation.")
        indices = np.arange(0, len(self.testing_dataset.target_data))
        self.readme_list.append("----- CV setup -----\n")
        self.readme_list.append("%i CV tests,\n" % self.num_cvtests)
        self.readme_list.append("each with %i folds\n" % self.num_folds)
        self.cvmodel = KFold(n_splits = self.num_folds, shuffle=True, 
                                random_state = None)
        for cvtest in range(0, self.num_cvtests):
            self.cvtest_dict[cvtest] = dict()
            foldidx=0
            fold_rmses = np.zeros(self.num_folds)
            for train, test in self.cvmodel.split(indices):
                fdict=dict()
                fdict['train_index'] = train
                fdict['test_index'] = test
                self.cvtest_dict[cvtest][foldidx] = dict(fdict)
                foldidx = foldidx + 1
        return
    
    def cv_fit_and_predict(self):
        for cvtest in self.cvtest_dict.keys():
            fold_rmses = np.zeros(self.num_folds)
            fold_mean_errors = np.zeros(self.num_folds)
            fold_array = np.zeros(len(self.testing_dataset.target_data))
            prediction_array = np.zeros(len(self.testing_dataset.target_data))
            error_array = np.zeros(len(self.testing_dataset.target_data))
            for fold in self.cvtest_dict[cvtest].keys():
                fdict = self.cvtest_dict[cvtest][fold]
                input_train = self.testing_dataset.input_data.iloc[fdict['train_index']]
                target_train = self.testing_dataset.target_data[fdict['train_index']]
                input_test = self.testing_dataset.input_data.iloc[fdict['test_index']]
                target_test = self.testing_dataset.target_data[fdict['test_index']]
                fit = self.model.fit(input_train, target_train)
                predict_test = self.model.predict(input_test)
                rmse = np.sqrt(mean_squared_error(predict_test, target_test))
                merr = np.mean(predict_test - target_test)
                fold_rmses[fold] = rmse
                fold_mean_errors[fold] = merr
                fold_array[fdict['test_index']] = fold
                prediction_array[fdict['test_index']] = predict_test
                error_array[fdict['test_index']] = predict_test - target_test
            self.cvtest_dict[cvtest]["avg_rmse"] = np.mean(fold_rmses)
            self.cvtest_dict[cvtest]["std_rmse"] = np.std(fold_rmses)
            self.cvtest_dict[cvtest]["avg_mean_error"] = np.mean(fold_mean_errors)
            self.cvtest_dict[cvtest]["std_mean_error"] = np.std(fold_mean_errors)
            self.cvtest_dict[cvtest]["fold_array"] = fold_array
            self.cvtest_dict[cvtest]["prediction_array"] = prediction_array
            self.cvtest_dict[cvtest]["error_array"] = error_array
        return

    def get_statistics(self):
        cvtest_avg_rmses = list()
        cvtest_avg_mean_errors = list()
        for cvtest in range(0, self.num_cvtests):
            cvtest_avg_rmses.append(self.cvtest_dict[cvtest]["avg_rmse"])
            cvtest_avg_mean_errors.append(self.cvtest_dict[cvtest]["avg_mean_error"])
        highest_rmse = max(cvtest_avg_rmses)
        self.worst_test_index = cvtest_avg_rmses.index(highest_rmse)
        lowest_rmse = min(cvtest_avg_rmses)
        self.best_test_index = cvtest_avg_rmses.index(lowest_rmse)
        self.statistics['avg_fold_avg_rmses'] = np.mean(cvtest_avg_rmses)
        self.statistics['std_fold_avg_rmses'] = np.std(cvtest_avg_rmses)
        self.statistics['avg_fold_avg_mean_errors'] = np.mean(cvtest_avg_mean_errors)
        self.statistics['std_fold_avg_mean_errors'] = np.std(cvtest_avg_mean_errors)
        self.statistics['fold_avg_rmse_best'] = lowest_rmse
        self.statistics['fold_avg_rmse_worst'] = highest_rmse
        self.statistics['residuals'] = self.get_cv_residuals()

        # Get average CV values and errors
        average_prediction = self.cvtest_dict[0]["prediction_array"]
        error = self.cvtest_dict[0]["error_array"]
        num_data = len(self.cvtest_dict[0]["error_array"].tolist())
        for cvtest in self.cvtest_dict.keys():
            if cvtest > 0:
                average_prediction += self.cvtest_dict[cvtest]["prediction_array"]
                error = np.vstack((error, self.cvtest_dict[cvtest]["error_array"]))
        average_prediction /= self.num_cvtests
        stand_err_in_mean_err = np.nanstd(error, axis=0, ddof=1) / np.sqrt(self.num_cvtests)
        std_err = np.nanstd(error, axis=0, ddof=1)

        self.statistics['stand_err_in_mean_err'] = stand_err_in_mean_err
        self.statistics['std_err'] = std_err
        self.statistics['average_prediction'] = average_prediction

        rsquared = self.get_rsquared(Xdata=self.testing_dataset.target_data, ydata=average_prediction)
        rsquared_noint = self.get_rsquared_noint(Xdata=self.testing_dataset.target_data, ydata=average_prediction)

        self.statistics['r2_score'] = rsquared
        self.statistics['r2_score_noint'] = rsquared_noint

        return
    
    def print_best_worst_output_csv(self, label=""):
        """
        """
        olabel = "%s_test_data.csv" % label
        ocsvname = os.path.join(self.save_path, olabel)
        self.testing_dataset.add_feature("Best Prediction", 
                    self.cvtest_dict[self.best_test_index]['prediction_array'])
        self.testing_dataset.add_feature("Folds for best", 
                    self.cvtest_dict[self.best_test_index]['fold_array'])
        self.testing_dataset.add_feature("Worst Prediction", 
                    self.cvtest_dict[self.worst_test_index]['prediction_array'])
        self.testing_dataset.add_feature("Folds for worst", 
                    self.cvtest_dict[self.worst_test_index]['fold_array'])
        addl_cols = list()
        addl_cols.append("Best Prediction")
        addl_cols.append("Folds for best")
        addl_cols.append("Worst Prediction")
        addl_cols.append("Folds for worst")
        cols = self.testing_dataset.print_data(ocsvname, addl_cols)
        self.readme_list.append("%s file created with columns:\n" % olabel)
        for col in cols:
            self.readme_list.append("    %s\n" % col)
        return
