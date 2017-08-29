import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from LeaveOutPercentCV import LeaveOutPercentCV
from SingleFit import timeit

class KFoldCV(LeaveOutPercentCV):
    """KFold cross validation
   
    Args:
        training_dataset, (Should be the same as testing_dataset)
        testing_dataset, (Should be the same as training_dataset)
        model,
        save_path,
        xlabel, 
        ylabel,
        mark_outlying_points,
        num_cvtests, (each test contains K folds)
        fix_random_for_testing: see parent class.
        num_folds (int): Number of folds for KFold CV
 
    Returns:
        Analysis in the save_path folder
        Plots results in a predicted vs. measured square plot.
    Raises:
        ValueError: if testing target data is None; CV must have
                testing target data
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
        self.plot_best_worst_overlay(notelist=list(notelist))
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
            self.cvtest_dict[cvtest]["avg_rmse"] = np.mean(fold_rmses)
            self.cvtest_dict[cvtest]["std_rmse"] = np.std(fold_rmses)
            self.cvtest_dict[cvtest]["avg_mean_error"] = np.mean(fold_mean_errors)
            self.cvtest_dict[cvtest]["std_mean_error"] = np.std(fold_mean_errors)
            self.cvtest_dict[cvtest]["fold_array"] = fold_array
            self.cvtest_dict[cvtest]["prediction_array"] = prediction_array
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
