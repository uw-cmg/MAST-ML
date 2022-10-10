"""
This module contains classes for quantifying the predicted model errors (uncertainty quantification), and preparing
provided residual (true errors) predicted model error data for plotting (e.g. residual vs. error plots), or for
recalibration of model errors using the method of Palmer et al.

ErrorUtils:
    Collection of functions to conduct error analysis on certain types of models (uncertainty quantification), and prepare
    residual and model error data for plotting, as well as recalibrate model errors with various methods

CorrectionFactors
    Class for performing recalibration of model errors (uncertainty quantification) based on the method from the
    work of Palmer et al.

"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
try:
    from forestci import random_forest_error
except:
    print('forestci is an optional dependency. To install latest forestci compatabilty with scikit-learn>=0.24, run '
          'pip install git+git://github.com/scikit-learn-contrib/forest-confidence-interval.git')

class ErrorUtils():
    '''
    Collection of functions to conduct error analysis on certain types of models (uncertainty quantification), and prepare
    residual and model error data for plotting, as well as recalibrate model errors with various methods

    Args:
        None

    Methods:
        _collect_error_data: method to collect all residuals, model errors, and dataset standard deviation over many data splits
            Args:
                savepath: (str), string denoting the path to save output to

                data_type: (str), string denoting the data type analyzed, e.g. train, test, leftout

            Returns:
                model_errors: (pd.Series), series containing the predicted model errors

                residuals: (pd.Series), series containing the true model errors (residuals)

                dataset_stdev: (float), standard deviation of the data set

        _recalibrate_errors: method to recalibrate the model errors using negative log likelihood function from work of Palmer et al.
            Args:
                model_errors: (pd.Series), series containing the predicted (uncalibrated) model errors

                residuals: (pd.Series), series containing the true model errors (residuals)

            Returns:
                model_errors: (pd.Series), series containing the predicted (calibrated) model errors

                a: (float), the slope of the recalibration linear fit

                b: (float), the intercept of the recalibration linear fit

        _parse_error_data: method to prepare the provided residuals and model errors for plotting the binned RvE (residual vs error) plots
            Args:
                model_errors: (pd.Series), series containing the predicted model errors

                residuals: (pd.Series), series containing the true model errors (residuals)

                dataset_stdev: (float), standard deviation of the data set

                number_of_bins: (int), the number of bins to digitize the data into for making the RvE (residual vs. error) plot

            Returns:
                bin_values: (np.array), the x-axis of the RvE plot: reduced model error values digitized into bins

                rms_residual_values: (np.array), the y-axis of the RvE plot: the RMS of the residual values digitized into bins

                num_values_per_bin: (np.array), the number of data samples in each bin

                number_of_bins: (int), the number of bins to put the model error and residual data into.

        _get_model_errors: method for generating the model error values using either the standard deviation of weak learners or jackknife-after-bootstrap method of Wager et al.
            Args:
                model: (mastml.models object), a MAST-ML model, e.g. SklearnModel or EnsembleModel

                X: (pd.DataFrame), dataframe of the X feature matrix

                X_train: (pd.DataFrame), dataframe of the X training data feature matrix

                X_test: (pd.DataFrame), dataframe of the X test data feature matrix

                error_method: (str), string denoting the UQ error method to use. Viable options are 'stdev_weak_learners' and 'jackknife_after_bootstrap'

                remove_outlier_learners: (bool), whether specific weak learners that are found to deviate from 3 sigma of the average prediction for a given data point are removed (Default False)

            Returns:
                model_errors: (pd.Series), series containing the predicted model errors

                num_removed_learners: (list), list of number of removed weak learners for each data point

        _remove_outlier_preds: method to flag and remove outlier weak learner predictions
            Args:
                preds: (list), list of predicted values of a given data point from an ensemble of weak learners

            Returns:
                preds_cleaned: (list), ammended list of predicted values of a given data point from an ensemble of weak learners, with predictions from outlier learners removed

                num_outliers: (int), the number of removed weak learners for the data point evaluated

    '''
    @classmethod
    def _collect_error_data(cls, savepath, data_type):
        if data_type not in ['train', 'test', 'leaveout']:
            print('Error: data_test_type must be one of "train", "test" or "leaveout"')
            exit()

        dfs_error = list()
        dfs_residuals = list()
        dfs_ytrue = list()
        residuals_files_to_parse = list()
        error_files_to_parse = list()
        ytrue_files_to_parse = list()

        splits = list()
        for folder, subfolders, files in os.walk(savepath):
            if 'split' in folder:
                splits.append(folder)

        for path in splits:
            if os.path.exists(os.path.join(path, 'model_errors_'+str(data_type)+'.xlsx')):
                error_files_to_parse.append(os.path.join(path, 'model_errors_' + str(data_type) + '.xlsx'))
            if os.path.exists(os.path.join(path, 'residuals_' + str(data_type) + '.xlsx')):
                residuals_files_to_parse.append(os.path.join(path, 'residuals_' + str(data_type) + '.xlsx'))
            if os.path.exists(os.path.join(path, 'y_train.xlsx')):
                ytrue_files_to_parse.append(os.path.join(path, 'y_train.xlsx'))

        for file in residuals_files_to_parse:
            df = pd.read_excel(file)
            dfs_residuals.append(np.array(df['residuals']))

        for file in error_files_to_parse:
            df = pd.read_excel(file)
            dfs_error.append(np.array(df['model_errors']))

        for file in ytrue_files_to_parse:
            df = pd.read_excel(file)
            dfs_ytrue.append(np.array(df['y_train']))

        ytrue_all = np.concatenate(dfs_ytrue).ravel()
        dataset_stdev = np.std(np.unique(ytrue_all))

        model_errors = np.concatenate(dfs_error).ravel().tolist()
        residuals = np.concatenate(dfs_residuals).ravel().tolist()

        return model_errors, residuals, dataset_stdev

    @classmethod
    def _recalibrate_errors(cls, model_errors, residuals):
        corrector = CorrectionFactors(residuals=residuals, model_errors=model_errors)
        a, b = corrector.nll()
        # shift the model errors by the correction factor
        model_errors = pd.Series(a * np.array(model_errors) + b, name='model_errors')
        return model_errors, a, b

    @classmethod
    def _parse_error_data(cls, model_errors, residuals, dataset_stdev, number_of_bins=15):

        # Normalize the residuals and model errors by dataset stdev
        model_errors = model_errors/dataset_stdev
        residuals = residuals/dataset_stdev

        abs_res = abs(residuals)

        # check to see if number of bins should increase, and increase it if so
        model_errors_sorted = np.sort(model_errors)
        ninety_percentile = int(len(model_errors_sorted) * 0.9)
        ninety_percentile_range = model_errors_sorted[ninety_percentile] - np.amin(model_errors)
        total_range = np.amax(model_errors) - np.amin(model_errors)
        number_of_bins = number_of_bins
        if ninety_percentile_range / total_range < 5 / number_of_bins:
            number_of_bins = int(5 * total_range / ninety_percentile_range)

        # Set bins for calculating RMS
        upperbound = np.amax(model_errors)
        lowerbound = np.amin(model_errors)
        bins = np.linspace(lowerbound, upperbound, number_of_bins, endpoint=False)

        # Create a vector determining bin of each data point
        digitized = np.digitize(model_errors, bins)

        # Record which bins contain data (to avoid trying to do calculations on empty bins)
        bins_present = []
        for i in range(1, number_of_bins + 1):
            if i in digitized:
                bins_present.append(i)

        # Create array of weights based on counts in each bin
        weights = []
        for i in range(1, number_of_bins + 1):
            if i in digitized:
                weights.append(np.count_nonzero(digitized == i))

        # Calculate RMS of the absolute residuals
        RMS_abs_res = [np.sqrt((abs_res[digitized == bins_present[i]] ** 2).mean()) for i in range(0, len(bins_present))]
        MS_abs_res = [(abs_res[digitized == bins_present[i]] ** 2).mean() for i in range(0, len(bins_present))]
        Var_squarederr_res = [np.std((abs_res[digitized == bins_present[i]] ** 2)) ** 2 for i in range(0, len(bins_present))]

        # Set the x-values to the midpoint of each bin
        bin_width = bins[1] - bins[0]
        binned_model_errors = np.zeros(len(bins_present))
        for i in range(0, len(bins_present)):
            curr_bin = bins_present[i]
            binned_model_errors[i] = bins[curr_bin - 1] + bin_width / 2

        #TODO this is temporary
        bin_values = np.array(binned_model_errors)
        rms_residual_values = np.array(RMS_abs_res)
        ms_residual_values = np.array(MS_abs_res)
        var_sq_residual_values = np.array(Var_squarederr_res)
        num_values_per_bin = np.array(weights)

        return bin_values, rms_residual_values, num_values_per_bin, number_of_bins, ms_residual_values, var_sq_residual_values

    @classmethod
    def _get_model_errors(cls, model, X, X_train, X_test, error_method='stdev_weak_learners', remove_outlier_learners=False):

        err_down = list()
        err_up = list()
        indices_TF = list()
        X_aslist = X.values.tolist()
        if model.model.__class__.__name__ in ['RandomForestRegressor', 'GradientBoostingRegressor', 'ExtraTreesRegressor',
                                              'BaggingRegressor', 'AdaBoostRegressor']:

            if error_method == 'jackknife_after_bootstrap':
                model_errors_var = random_forest_error(forest=model.model, X_test=X_test, X_train=X_train)
                # Wager method returns the variance. Take sqrt to turn into stdev
                model_errors = np.sqrt(model_errors_var)
                num_removed_learners = list()
                if remove_outlier_learners is True:
                    print("Warning: removal of outlier learners isn't supported with jackknife after bootstrap")
                for _ in model_errors:
                    num_removed_learners.append(0)

            elif error_method == 'stdev_weak_learners':
                num_removed_learners = list()
                for x in range(len(X_aslist)):
                    preds = list()
                    if model.model.__class__.__name__ == 'RandomForestRegressor':
                        for pred in model.model.estimators_:
                            preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
                    elif model.model.__class__.__name__ == 'BaggingRegressor':
                        for pred in model.model.estimators_:
                            if pred.__class__.__name__ == 'KerasRegressor':
                                preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1)))
                            elif pred.__class__.__name__ == 'Sequential':
                                preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1)))
                            else:
                                preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
                    elif model.model.__class__.__name__ == 'ExtraTreesRegressor':
                        for pred in model.model.estimators_:
                            preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
                    elif model.model.__class__.__name__ == 'GradientBoostingRegressor':
                        for pred in model.model.estimators_.tolist():
                            preds.append(pred[0].predict(np.array(X_aslist[x]).reshape(1, -1))[0])
                    elif model.model.__class__.__name__ == 'AdaBoostRegressor':
                        for pred in model.model.estimators_:
                            preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])

                    # HERE flag outlier predictions, perhaps result of e.g. numerical issues in ensemble of models
                    if remove_outlier_learners == True:
                        preds, num_outliers = cls._remove_outlier_preds(preds=preds)
                        num_removed_learners.append(num_outliers)
                    else:
                        num_removed_learners.append(0)

                    e_down = np.std(preds)
                    e_up = np.std(preds)
                    err_down.append(e_down)
                    err_up.append(e_up)

                nan_indices = np.where(np.isnan(err_up))
                nan_indices_sorted = np.array(sorted(nan_indices[0], reverse=True))
                for i, val in enumerate(list(err_up)):
                    if i in nan_indices_sorted:
                        indices_TF.append(False)
                    else:
                        indices_TF.append(True)

                model_errors = (np.array(err_up) + np.array(err_down)) / 2

            else:
                print('ERROR: error_method must be one of "stdev_weak_learners" or "jackknife_after_bootstrap"')
                sys.exit()

        if model.model.__class__.__name__ == 'GaussianProcessRegressor':
            preds = model.model.predict(X, return_std=True)[1]  # Get the stdev model error from the predictions of GPR
            err_up = preds
            err_down = preds
            model_errors = (np.array(err_up) + np.array(err_down)) / 2
            nan_indices = np.where(np.isnan(err_up))
            nan_indices_sorted = np.array(sorted(nan_indices[0], reverse=True))
            num_removed_learners = list()
            for i, val in enumerate(list(err_up)):
                num_removed_learners.append(0)
                if i in nan_indices_sorted:
                    indices_TF.append(False)
                else:
                    indices_TF.append(True)

        model_errors = pd.Series(model_errors, name='model_errors')
        num_removed_learners = pd.Series(num_removed_learners, name='num_removed_learners')

        return model_errors, num_removed_learners

    @classmethod
    def _remove_outlier_preds(cls, preds):
        # set the outlier flag to be number of standard deviations, obtained by dividing 2*number of predictors by 100.
        n_stdev = 3
        mean_pred = np.mean(preds)
        stdev_pred = np.std(preds)
        preds_cleaned = list()
        num_outliers = 0
        for pred in preds:
            if pred < mean_pred-n_stdev*stdev_pred:
                num_outliers += 1
            elif pred > mean_pred+n_stdev*stdev_pred:
                num_outliers += 1
            else:
                preds_cleaned.append(pred)
        #print('num outliers', num_outliers)
        #print('Found number of outlier predictions in ensemble error analysis:', num_outliers, 'which reduces number of ensemble predictions used in error analysis from ', len(preds), 'to ', len(preds_cleaned))
        return np.array(preds_cleaned), num_outliers

class CorrectionFactors():
    '''
    Class for performing recalibration of model errors (uncertainty quantification) based on the method from the
    work of Palmer et al.

    Args:
        residuals: (pd.Series), series containing the true model errors (residuals)

        model_errors: (pd.Series), series containing the predicted model errors

    Methods:
        nll: Method to perform optimization of normalized model error distribution using the negative log likelihood function
            Args:
                None

            Returns:
                a: (float), the slope of the recalibration linear fit

                b: (float), the intercept of the recalibration linear fit

        _nll_opt: Method for evaluating the negative log likelihood function for set of residuals and model errors
            Args:
                x: (np.array), array providing the initial guess of (a, b)

            Returns:
                _ : (float), the recalibrated error value

    '''
    def __init__(self, residuals, model_errors):
        self.residuals = residuals
        self.model_errors = model_errors

    def nll(self):
        x0 = np.array([1.0, 0.0])
        res = minimize(self._nll_opt, x0, method='nelder-mead')
        a = res.x[0]
        b = res.x[1]
        success = res.success
        if success is True:
            pass
        elif success is False:
            print("Warning: NLL optimization failed!")
        # print(res)
        #r_squared = self._direct_rsquared(a, b)
        return a, b

    def _nll_opt(self, x):
        sum = 0
        for i in range(0, len(self.residuals)):
            sum += np.log(2 * np.pi) + np.log((x[0] * self.model_errors[i] + x[1]) ** 2) + (self.residuals[i]) ** 2 / (x[0] * self.model_errors[i] + x[1]) ** 2
        return 0.5 * sum / len(self.residuals)
