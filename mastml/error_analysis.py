import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import minimize
try:
    from forestci import random_forest_error
except:
    print('To install latest forestci compatabilty with scikit-learn>=0.24, run '
          'pip install git+git://github.com/scikit-learn-contrib/forest-confidence-interval.git')

class ErrorUtils():
    '''

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
        RMS_abs_res = [np.sqrt((abs_res[digitized == bins_present[i]] ** 2).mean()) for i in
                       range(0, len(bins_present))]

        # Set the x-values to the midpoint of each bin
        bin_width = bins[1] - bins[0]
        binned_model_errors = np.zeros(len(bins_present))
        for i in range(0, len(bins_present)):
            curr_bin = bins_present[i]
            binned_model_errors[i] = bins[curr_bin - 1] + bin_width / 2

        #TODO this is temporary
        bin_values = np.array(binned_model_errors)
        rms_residual_values = np.array(RMS_abs_res)
        num_values_per_bin = np.array(weights)

        return bin_values, rms_residual_values, num_values_per_bin, number_of_bins

    @classmethod
    def _get_model_errors(cls, model, X, X_train, X_test, error_method='stdev_weak_learners'):
        """
        Method to calculate prediction intervals when using Random Forest and Gaussian Process regression models.

        Prediction intervals for random forest adapted from https://blog.datadive.net/prediction-intervals-for-random-forests/

        Args:

            model: (scikit-learn model/estimator object), a scikit-learn model object

            X: (numpy array), array of X features

            method: (str), type of error bar to formulate (e.g. "stdev" is standard deviation of predicted errors, "confint"
            is error bar as confidence interval

            percentile: (float), percentile for which to form error bars

        Returns:

            err_up: (list), list of upper bounds of error bars for each data point

            err_down: (list), list of lower bounds of error bars for each data point

        """

        err_down = list()
        err_up = list()
        nan_indices = list()
        indices_TF = list()
        X_aslist = X.values.tolist()
        if model.model.__class__.__name__ in ['RandomForestRegressor', 'GradientBoostingRegressor', 'ExtraTreesRegressor',
                                              'BaggingRegressor']:

            if error_method == 'jackknife_after_bootstrap':
                model_errors_var = random_forest_error(forest=model.model, X_test=X_test, X_train=X_train)
                # Wager method returns the variance. Take sqrt to turn into stdev
                model_errors = np.sqrt(model_errors_var)

            elif error_method == 'stdev_weak_learners':
                for x in range(len(X_aslist)):
                    preds = list()
                    if model.model.__class__.__name__ == 'RandomForestRegressor':
                        for pred in model.model.estimators_:
                            preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
                    elif model.model.__class__.__name__ == 'BaggingRegressor':
                        for pred in model.model.estimators_:
                            preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
                    elif model.model.__class__.__name__ == 'GradientBoostingRegressor':
                        for pred in model.model.estimators_.tolist():
                            preds.append(pred[0].predict(np.array(X_aslist[x]).reshape(1, -1))[0])
                    elif model.model.__class__.__name__ == 'EnsembleRegressor':
                        for pred in model.model:
                            preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])

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
            preds = model.predict(X, return_std=True)[1]  # Get the stdev model error from the predictions of GPR
            err_up = preds
            err_down = preds
            nan_indices = np.where(np.isnan(err_up))
            nan_indices_sorted = np.array(sorted(nan_indices[0], reverse=True))
            for i, val in enumerate(list(err_up)):
                if i in nan_indices_sorted:
                    indices_TF.append(False)
                else:
                    indices_TF.append(True)

        model_errors = pd.Series(model_errors, name='model_errors')

        return model_errors

class CorrectionFactors():
    '''

    '''
    def __init__(self, residuals, model_errors):
        self.residuals = residuals
        self.model_errors = model_errors

    # Function to find scale factors by directly optimizing the r-stat distribution
    # The r^2 value returned is obtained by making a binned residual vs. error plot and
    # fitting a line, after scaling with the a and b found by this function.
    def direct(self):
        x0 = np.array([1.0, 0.0])
        res = minimize(self._direct_opt, x0, method='nelder-mead')
        a = res.x[0]
        b = res.x[1]
        success = res.success
        if success is True:
            print("r-stat optimization successful!")
        elif success is False:
            print("r-stat optimization failed.")
        # print(res)
        r_squared = self._direct_rsquared(a, b)
        return a, b, r_squared

    def nll(self):
        x0 = np.array([1.0, 0.0])
        res = minimize(self._nll_opt, x0, method='nelder-mead')
        a = res.x[0]
        b = res.x[1]
        success = res.success
        if success is True:
            print("NLL optimization successful!")
        elif success is False:
            print("NLL optimization failed.")
        # print(res)
        #r_squared = self._direct_rsquared(a, b)
        return a, b

    # Function to find scale factors using binned residual vs. model error plot
    def rve(self, number_of_bins=15):
        model_errors = self.model_errors
        abs_res = abs(self.residuals)

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
        RMS_abs_res = [np.sqrt((abs_res[digitized == bins_present[i]] ** 2).mean()) for i in
                       range(0, len(bins_present))]

        # Set the x-values to the midpoint of each bin
        bin_width = bins[1] - bins[0]
        binned_model_errors = np.zeros(len(bins_present))
        for i in range(0, len(bins_present)):
            curr_bin = bins_present[i]
            binned_model_errors[i] = bins[curr_bin - 1] + bin_width / 2

        # Fit a line to the data
        model = LinearRegression(fit_intercept=True)
        model.fit(binned_model_errors[:, np.newaxis],
                  RMS_abs_res,
                  sample_weight=weights)  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
        xfit = binned_model_errors
        yfit = model.predict(xfit[:, np.newaxis])

        # Calculate r^2 value
        r_squared = r2_score(RMS_abs_res, yfit, sample_weight=weights)
        # Calculate slope
        slope = model.coef_
        # Calculate y-intercept
        intercept = model.intercept_

        # print("rf slope: {}".format(slope))
        # print("rf intercept: {}".format(intercept))

        return slope, intercept, r_squared

    def _direct_opt(self, x):
        ratio = self.residuals / (self.model_errors * x[0] + x[1])
        sigma = np.std(ratio)
        mu = np.mean(ratio)
        return mu ** 2 + (sigma - 1) ** 2

    def _nll_opt(self, x):
        sum = 0
        for i in range(0, len(self.residuals)):
            sum += np.log(2 * np.pi) + np.log((x[0] * self.model_errors[i] + x[1]) ** 2) + (self.residuals[i]) ** 2 / (x[0] * self.model_errors[i] + x[1]) ** 2
        return 0.5 * sum / len(self.residuals)

    def _direct_rsquared(self, a, b, number_of_bins=15):
        model_errors = self.model_errors * a + b
        abs_res = abs(self.residuals)

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
        RMS_abs_res = [np.sqrt((abs_res[digitized == bins_present[i]] ** 2).mean()) for i in
                       range(0, len(bins_present))]

        # Set the x-values to the midpoint of each bin
        bin_width = bins[1] - bins[0]
        binned_model_errors = np.zeros(len(bins_present))
        for i in range(0, len(bins_present)):
            curr_bin = bins_present[i]
            binned_model_errors[i] = bins[curr_bin - 1] + bin_width / 2

        # Fit a line to the data
        model = LinearRegression(fit_intercept=True)
        model.fit(binned_model_errors[:, np.newaxis],
                  RMS_abs_res,
                  sample_weight=weights)  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
        xfit = binned_model_errors
        yfit = model.predict(xfit[:, np.newaxis])

        # Calculate r^2 value
        r_squared = r2_score(RMS_abs_res, yfit, sample_weight=weights)
        # Calculate slope
        slope = model.coef_
        # Calculate y-intercept
        intercept = model.intercept_

        # print("rf slope: {}".format(slope))
        # print("rf intercept: {}".format(intercept))

        return r_squared