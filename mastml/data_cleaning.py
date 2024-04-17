"""
This module provides various methods for cleaning data that has been imported into MAST-ML, prior to model fitting.

DataCleaning:
    Class that enables easy use of various data cleaning methods, such as removal of missing values, different
    modes of data imputation, or using principal componenet analysis to fill interpolate missing values.

DataUtilities:
    Support class used to evaluate some basic statistics of imported data, such as its distribution, mean, etc.
    Also provides a means of flagging potential outlier datapoints based on their deviation from the overall data
    distribution.

PPCA:
    Class used by the PCA data cleaning routine in the DataCleaning class to perform probabilistic PCA to fill in
    missing data.

"""

import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.linalg import orth
try:
    from collections.abc import Counter
except ImportError:
    from collections import Counter
from datetime import datetime

from mastml.plots import Histogram

class DataCleaning():
    """
    Class to perform various data cleaning operations, such as imputation or NaN removal

    Args:
        None

    Methods:
        remove: Method that removes a full column or row of data values if one column or row contains NaN or is blank
            Args:
                X: (pd.DataFrame), dataframe containing X data

                y: (pd.Series), series containing y data

                axis: (int), whether to remove rows (axis=0) or columns (axis=1)

            Returns:
                X: (pd.DataFrame): dataframe of cleaned X data

                y: (pd.Series): series of cleaned y data

        imputation: Method that imputes values to the missing places based on the median, mean, etc. of the data in the column
            Args:
                X: (pd.DataFrame), dataframe containing X data

                y: (pd.Series), series containing y data

                strategy: (str), method of imputation, e.g. median, mean, etc.

            Returns:
                X: (pd.DataFrame): dataframe of cleaned X data

                y: (pd.Series): series of cleaned y data

        ppca: Method that imputes data using principal component analysis to interpolate missing values
            Args:
                X: (pd.DataFrame), dataframe containing X data

                y: (pd.Series), series containing y data

            Returns:
                X: (pd.DataFrame): dataframe of cleaned X data

                y: (pd.Series): series of cleaned y data

        evaluate: Main method to evaluate initial data analysis routines (e.g. flag outliers), perform data cleaning and save output to folder
            Args:
                X: (pd.DataFrame), dataframe containing X data

                y: (pd.Series), series containing y data

                method: (str), data cleaning method name, must be one of 'remove', 'imputation' or 'ppca'

                savepath: (str), string containing the savepath information

                kwargs: additional keyword arguments needed for the remove, imputation or ppca methods

            Returns:
                X: (pd.DataFrame): dataframe of cleaned X data

                y: (pd.Series): series of cleaned y data

        _setup_savedir: method to create a savedir based on the provided model, splitter, selector names and datetime
            Args:
                savepath: (str), string designating the savepath

            Returns:
                splitdir: (str), string containing the new subdirectory to save results to

    """

    def __init__(self):
        pass

    def remove(self, X, y, axis):
        df = pd.concat([X, y], axis=1)
        try:
            target = y.name
        except:
            target = y.columns.tolist()[0]
        df = df.dropna(axis=axis, how='any')
        y = df[target]
        X = df[[col for col in df.columns if col != target]]
        return X, y

    def imputation(self, X, y, strategy):
        df = pd.concat([X, y], axis=1)
        columns = df.columns.tolist()
        df = pd.DataFrame(SimpleImputer(missing_values=np.nan, strategy=strategy).fit_transform(df), columns=columns)
        try:
            target = y.name
        except:
            target = y.columns.tolist()[0]
        y = df[target]
        X = df[[col for col in df.columns if col != target]]
        return X, y

    def ppca(self, X, y):
        df = pd.concat([X, y], axis=1)
        try:
            target = y.name
        except:
            target = y.columns.tolist()[0]
        columns = df.columns.tolist()
        pca_magic = PPCA()
        pca_magic.fit(np.array(df))

        # Need to un-standardize the pca-transformed data
        df = pd.DataFrame(pca_magic.data*pca_magic.stds+pca_magic.means, columns=columns)

        y = df[target]
        X = df[[col for col in columns if col != target]]
        return X, y

    def evaluate(self, X, y, method, savepath=None, make_new_dir=True, **kwargs):
        if not savepath:
            savepath = os.getcwd()
        if make_new_dir is True:
            splitdir = self._setup_savedir(savepath=savepath)
            savepath = splitdir
        self.splitdir = splitdir
        DataUtilities().flag_columns_with_strings(X=X, y=y, savepath=savepath)
        DataUtilities().flag_outliers(X=X, y=y, savepath=savepath, n_stdevs=3)
        df_orig = pd.concat([X, y], axis=1)
        self.cleaner = getattr(self, method)
        X, y = self.cleaner(X, y, **kwargs)
        df_cleaned = pd.concat([X, y], axis=1)
        df_orig.to_excel(os.path.join(savepath, 'data_original.xlsx'), index=False)
        df_cleaned.to_excel(os.path.join(savepath, 'data_cleaned.xlsx'), index=False)

        # Make histogram of the input data
        Histogram.plot_histogram(df=y, file_name='histogram_target_values', savepath=savepath, x_label='Target values')

        return X, y

    def _setup_savedir(self, savepath):
        now = datetime.now()
        dirname = self.__class__.__name__
        dirname = f"{dirname}_{now.month:02d}_{now.day:02d}" \
            f"_{now.hour:02d}_{now.minute:02d}_{now.second:02d}"
        if savepath == None:
            splitdir = os.getcwd()
        else:
            splitdir = os.path.join(savepath, dirname)
        if not os.path.exists(splitdir):
            os.mkdir(splitdir)
        return splitdir


class DataUtilities():
    """
    Class that contains some basic data analysis utilities, such as flagging columns that contain problematic string
    entries, or flagging potential outlier values based on threshold values

    Args:
        None

    Methods:
        flag_outliers: Method that scans values in each X feature matrix column and flags values that are larger than X standard deviations from the average of that column value. The index and column values of potentially problematic points are listed and written to an output file.
            Args:
                X: (pd.DataFrame), dataframe containing X data

                y: (pd.Series), series containing y data

                savepath: (str), string containing the save path directory

                n_stdevs: (int), number of standard deviations to use as threshold value

            Returns:
                None

        flag_columns_with_strings: Method that ascertains which columns in data contain string entries
            Args:
                X: (pd.DataFrame), dataframe containing X data

                y: (pd.Series), series containing y data

                savepath: (str), string containing the save path directory

            Returns:
                None
    """
    @classmethod
    def flag_outliers(cls, X, y, savepath, n_stdevs=3):
        df = pd.concat([X, y], axis=1)
        n_rows = df.shape[0]
        outlier_dict = dict()
        outlier_rows_all = list()
        for col in df.columns:
            outlier_rows = list()
            outlier_vals = list()
            avg = np.average(df[col])
            stdev = np.std(df[col])
            for row in range(n_rows):
                if df[col].iloc[row] > avg + n_stdevs*stdev:
                    outlier_rows.append(row)
                    outlier_vals.append(df[col].iloc[row])
                elif df[col].iloc[row] < avg - n_stdevs*stdev:
                    outlier_rows.append(row)
                    outlier_vals.append(df[col].iloc[row])
                else:
                    pass
            outlier_dict[col] = (outlier_rows, outlier_vals)
            outlier_rows_all.append(outlier_rows)

        # Save data to file
        pd.DataFrame().from_dict(data=outlier_dict, orient='index',
                                 columns=['Indices', 'Values']).to_excel(os.path.join(savepath, 'data_outliers_all.xlsx'))

        # Also get values of rows that occur most often
        outlier_rows_all = np.concatenate(outlier_rows_all).ravel()
        outlier_counts = Counter(outlier_rows_all)
        # Save summary data of outlier counts to file
        pd.DataFrame().from_dict(data=outlier_counts, orient='index',
                                 columns=['Number of occurrences']).to_excel(os.path.join(savepath, 'data_outliers_summary.xlsx'))

        return

    @classmethod
    def flag_columns_with_strings(cls, X, y, savepath):
        df = pd.concat([X, y], axis=1)
        str_summary = pd.DataFrame(df.applymap(type).eq(str).any())
        str_columns = str_summary.index[str_summary[0] == True].tolist()
        d = {'columns with strings': str_columns}
        pd.DataFrame().from_dict(data=d).to_excel(os.path.join(savepath, 'data_columns_with_strings.xlsx'))
        return


class PPCA():
    """
    Class to perform probabilistic principal component analysis (PPCA) to fill in missing data.

    This PPCA routine was taken directly from https://github.com/allentran/pca-magic. Due to import errors, for ease of use
    we have elected to copy the module here. This github repo was last accessed on 8/27/18. The code comprising the PPCA
    class below was not developed by and is not owned by the University of Wisconsin-Madison MAST-ML development team.

    """

    def __init__(self):

        self.raw = None
        self.data = None
        self.C = None
        self.means = None
        self.stds = None
        self.eig_vals = None

    def _standardize(self, X):

        if self.means is None or self.stds is None:
            raise RuntimeError("Fit model first")

        return (X - self.means) / self.stds

    def fit(self, data, d=None, tol=1e-4, min_obs=10, verbose=False):

        self.raw = data
        self.raw[np.isinf(self.raw)] = np.max(self.raw[np.isfinite(self.raw)])

        valid_series = np.sum(~np.isnan(self.raw), axis=0) >= min_obs

        data = self.raw[:, valid_series].copy()
        N = data.shape[0]
        D = data.shape[1]

        self.means = np.nanmean(data, axis=0)
        self.stds = np.nanstd(data, axis=0)

        data = self._standardize(data)
        observed = ~np.isnan(data)
        missing = np.sum(~observed)
        data[~observed] = 0

        # initial

        if d is None:
            d = data.shape[1]

        if self.C is None:
            C = np.random.randn(D, d)
        else:
            C = self.C
        CC = np.dot(C.T, C)
        X = np.dot(np.dot(data, C), np.linalg.inv(CC))
        recon = np.dot(X, C.T)
        recon[~observed] = 0
        ss = np.sum((recon - data) ** 2) / (N * D - missing)

        v0 = np.inf
        counter = 0

        while True:

            Sx = np.linalg.inv(np.eye(d) + CC / ss)

            # e-step
            ss0 = ss
            if missing > 0:
                proj = np.dot(X, C.T)
                data[~observed] = proj[~observed]
            X = np.dot(np.dot(data, C), Sx) / ss

            # m-step
            XX = np.dot(X.T, X)
            C = np.dot(np.dot(data.T, X), np.linalg.pinv(XX + N * Sx))
            CC = np.dot(C.T, C)
            recon = np.dot(X, C.T)
            recon[~observed] = 0
            ss = (np.sum((recon - data) ** 2) + N * np.sum(CC * Sx) + missing * ss0) / (N * D)

            # calc diff for convergence
            det = np.log(np.linalg.det(Sx))
            if np.isinf(det):
                det = abs(np.linalg.slogdet(Sx)[1])
            v1 = N * (D * np.log(ss) + np.trace(Sx) - det) \
                + np.trace(XX) - missing * np.log(ss0)
            diff = abs(v1 / v0 - 1)
            if verbose:
                print(diff)
            if (diff < tol) and (counter > 5):
                break

            counter += 1
            v0 = v1

        C = orth(C)
        vals, vecs = np.linalg.eig(np.cov(np.dot(data, C).T))
        order = np.flipud(np.argsort(vals))
        vecs = vecs[:, order]
        vals = vals[order]

        C = np.dot(C, vecs)

        # attach objects to class
        self.C = C
        self.data = data
        self.eig_vals = vals
        self._calc_var()

    def transform(self, data=None):

        if self.C is None:
            raise RuntimeError('Fit the data model first.')
        if data is None:
            return np.dot(self.data, self.C)
        return np.dot(data, self.C)

    def _calc_var(self):

        if self.data is None:
            raise RuntimeError('Fit the data model first.')

        data = self.data.T

        # variance calc
        var = np.nanvar(data, axis=1)
        total_var = var.sum()
        self.var_exp = self.eig_vals.cumsum() / total_var

    def save(self, fpath):

        np.save(fpath, self.C)

    def load(self, fpath):

        assert os.path.isfile(fpath)

        self.C = np.load(fpath)
