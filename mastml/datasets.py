
import pandas as pd
import os
import numpy as np
from pprint import pprint
from collections import Counter
import shutil
from datetime import datetime
import pickle

from sklearn.impute import SimpleImputer
from scipy.linalg import orth
import sklearn.datasets
from mdf_forge import Forge

from matminer.datasets.dataset_retrieval import load_dataset, get_available_datasets

from mastml.plots import Histogram

try:
    from figshare.figshare.figshare import Figshare
except:
    print('To import data from figshare, manually install figshare via git clone of '
          'git clone https://github.com/cognoma/figshare.git')


class SklearnDatasets():
    """
    Class wrapping the sklearn.datasets funcionality for easy import of toy datasets from sklearn. Added some changes
    to make all datasets operate more consistently, e.g. boston housing data

    Args:

        return_X_y: (bool), whether to return X, y data as (X, y) tuple (should be true for easiest use in MASTML)
        as_frame: (bool), whether to return X, y data as pandas dataframe objects
        n_class: (int), number of classes (only applies to load_digits method)

    Methods:

        load_boston: Loads the Boston housing data (regression)

        load_iris: Loads the flower iris data (classification)

        load_diabetes: Loads the diabetes data set (regression)

        load_digits: Loads the MNIST digits data set (classification)

        load_linnerud: Loads the linnerud data set (regression)

        load_wine: Loads the wine data set (classification)

        load_breast_cancer: Loads the breast cancer data set (classification)

    """
<<<<<<< HEAD

    def __init__(self, return_X_y=True, as_frame=False, n_class=None):
=======
    def __init__(self, return_X_y=True, as_frame=False):
>>>>>>> origin/dev_Ryan_2020-12-21
        self.return_X_y = return_X_y
        self.as_frame = as_frame

    def load_boston(self):
        if self.as_frame:
            boston = sklearn.datasets.load_boston()
            X = pd.DataFrame(boston.data, columns=boston.feature_names)
            y = pd.DataFrame(boston.target, columns=['MEDV'])
            return X, y
        return sklearn.datasets.load_boston(return_X_y=self.return_X_y)

    def load_iris(self):
        return sklearn.datasets.load_iris(return_X_y=self.return_X_y, as_frame=self.as_frame)

    def load_diabetes(self):
        return sklearn.datasets.load_diabetes(return_X_y=self.return_X_y, as_frame=self.as_frame)

    def load_digits(self, n_class):
        return sklearn.datasets.load_digits(self.return_X_y, self.as_frame)

    def load_linnerud(self):
        return sklearn.datasets.load_linnerud(return_X_y=self.return_X_y, as_frame=self.as_frame)

    def load_wine(self):
        return sklearn.datasets.load_wine(return_X_y=self.return_X_y, as_frame=self.as_frame)

    def load_breast_cancer(self):
        return sklearn.datasets.load_breast_cancer(return_X_y=self.return_X_y, as_frame=self.as_frame)

<<<<<<< HEAD
=======
    def load_friedman(self, n_samples=100, n_features=10, noise=0.0):
        X, y = sklearn.datasets.make_friedman1(n_samples=n_samples, n_features=n_features, noise=noise)
        if self.as_frame:
            return pd.DataFrame(X, columns=["x"+str(i) for i in range(n_features)]), pd.DataFrame(y, columns=['target'])
        else:
            return X, y
>>>>>>> origin/dev_Ryan_2020-12-21

class LocalDatasets():
    """
    Class to handle import and organization of a dataset stored locally.

    Args:
        file_path: (str), path to the data file to import
        feature_names: (list), list of strings containing the X feature names
        target: (str), string denoting the y data (target) name
        extra_columns: (list), list of strings containing additional column names that are not features or target
        group_column: (str), string denoting the name of an input column to be used to group data
        testdata_columns: (list), list of strings containing column names denoting sets of left-out data. Entries should
            be marked with a 0 (not left out) or 1 (left out)
        as_frame: (bool), whether to return data as pandas dataframe (otherwise will be numpy array)

    Methods:
        _import: imports the data. Should be either .csv or .xlsx format

            Args:
                None

            Returns:
                df: (pd.DataFrame), pandas dataframe of full dataset

        _get_features: Method to assess which columns below to target, feature_names

            Args:
                df: (pd.DataFrame), pandas dataframe of full dataset

            Returns:
                None

        load_data: Method to import the data and ascertain which columns are features, target and extra based on provided
            input.

            Args:
                None

            Returns:
                X: (pd.DataFrame or numpy array), dataframe or array of X data
                y: (pd.DataFrame or numpy array), dataframe or array of y data
                groups: (pd.Series or numpy array), Series or array denoting user-specified groups of data
                X_extra: (pd.DataFrame or numpy array), dataframe or array of extra data not used in fitting (e.g. references, compositions, other notes)
                X_testdata: (list), list of arrays denoting indices of left-out data sets
    """
<<<<<<< HEAD

    def __init__(self, file_path, feature_names=None, target=None, extra_columns=None, as_frame=False):
=======
    def __init__(self, file_path, feature_names=None, target=None, extra_columns=None, group_column=None,
                 testdata_columns=None, as_frame=False):
>>>>>>> origin/dev_Ryan_2020-12-21
        self.file_path = file_path
        self.feature_names = feature_names
        self.target = target
        self.extra_columns = extra_columns
        self.group_column = group_column
        self.testdata_columns = testdata_columns
        self.as_frame = as_frame
        if self.extra_columns is None:
            self.extra_columns = list()
        if self.testdata_columns is None:
            self.testdata_columns = list()

    def _import(self):
        fname, ext = os.path.splitext(self.file_path)
        if ext == '.csv':
            df = pd.read_csv(self.file_path)
        elif ext == '.xlsx':
            try:
                df = pd.read_excel(self.file_path)
            except:
                df = pd.read_excel(self.file_path, engine='openpyxl')
        elif ext == '.pickle':
            with open(self.file_path, "rb") as input_file:
                data = pickle.load(input_file)
                df = pd.DataFrame(data)
        else:
            raise ValueError('file_path must be .csv, .xlsx or .pickle for data local data import')
        return df

    def _get_features(self, df):
        if self.feature_names is None and self.target is None:
            print('WARNING: feature_names and target are not specified. Assuming last column is target value and remaining columns are features')
            self.target = df.columns[-1]
            self.feature_names = [col for col in df.columns if col not in [self.extra_columns, self.target, self.testdata_columns]]
        elif self.feature_names is None:  # input is all the features except the target feature
            print('WARNING: feature_names not specified but target was specified. Assuming all columns except target and extra columns are features')
            cols = [col for col in df.columns if col != self.target]
            self.feature_names = [col for col in cols if col not in self.extra_columns and col not in self.testdata_columns]
        elif self.target is None:  # target is the last non-input feature
            for col in df.columns[::-1]:
                if col not in self.feature_names:
                    target = col
                    break
        return

    def load_data(self):
        data_dict = dict()

        # Import data from file
        df = self._import()

        # Assign default values to input_features and target_feature
        self._get_features(df=df)

        X, y = df[self.feature_names], pd.DataFrame(df[self.target], columns=[self.target]).squeeze()

        data_dict['X'] = X
        data_dict['y'] = y

        if self.group_column:
            groups = df[self.group_column]
            data_dict['groups'] = groups
        else:
            data_dict['groups'] = None

        if self.extra_columns:
            X_extra = df[self.extra_columns]
            data_dict['X_extra'] = X_extra
        else:
            data_dict['X_extra'] = None

        if self.testdata_columns:
            X_testdata = list()
            for col in self.testdata_columns:
                X_testdata.append(np.array(df.loc[df[col] == 1].index).ravel())
            data_dict['X_testdata'] = X_testdata
        else:
            data_dict['X_testdata'] = None

        if self.as_frame == True:
            return data_dict
        else:
            for k, v in data_dict.items():
                data_dict[k] = np.array(v)
            return data_dict


class FigshareDatasets():
    """
    Class to download datasets hosted on Figshare. To install: git clone https://github.com/cognoma/figshare.git

    Args:
        None

    Methods:

        download_data: downloads specified data from Figshare and saves to current directory

            Args:

                article_id: (int), the number denoting the Figshare article ID. Can be obtained from the URL to the
                Figshare dataset

                savepath: (str), string denoting the savepath of the MAST-ML run

            Returns:
                None
    """

    def __init__(self):
        pass

    def download_data(self, article_id, savepath=None):
        fs = Figshare()
        fs.retrieve_files_from_article(article_id)
        if savepath:
            try:
                shutil.move(os.path.join(os.getcwd(), 'figshare_'+str(article_id)), savepath)
            except shutil.Error:
                print('Warning: could not move downloaded data to specified savepath, maybe because savepath is the current working directory')
        return


class FoundryDatasets():
    """
    Class to download datasets hosted on Materials Data Facility

    Args:
        no_local_server: (bool), whether or not the server is local. Set to True if running on e.g. Google Colab
        anonymous: (bool), whether to use your MDF user or be anonymous. Some functionality may be disabled if True
        test: (bool), whether to be in test mode. Some functionality may be disabled if True

    Methods:

        download_data: downloads specified data from MDF and saves to current directory

            Args:
                name: (str), name of the dataset to download
                doi: (str), digital object identifier of the dataset to download
                download: (bool), whether or not to download the full dataset

            Returns:
                None
    """

    def __init__(self, no_local_server, anonymous, test):
        self.no_local_server = no_local_server
        self.anonymous = anonymous
        self.test = test
        self.mdf = Forge(no_local_server=self.no_local_server,
                         anonymous=self.anonymous,
                         test=self.test)

    def download_data(self, name=None, doi=None, download=False):
        if name is not None:
            self.mdf.match_source_names(name)
        elif doi is not None:
            self.mdf.match_dois(doi)
        else:
            print('ERROR: please specify either the dataset name or DOI for lookup MDF')
        result = self.mdf.search()
        if len(result) == 1:
            print('Successfully found the desired dataset on MDF')
            print('MDF entry:')
            pprint(result)
            if download == True:
                print('Downloading dataset from MDF')
                self.mdf.globus_download(results=result)
        return

<<<<<<< HEAD
=======
class MatminerDatasets():
    """
    Class to download datasets hosted from the Matminer package's Figshare page. A summary of available datasets
    can be found at: https://hackingmaterials.lbl.gov/matminer/dataset_summary.html

    Args:
        None

    Methods:

        download_data: downloads specified data from Matminer/Figshare and saves to current directory

            Args:

                name: (str), name of the dataset to download. For compatible names, call get_available_datasets

                save_data: (bool), whether to save the downloaded data to the current working directory

            Returns:
                df: (dataframe), dataframe of downloaded data

        get_available_datasets: returns information on the available dataset names and details one can downlaod

            Args:
                None.

            Returns:
                None.
    """
    def __init__(self):
        pass

    def download_data(self, name, save_data=True):
        df = load_dataset(name=name)
        if save_data == True:
            df.to_excel(name+'.xlsx', index=False)
            with open('%s.pickle' % name, 'wb') as data_file:
                pickle.dump(df, data_file)
        return df

    def get_available_datasets(self):
        datasets = get_available_datasets()
        return
>>>>>>> origin/dev_Ryan_2020-12-21

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

        evaluate: main method to evaluate initial data analysis routines (e.g. flag outliers), perform data cleaning and
            save output to folder.

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
        target = y.name
        df = df.dropna(axis=axis, how='any')
        y = df[target]
        X = df[[col for col in df.columns if col != target]]
        return X, y

    def imputation(self, X, y, strategy):
        df = pd.concat([X, y], axis=1)
        columns = df.columns.tolist()
        df = pd.DataFrame(SimpleImputer(missing_values=np.nan, strategy=strategy).fit_transform(df), columns=columns)
        target = y.name
        y = df[target]
        X = df[[col for col in df.columns if col != target]]
        return X, y

    def ppca(self, X, y):
        df = pd.concat([X, y], axis=1)
        target = y.name
        columns = df.columns.tolist()
        pca_magic = PPCA()
        pca_magic.fit(np.array(df))

        # Need to un-standardize the pca-transformed data
        df = pd.DataFrame(pca_magic.data*pca_magic.stds+pca_magic.means, columns=columns)

        y = df[target]
        X = df[[col for col in columns if col != target]]
        return X, y

    def evaluate(self, X, y, method, savepath=None, **kwargs):
        if not savepath:
            savepath = os.getcwd()
        splitdir = self._setup_savedir(savepath=savepath)
        self.splitdir = splitdir
        DataUtilities().flag_columns_with_strings(X=X, y=y, savepath=splitdir)
        DataUtilities().flag_outliers(X=X, y=y, savepath=splitdir, n_stdevs=3)
        df_orig = pd.concat([X, y], axis=1)
        self.cleaner = getattr(self, method)
        X, y = self.cleaner(X, y, **kwargs)
        df_cleaned = pd.concat([X, y], axis=1)
        df_orig.to_excel(os.path.join(splitdir, 'data_original.xlsx'))
        df_cleaned.to_excel(os.path.join(splitdir, 'data_cleaned.xlsx'))

        # Make histogram of the input data
        Histogram.plot_histogram(df=y, file_name='histogram_target_values', savepath=splitdir, x_label='Target values')

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

        flag_outliers: Method that scans values in each X feature matrix column and flags values that are larger than
            X standard deviations from the average of that column value. The index and column values of potentially
            problematic points are listed and written to an output file.

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
