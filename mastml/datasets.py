"""
This module provides various methods for importing data into MAST-ML.

SklearnDatasets:
    Enables easy import of model datasets from scikit-learn, such as boston housing data, friedman, etc.

LocalDatasets:
    Main method for importing datasets that are stored in an accessible path. Main file format is Excel
    spreadsheet (.xls or .xlsx). This method also makes it easy for separately denoting other data features
    that are not directly the X or y data, such as features used for grouping, extra features no used in
    fitting, or features that denote manually held-out test data

FigshareDatasets:
    Method to download data that is stored on Figshare, an open-source data hosting service. This class
    can be used to download data, then subsquently the LocalDatasets class can be used to import the data.

FoundryDatasets:
    Method to download data this stored on the Materials Data Facility (MDF) Foundry data hosting service.
    This class can be used to download data, then subsquently the LocalDatasets class can be used to import
    the data.

MatminerDatasets:
    Method to download data this stored as part of the matminer machine learning package
    (https://github.com/hackingmaterials/matminer). This class can be used to download data, then
    subsquently the LocalDatasets class can be used to import the data.

"""

import pandas as pd
import os
import numpy as np
from pprint import pprint
import shutil
import pickle

import sklearn.datasets
from mdf_forge import Forge

from matminer.datasets.dataset_retrieval import load_dataset, get_available_datasets

try:
    from figshare.figshare.figshare import Figshare
except:
    print('Figshare is an optional dependency. To import data from figshare, manually install figshare via git clone of '
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

        load_friedman: Loads the Friedman data set (regression)

    """

    def __init__(self, return_X_y=True, as_frame=False):
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

    def load_digits(self, n_class=10):
        return sklearn.datasets.load_digits(return_X_y=self.return_X_y, as_frame=self.as_frame, n_class=n_class)

    def load_linnerud(self):
        return sklearn.datasets.load_linnerud(return_X_y=self.return_X_y, as_frame=self.as_frame)

    def load_wine(self):
        return sklearn.datasets.load_wine(return_X_y=self.return_X_y, as_frame=self.as_frame)

    def load_breast_cancer(self):
        return sklearn.datasets.load_breast_cancer(return_X_y=self.return_X_y, as_frame=self.as_frame)

    def load_friedman(self, n_samples=100, n_features=10, noise=0.0):
        X, y = sklearn.datasets.make_friedman1(n_samples=n_samples, n_features=n_features, noise=noise)
        if self.as_frame:
            return pd.DataFrame(X, columns=["x"+str(i) for i in range(n_features)]), pd.DataFrame(y, columns=['target'])
        else:
            return X, y


class LocalDatasets():
    """
    Class to handle import and organization of a dataset stored locally.

    Args:
        file_path: (str), path to the data file to import

        feature_names: (list), list of strings containing the X feature names

        target: (str), string denoting the y data (target) name

        extra_columns: (list), list of strings containing additional column names that are not features or target

        group_column: (str), string denoting the name of an input column to be used to group data

        testdata_columns: (list), list of strings containing column names denoting sets of left-out data. Entries should be marked with a 0 (not left out) or 1 (left out)

        average_duplicates: (bool), whether to average duplicate entries from the imported data.

        average_duplicates_col: (str), string denoting column name to perform averaging of duplicate entries. Needs to be specified if average_duplicates is True.

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

        load_data: Method to import the data and ascertain which columns are features, target and extra based on provided input.
            Args:
                copy: (bool), whether or not to copy the imported data to the designated savepath

                savepath: (str), path to save the data to (used if copy=True)

            Returns:
                data_dict: (dict), dictionary containing dataframes of X, y, groups, X_extra, X_testdata

    """

    def __init__(self, file_path, feature_names=None, target=None, extra_columns=None, group_column=None,
                 testdata_columns=None, average_duplicates=False, average_duplicates_col=None, as_frame=False):
        self.file_path = file_path
        self.feature_names = feature_names
        self.target = target
        self.extra_columns = extra_columns
        self.group_column = group_column
        self.testdata_columns = testdata_columns
        self.average_duplicates = average_duplicates
        self.average_duplicates_col = average_duplicates_col
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

    def load_data(self, copy=False, savepath=None):
        if copy == True:
            if savepath is not None:
                fname = os.path.normpath(self.file_path).split(os.path.sep)[-1]
                shutil.copy(self.file_path, os.path.join(savepath, fname))

        data_dict = dict()

        # Import data from file
        df = self._import()

        # Average duplicate entries, if specified
        if self.average_duplicates == True:
            if self.average_duplicates_col is not None:
                df = df.groupby(df[self.average_duplicates_col]).mean().reset_index()
            else:
                print('Error: you need to specify average_duplicates_col if average_duplicates is True')

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
                article_id: (int), the number denoting the Figshare article ID. Can be obtained from the URL to the Figshare dataset

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
