import sklearn.datasets
import pandas as pd
import os
import numpy as np
from mdf_forge import Forge
try:
    from figshare.figshare.figshare import Figshare
except:
    print('To import data from figshare, manually install figshare via git clone of '
          'git clone https://github.com/cognoma/figshare.git')
from pprint import pprint

from sklearn.impute import SimpleImputer

from scipy.linalg import orth

from collections import Counter
import shutil

from mastml.plots import Histogram

from datetime import datetime

class SklearnDatasets():
    '''

    '''
    def __init__(self, return_X_y=False, as_frame=False, n_class=None):
        self.return_X_y = return_X_y
        self.as_frame = as_frame
        self.n_class = n_class

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

    def load_digits(self):
        if self.n_class:
            return sklearn.datasets.load_digits(self.n_class, self.return_X_y, self.as_frame)
        return sklearn.datasets.load_digits(n_class=10, return_X_y=self.return_X_y, as_frame=self.as_frame)

    def load_linnerud(self):
        return sklearn.datasets.load_linnerud(return_X_y=self.return_X_y, as_frame=self.as_frame)

    def load_wine(self):
        return sklearn.datasets.load_wine(return_X_y=self.return_X_y, as_frame=self.as_frame)

    def load_breast_cancer(self):
        return sklearn.datasets.load_breast_cancer(return_X_y=self.return_X_y, as_frame=self.as_frame)

class LocalDatasets():
    '''

    '''
    def __init__(self, file_path, feature_names=None, target=None, extra_columns=None, as_frame=False):
        self.file_path = file_path
        self.feature_names = feature_names
        self.target = target
        self.extra_columns = extra_columns
        self.as_frame = as_frame
        if self.extra_columns is None:
            self.extra_columns = list()

    def _import(self):
        fname, ext = os.path.splitext(self.file_path)
        if ext == '.csv':
            df = pd.read_csv(self.file_path)
        elif ext == '.xlsx':
            df = pd.read_excel(self.file_path)
        else:
            raise ValueError('file_path must be .csv or .xlsx for data local data import')
        return df

    def _get_features(self, df):
        if self.feature_names is None and self.target is None:
            print('WARNING: feature_names and target are not specified. Assuming last column is target value and remaining columns are features')
            self.target = df.columns[-1]
            self.feature_names = [col for col in df.columns if col not in [self.extra_columns, self.target]]
        elif self.feature_names is None:  # input is all the features except the target feature
            print('WARNING: feature_names not specified but target was specified. Assuming all columns except target and extra columns are features')
            cols = [col for col in df.columns if col != self.target]
            self.feature_names = [col for col in cols if col not in self.extra_columns]
        elif self.target is None:  # target is the last non-input feature
            for col in df.columns[::-1]:
                if col not in self.feature_names:
                    target = col
                    break
        return

    def load_data(self):
        # Import data from file
        df = self._import()

        # Assign default values to input_features and target_feature
        self._get_features(df=df)

        X, y = df[self.feature_names], pd.DataFrame(df[self.target], columns=[self.target]).squeeze()

        if self.as_frame:
            return X, y
        return np.array(X), np.array(y).ravel()

class FigshareDatasets():
    '''
    To install:
    git clone https://github.com/cognoma/figshare.git

    diffusion data article id: 7418492

    '''
    def __init__(self):
        pass

    def download_data(self, article_id, savepath=None):
        fs = Figshare()
        fs.retrieve_files_from_article(article_id)
        if savepath:
            shutil.move(os.path.join(os.getcwd(), 'figshare_'+str(article_id)), savepath)
        return

class FoundryDatasets():
    '''

    '''
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

class DataCleaning():
    '''


    '''
    def __init__(self):
        pass

    def remove(self, X, y, axis):
        """
        Method that removes a full column or row of data values if one column or row contains NaN or is blank

        Args:
            df: (dataframe), pandas dataframe containing data
            axis: (int), whether to remove rows (axis=0) or columns (axis=1)

        Returns:
            df: (dataframe): dataframe with NaN or missing values removed

        """
        df = pd.concat([X, y], axis=1)
        target = y.name
        df = df.dropna(axis=axis, how='any')
        y = df[target]
        X = df[[col for col in df.columns if col != target]]
        return X, y

    def imputation(self, X, y, strategy):
        """
        Method that imputes values to the missing places based on the median, mean, etc. of the data in the column

        Args:
            df: (dataframe), pandas dataframe containing data
            strategy: (str), method of imputation, e.g. median, mean, etc.
            cols_to_leave_out: (list), list of column indices to not include in imputation

        Returns:
            df: (dataframe): dataframe with NaN or missing values resolved via imputation

        """
        df = pd.concat([X, y], axis=1)
        columns = df.columns.tolist()
        df = pd.DataFrame(SimpleImputer(missing_values=np.nan, strategy=strategy).fit_transform(df), columns=columns)
        target = y.name
        y = df[target]
        X = df[[col for col in df.columns if col != target]]
        return X, y

    def ppca(self, X, y):
        """
        Method that performs a recursive PCA routine to use PCA of known columns to fill in missing values in particular column

        Args:
            df: (dataframe), pandas dataframe containing data
            cols_to_leave_out: (list), list of column indices to not include in imputation

        Returns:
            df: (dataframe): dataframe with NaN or missing values resolved via imputation

        """
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
        '''
        cleaner : 'remove', 'imputation', 'ppca'
        '''
        if not savepath:
            savepath = os.getcwd()
        splitdir = self._setup_savedir(savepath=savepath)
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
    '''


    '''

    @classmethod
    def flag_outliers(cls, X, y, savepath, n_stdevs=3):
        """
        Method that scans values in each X feature matrix column and flags values that are larger than 3 standard deviations
        from the average of that column value. The index and column values of potentially problematic points are listed and
        written to an output file.

        Args:
            df: (dataframe), pandas dataframe containing data

        Returns:
            None, just writes results to file

        """
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
        """
        Method that ascertains which columns in data contain string entries

        Args:
            df: (dataframe), pandas dataframe containing data

        Returns:
            str_columns: (list), list containing indices of columns containing strings

        """
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

