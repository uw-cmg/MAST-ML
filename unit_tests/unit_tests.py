import unittest
import pandas as pd
import numpy as np
import os
import sys
testdir = os.path.realpath(os.path.dirname(sys.argv[0]))
moduledir = '/Users/ryanjacobs/PycharmProjects/MASTML/'
print(testdir)
print(moduledir)
sys.path.append(moduledir)
from DataOperations import DataframeUtilities
from MASTMLInitializer import ConfigFileParser

class TestDataOperations(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.read_csv(testdir+'/'+'testcsv1.csv')
        self.df2 = pd.read_csv(testdir+'/'+'testcsv2.csv')
        self.arr1 = np.array(self.df1)
        self.arr2 = np.array(self.df2)
        return

    def test_merge_dataframe_columns(self):
        df = DataframeUtilities().merge_dataframe_columns(dataframe1=self.df1, dataframe2=self.df2)
        self.assertFalse(df.shape == self.df1.shape)
        self.assertIsInstance(df, pd.DataFrame)
        return

    def test_merge_dataframe_rows(self):
        df = DataframeUtilities().merge_dataframe_rows(dataframe1=self.df1, dataframe2=self.df2)
        self.assertFalse(df.shape == self.df1.shape)
        self.assertIsInstance(df, pd.DataFrame)
        return

    def test_get_dataframe_statistics(self):
        df = DataframeUtilities().get_dataframe_statistics(dataframe=self.df1)
        self.assertIsInstance(df, pd.DataFrame)
        return

    def test_dataframe_to_array(self):
        arr = DataframeUtilities().dataframe_to_array(dataframe=self.df1)
        self.assertNotIsInstance(arr, pd.DataFrame)
        self.assertIsInstance(arr, np.ndarray)
        return

    def test_array_to_dataframe(self):
        df1 = DataframeUtilities().array_to_dataframe(array=self.arr1)
        self.assertNotIsInstance(df1, np.ndarray)
        self.assertIsInstance(df1, pd.DataFrame)
        return

    def test_concatenate_arrays(self):
        arr = DataframeUtilities().concatenate_arrays(X_array=self.arr1, y_array=self.arr2)
        self.assertIsInstance(arr, np.ndarray)
        self.assertFalse(arr.shape == self.arr1.shape)
        return

    def test_assign_columns_as_features(self):
        df = DataframeUtilities().assign_columns_as_features(dataframe=self.df1, x_features=["Material compositions"],
                                                             y_feature="O_pband_center", remove_first_row=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.shape == self.df1.shape)
        return

    def test_save_all_dataframe_statistics(self):
        DataframeUtilities().save_all_dataframe_statistics(dataframe=self.df1, data_path=testdir+'/'+'testcsv1.csv',
                                                           configfile_name='test.conf', configfile_path=testdir)
        return

    def test_plot_dataframe_histogram(self):
        configdict = ConfigFileParser(configfile='test.conf').get_config_dict(path_to_file=testdir)
        DataframeUtilities().plot_dataframe_histogram(dataframe=self.df1, configdict=configdict, y_feature="O_pband_center")
        return

if __name__ == '__main__':
    unittest.main()