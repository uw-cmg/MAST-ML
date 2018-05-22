import unittest
import pandas as pd
import numpy as np
import os
import sys

testdir = os.path.realpath(os.path.dirname(sys.argv[0]))
moduledir = '../'

sys.path.append(moduledir)
from DataOperations import DataframeUtilities, DataParser
from MASTMLInitializer import ConfigFileParser

class TestDataframeUtilities(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.read_csv(testdir+'/'+'testcsv1.csv')
        self.df2 = pd.read_csv(testdir+'/'+'testcsv2.csv')
        self.arr1 = np.array(self.df1)
        self.arr2 = np.array(self.df2)
        self.configfile = 'test_unittest_dataoperations.conf'
        self.configdict = ConfigFileParser(configfile=self.configfile).get_config_dict(path_to_file=testdir)
        self.files = list()
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
        fname = DataframeUtilities().save_all_dataframe_statistics(dataframe=self.df1, configdict=self.configdict)
        self.files.append(fname)
        return

    def test_plot_dataframe_histogram(self):
        configdict = ConfigFileParser(configfile='test_unittest_dataoperations.conf').get_config_dict(path_to_file=testdir)
        fname = DataframeUtilities().plot_dataframe_histogram(dataframe=self.df1, configdict=configdict, y_feature="O_pband_center_regression")
        self.files.append(fname)
        return

    def tearDown(self):
        for file in self.files:
            os.remove(file)
        return

class TestDataParser(unittest.TestCase):

    def setUp(self):
        self.configdict = ConfigFileParser(configfile='test_unittest_dataoperations.conf').get_config_dict(path_to_file=testdir)
        self.datapath = testdir+'/'+'testcsv1.csv'
        self.df1 = pd.read_csv(testdir + '/' + 'testcsv1.csv')
        self.target_feature = "O_pband_center_regression"
        self.x_features = [f for f in self.df1.columns.values.tolist() if f != self.target_feature]
        return

    def test_parse_fromfile(self):
        Xdata, ydata, x_features, y_feature, dataframe = DataParser(configdict=self.configdict).parse_fromfile(
            datapath=self.datapath, as_array=False)
        self.assertIsInstance(Xdata, pd.DataFrame)
        self.assertIsInstance(ydata, pd.Series)
        self.assertIsInstance(dataframe, pd.DataFrame)
        Xdata, ydata, x_features, y_feature, dataframe = DataParser(configdict=self.configdict).parse_fromfile(
            datapath=self.datapath, as_array=True)
        self.assertIsInstance(Xdata, np.ndarray)
        self.assertIsInstance(ydata, np.ndarray)
        self.assertIsInstance(dataframe, pd.DataFrame)
        return

    def test_parse_fromdataframe(self):
        Xdata, ydata, x_features, y_feature, dataframe = DataParser(configdict=self.configdict).parse_fromdataframe(
            dataframe=self.df1, target_feature=self.target_feature, as_array=False)
        self.assertIsInstance(Xdata, pd.DataFrame)
        self.assertIsInstance(ydata, pd.Series)
        self.assertIsInstance(dataframe, pd.DataFrame)
        Xdata, ydata, x_features, y_feature, dataframe = DataParser(configdict=self.configdict).parse_fromdataframe(
            dataframe=self.df1, target_feature=self.target_feature, as_array=True)
        self.assertIsInstance(Xdata, np.ndarray)
        self.assertIsInstance(ydata, np.ndarray)
        self.assertIsInstance(dataframe, pd.DataFrame)
        return

    def test_import_data(self):
        dataframe = DataParser(configdict=self.configdict).import_data(datapath=self.datapath)
        self.assertIsInstance(dataframe, pd.DataFrame)
        return

    def test_get_features(self):
        x_features, y_feature = DataParser(configdict=self.configdict).get_features(dataframe=self.df1,
                                                                                    target_feature=self.target_feature,
                                                                                    from_input_file=False)
        self.assertTrue(type(x_features) is list)
        self.assertTrue(y_feature == self.target_feature)
        return

    def test_get_data(self):
        Xdata, ydata = DataParser(configdict=self.configdict).get_data(dataframe=self.df1, x_features=self.x_features,
                                                                       y_feature=self.target_feature)
        return

    def tearDown(self):
        return

if __name__ == '__main__':
    unittest.main()
