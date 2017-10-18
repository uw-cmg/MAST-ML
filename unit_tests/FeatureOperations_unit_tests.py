import unittest
import pandas as pd
import os
import sys
import numpy as np
testdir = os.path.realpath(os.path.dirname(sys.argv[0]))
moduledir = '/Users/ryanjacobs/PycharmProjects/MASTML/'
sys.path.append(moduledir)
from MASTMLInitializer import ConfigFileParser
from FeatureOperations import FeatureNormalization, FeatureIO, MiscFeatureOperations

class TestMiscFeatureOperations(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.read_csv(testdir+'/'+'testcsv1string.csv')
        self.configdict = ConfigFileParser(configfile='test_unittest_featuregeneration.conf').get_config_dict(path_to_file=testdir)
        self.target_feature = "O_pband_center_regression"
        self.x_features = [f for f in self.df1.columns.values.tolist()]
        return

    def test_remove_features_containing_strings(self):
        df1columns = self.df1.shape[1]
        x_features_pruned, df = MiscFeatureOperations(configdict=self.configdict).remove_features_containing_strings(dataframe=self.df1,
                                                                                                  x_features=self.x_features)
        self.assertTrue(df.shape[1] < df1columns)
        return

class TestFeatureIO(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.read_csv(testdir + '/' + 'testcsv1.csv')
        self.x_features = [f for f in self.df1.columns.values.tolist()]
        return

    def test_remove_duplicate_columns(self):
        # Manually add duplicate column to dataframe
        self.df1['AtomicNumber_composition_average_copy'] = pd.Series(self.df1['AtomicNumber_composition_average'], index=self.df1.index)
        self.df1.rename(columns={'AtomicNumber_composition_average_copy': 'AtomicNumber_composition_average'}, inplace=True)
        df = FeatureIO(dataframe=self.df1).remove_duplicate_columns()
        self.assertTrue(df.shape[1] < self.df1.shape[1])
        return

    def test_remove_duplicate_features_by_values(self):
        # Manually add duplicate column to dataframe
        self.df1['AtomicNumber_composition_average_copy'] = pd.Series(self.df1['AtomicNumber_composition_average'], index=self.df1.index)
        self.df1.rename(columns={'AtomicNumber_composition_average_copy': 'AtomicNumber_composition_average'}, inplace=True)
        df = FeatureIO(dataframe=self.df1).remove_duplicate_features_by_values()
        self.assertTrue(df.shape[1] < self.df1.shape[1])
        return

    def test_remove_custom_features(self):
        df = FeatureIO(dataframe=self.df1).remove_custom_features(features_to_remove=['AtomicNumber_composition_average'])
        self.assertTrue('AtomicNumber_composition_average' not in df.columns.values.tolist())
        return

    def test_keep_custom_features(self):
        df = FeatureIO(dataframe=self.df1).keep_custom_features(features_to_keep=['AtomicNumber_composition_average'])
        self.assertTrue(['AtomicNumber_composition_average'] == df.columns.values.tolist())
        return

    def test_add_custom_features(self):
        df1columns = self.df1.shape[1]
        df = FeatureIO(dataframe=self.df1).add_custom_features(features_to_add=['test_feature'], data_to_add=np.zeros(shape=[self.df1.shape[0],]))
        self.assertTrue(df1columns < df.shape[1])
        return

    def test_custom_feature_filter(self):
        df = FeatureIO(dataframe=self.df1).custom_feature_filter(feature='AtomicNumber_composition_average', operator='<', threshold=21)
        self.assertTrue(df.shape[0] < self.df1.shape[0])
        return

class TestFeatureNormalization(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.read_csv(testdir + '/' + 'testcsv1.csv')
        del self.df1['Material compositions']
        self.configdict = ConfigFileParser(configfile='test_unittest_featuregeneration.conf').get_config_dict(path_to_file=testdir)
        self.target_feature = "O_pband_center_regression"
        self.x_features = [f for f in self.df1.columns.values.tolist()]
        self.x_features.remove('O_pband_center_regression')
        self.files = list()
        return

    def test_normalize_features(self):
        df, scaler = FeatureNormalization(dataframe=self.df1, configdict=self.configdict).normalize_features(x_features=self.x_features,
                                                                                 y_feature=self.target_feature,
                                                                                 normalize_x_features=True,
                                                                                 normalize_y_feature=False,
                                                                                 to_csv=True)
        self.assertTrue(np.mean(df['AtomicNumber_composition_average']) < 10**-8)
        self.assertTrue(np.std(df['AtomicNumber_composition_average']) == 1.0)
        self.files.append('input_data_normalized_O_pband_center_regression.csv')
        return

    def test_minmax_scale_single_feature(self):
        scaled_feature = FeatureNormalization(dataframe=self.df1, configdict=self.configdict).minmax_scale_single_feature(featurename='AtomicNumber_composition_average',
                                                                                                                          smin=5,
                                                                                                                          smax=10)
        self.assertTrue(np.mean(scaled_feature) > 3)
        return

    def test_unnormalize_features(self):
        df, scaler = FeatureNormalization(dataframe=self.df1, configdict=self.configdict).normalize_features(x_features=self.x_features,
                                                                                 y_feature=self.target_feature,
                                                                                 normalize_x_features=True,
                                                                                 normalize_y_feature=False,
                                                                                 to_csv=True)
        df, scaler = FeatureNormalization(dataframe=df, configdict=self.configdict).unnormalize_features(x_features=self.x_features,
                                                                                                         y_feature=self.target_feature,
                                                                                                         scaler=scaler)
        self.assertTrue(np.mean(df['AtomicNumber_composition_average']) > 10**-8)
        self.assertTrue(np.std(df['AtomicNumber_composition_average']) != 1.0)
        self.files.append('input_data_normalized_O_pband_center_regression.csv')
        return

    def test_normalize_and_merge_with_original_dataframe(self):
        df = FeatureNormalization(dataframe=self.df1, configdict=self.configdict).normalize_and_merge_with_original_dataframe(x_features=self.x_features,
                                                                                                                              y_feature=self.target_feature,
                                                                                                                              normalize_x_features=True,
                                                                                                                              normalize_y_feature=True)
        self.assertTrue(df.shape[1] == 2*self.df1.shape[1])
        self.files.append('input_data_normalized_O_pband_center_regression.csv')
        return

    def tearDown(self):
        for f in self.files:
            os.remove(f)
        return

if __name__=='__main__':
    unittest.main()