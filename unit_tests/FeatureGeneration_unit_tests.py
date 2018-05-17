import unittest
import pandas as pd
import os
import sys
testdir = os.path.realpath(os.path.dirname(sys.argv[0]))
print(testdir)
moduledir = '/Users/ryanjacobs/PycharmProjects/MASTML_2018-03-08/MAST-ml-private/'
sys.path.append(moduledir)
from FeatureGeneration import MagpieFeatureGeneration, MaterialsProjectFeatureGeneration, CitrineFeatureGeneration
from MASTMLInitializer import ConfigFileParser

class TestMagpieFeatureGeneration(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.read_csv(testdir+'/'+'testcsv1.csv')
        self.configdict = ConfigFileParser(configfile='test_unittest_featuregeneration.conf').get_config_dict(path_to_file=testdir)
        self.files = list()
        return

    def test_generate_magpie_features(self):
        dataframe = MagpieFeatureGeneration(configdict=self.configdict, dataframe=self.df1).generate_magpie_features(save_to_csv=True)
        self.assertIsInstance(dataframe, pd.DataFrame)
        fname = testdir+'/'+'input_with_magpie_features_O_pband_center_regression.csv'
        self.assertTrue(os.path.exists(fname))
        self.files.append(fname)
        return

    def tearDown(self):
        for f in self.files:
            os.remove(f)
        return

class TestMaterialsProjectFeatureGeneration(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.read_csv(testdir+'/'+'testcsv1matproj.csv')
        self.configdict = ConfigFileParser(configfile='test_unittest_featuregeneration.conf').get_config_dict(path_to_file=testdir)
        self.files = list()
        self.mapi_key = self.configdict['Feature Generation']['materialsproject_apikey']
        return

    def test_generate_materialsproject_features(self):
        dataframe = MaterialsProjectFeatureGeneration(configdict=self.configdict, dataframe=self.df1,
                                                      mapi_key=self.mapi_key).generate_materialsproject_features(save_to_csv=True)
        self.assertIsInstance(dataframe, pd.DataFrame)
        fname = testdir+'/'+'input_with_matproj_features_O_pband_center_regression.csv'
        self.assertTrue(os.path.exists(fname))
        self.files.append(fname)
        return

    def tearDown(self):
        for f in self.files:
            os.remove(f)
        return

class TestCitrineFeatureGeneration(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.read_csv(testdir+'/'+'testcsv1matproj.csv')
        self.configdict = ConfigFileParser(configfile='test_unittest_featuregeneration.conf').get_config_dict(path_to_file=testdir)
        self.files = list()
        self.api_key = self.configdict['Feature Generation']['citrine_apikey']
        return

    def test_generate_citrine_features(self):
        dataframe = CitrineFeatureGeneration(configdict=self.configdict, dataframe=self.df1, api_key=self.api_key).generate_citrine_features(save_to_csv=True)
        self.assertIsInstance(dataframe, pd.DataFrame)
        fname = testdir+'/'+'input_with_citrine_features_O_pband_center_regression.csv'
        self.assertTrue(os.path.exists(fname))
        self.files.append(fname)
        return

    def tearDown(self):
        for f in self.files:
            os.remove(f)
        return

if __name__=='__main__':
    unittest.main()