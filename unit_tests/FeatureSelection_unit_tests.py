import unittest
import pandas as pd
import os
import sys
testdir = os.path.realpath(os.path.dirname(sys.argv[0]))
moduledir = '/Users/ryanjacobs/PycharmProjects/MASTML/'
sys.path.append(moduledir)
from FeatureSelection import DimensionalReduction, FeatureSelection, LearningCurve
from MASTMLInitializer import ConfigFileParser
from FeatureOperations import FeatureNormalization

class TestDimensionalReduction(unittest.TestCase):

    def setUp(self):
        self.df1constant = pd.read_csv(testdir+'/'+'testcsv1constant.csv')
        self.df1 = pd.read_csv(testdir+'/'+'testcsv1.csv')
        del self.df1constant['Material compositions'] # Need to remove string entries before doing feature selection
        del self.df1['Material compositions']
        self.configdict = ConfigFileParser(configfile='test_unittest_featuregeneration.conf').get_config_dict(path_to_file=testdir)
        self.target_feature = "O_pband_center_regression"
        self.x_features = [f for f in self.df1.columns.values.tolist() if f != self.target_feature]
        return

    def test_remove_constant_features(self):
        dataframe = DimensionalReduction(dataframe=self.df1constant, x_features=self.x_features, y_feature=self.target_feature).remove_constant_features()
        self.assertTrue(dataframe.shape[1] < self.df1constant.shape[1])
        dataframe = DimensionalReduction(dataframe=self.df1, x_features=self.x_features, y_feature=self.target_feature).remove_constant_features()
        self.assertTrue(dataframe.shape[1] == self.df1.shape[1])
        return

    def test_principal_component_analysis(self):
        dataframe = DimensionalReduction(dataframe=self.df1, x_features=self.x_features, y_feature=self.target_feature).principal_component_analysis()
        self.assertIsInstance(dataframe, pd.DataFrame)
        return

class TestFeatureSelection(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.read_csv(testdir+'/'+'testcsv1featureselection.csv')
        del self.df1['Material compositions']
        self.configdict = ConfigFileParser(configfile='test_unittest_featuregeneration.conf').get_config_dict(path_to_file=testdir)
        self.target_feature = "O_pband_center_regression"
        self.x_features = [f for f in self.df1.columns.values.tolist() if f != self.target_feature]
        self.model_type = 'gkrr_model_regressor'
        self.files = list()
        # Need to normalize features for feature selection
        self.df1, scaler = FeatureNormalization(dataframe=self.df1, configdict=self.configdict).normalize_features(x_features=self.x_features,
                                                                               y_feature=self.target_feature,
                                                                               normalize_x_features= True,
                                                                               normalize_y_feature= False,
                                                                               to_csv=False)
        return

    def test_sequential_forward_selection(self):
        dataframe = FeatureSelection(configdict=self.configdict, dataframe=self.df1, x_features=self.x_features,
                                     y_feature=self.target_feature, model_type=self.model_type).sequential_forward_selection(number_features_to_keep=2)
        self.assertTrue(dataframe.shape[1] == 3) # number of columns will be # features chosen + 1
        self.files.append('sequential_forward_selection_data_O_pband_center_regression.csv')
        self.files.append('input_with_sequential_forward_selection_O_pband_center_regression.csv')
        self.files.append('sequential_forward_selection_learning_curve_O_pband_center_regression.pdf')
        return

    def test_univariate_feature_selection(self):
        dataframe = FeatureSelection(configdict=self.configdict, dataframe=self.df1, x_features=self.x_features,
                                     y_feature=self.target_feature, model_type=self.model_type).\
            feature_selection(feature_selection_type='univariate_feature_selection', number_features_to_keep=2,
                              use_mutual_info=False)
        self.assertTrue(dataframe.shape[1] == 3)  # number of columns will be # features chosen + 1
        dataframe = FeatureSelection(configdict=self.configdict, dataframe=self.df1, x_features=self.x_features,
                                     y_feature=self.target_feature, model_type=self.model_type).\
            feature_selection(feature_selection_type='univariate_feature_selection', number_features_to_keep=2,
                              use_mutual_info=True)
        self.assertTrue(dataframe.shape[1] == 3)  # number of columns will be # features chosen + 1
        self.files.append('input_with_univariate_feature_selection_O_pband_center_regression.csv')
        return

    def test_rfe_feature_selection(self):
        dataframe = FeatureSelection(configdict=self.configdict, dataframe=self.df1, x_features=self.x_features,
                                     y_feature=self.target_feature, model_type=self.model_type).\
            feature_selection(feature_selection_type='recursive_feature_elimination', number_features_to_keep=2,
                              use_mutual_info=False)
        self.assertTrue(dataframe.shape[1] == 3)  #
        # number of columns will be # features chosen + 1
        self.files.append('input_with_recursive_feature_elimination_O_pband_center_regression.csv')
        return

    def tearDown(self):
        for f in self.files:
            os.remove(f)
        return

class TestLearningCurve(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.read_csv(testdir+'/'+'testcsv1featureselection.csv')
        del self.df1['Material compositions']
        self.configdict = ConfigFileParser(configfile='test_unittest_featuregeneration.conf').get_config_dict(path_to_file=testdir)
        self.target_feature = "O_pband_center_regression"
        self.x_features = [f for f in self.df1.columns.values.tolist() if f != self.target_feature]
        self.model_type = 'gkrr_model_regressor'
        # Need to normalize features for feature selection
        self.df1, scaler = FeatureNormalization(dataframe=self.df1, configdict=self.configdict).normalize_features(x_features=self.x_features,
                                                                               y_feature=self.target_feature,
                                                                               normalize_x_features= True,
                                                                               normalize_y_feature= False,
                                                                               to_csv=False)
        self.files = list()
        return

    def test_generate_feature_learning_curve(self):
        feature_selection_algorithms = ['univariate_feature_selection', 'recursive_feature_elimination']
        for fsa in feature_selection_algorithms:
            LearningCurve(configdict=self.configdict, dataframe=self.df1, model_type=self.model_type).generate_feature_learning_curve(feature_selection_algorithm=fsa)
        self.files.append('univariate_feature_selection_learning_curve_featurenumber.pdf')
        self.files.append('univariate_feature_selection_learning_curve_trainingdata.pdf')
        self.files.append('recursive_feature_elimination_learning_curve_trainingdata.pdf')
        self.files.append('recursive_feature_elimination_learning_curve_featurenumber.pdf')
        self.files.append('input_with_univariate_feature_selection_O_pband_center_regression.csv')
        self.files.append('input_with_recursive_feature_elimination_O_pband_center_regression.csv')
        return

    def tearDown(self):
        for f in self.files:
            os.remove(f)
        return

if __name__=='__main__':
    unittest.main()