import unittest
import pandas as pd
import os
import sys
testdir = os.path.realpath(os.path.dirname(sys.argv[0]))
moduledir = '/Users/ryanjacobs/PycharmProjects/MASTML/'
sys.path.append(moduledir)
from MASTMLInitializer import ConfigFileParser, ConfigFileValidator, MASTMLWrapper
from DataHandler import DataHandler
import configobj

class TestConfigFileParser(unittest.TestCase):

    def setUp(self):
        self.configfile = 'test_unittest_featuregeneration.conf'
        return

    def test_get_config_dict(self):
        configdict = ConfigFileParser(configfile=self.configfile).get_config_dict(path_to_file=testdir)
        self.assertIsInstance(configdict, configobj.ConfigObj)
        return

class TestConfigFileValidator(unittest.TestCase):

    def setUp(self):
        self.configfile = 'test_unittest_featuregeneration.conf'
        self.configfilebroken = 'test_unittest_featuregeneration_broken.conf'
        return

    def test_run_config_validation(self):
        configdict, errors_present = ConfigFileValidator(configfile=self.configfile).run_config_validation()
        self.assertTrue(errors_present is False)
        configdict, errors_present = ConfigFileValidator(configfile=self.configfilebroken).run_config_validation()
        self.assertTrue(errors_present is True)
        return

class TestMASTMLInitializer(unittest.TestCase):

    def setUp(self):
        self.y_features = ['classification', 'regression']
        self.classification_model_types = ['support_vector_machine_model_classifier','logistic_regression_model_classifier',
                            'decision_tree_model_classifier','random_forest_model_classifier','extra_trees_model_classifier',
                            'adaboost_model_classifier','nn_model_classifier']
        self.regression_model_types = ['linear_model_regressor', 'linear_model_lasso_regressor', 'support_vector_machine_model_regressor',
                                       'lkrr_model_regressor','gkrr_model_regressor','decision_tree_model_regressor',
                                       'extra_trees_model_regressor', 'randomforest_model_regressor', 'adaboost_model_regressor',
                                       'nn_model_regressor']
        self.configfile = 'test_unittest_featuregeneration.conf'
        self.df1 = pd.read_csv(testdir+'/'+'testcsv1featureselection.csv')
        del self.df1['Material compositions']
        self.configdict = ConfigFileParser(configfile='test_unittest_featuregeneration.conf').get_config_dict(path_to_file=testdir)
        self.target_feature = "O_pband_center_regression"
        self.x_features = [f for f in self.df1.columns.values.tolist() if f != self.target_feature]
        return

    def test_get_machinelearning_model(self):
        for feature_type in self.y_features:
            if feature_type == 'classification':
                for model_type in self.classification_model_types:
                    model = MASTMLWrapper(configdict=self.configdict).get_machinelearning_model(model_type=model_type, y_feature=feature_type)
                    self.assertTrue(type(model) is not None)
            if feature_type == 'regression':
                for model_type in self.regression_model_types:
                    model = MASTMLWrapper(configdict=self.configdict).get_machinelearning_model(model_type=model_type, y_feature=feature_type)
                    self.assertTrue(type(model) is not None)
        return

    def test_get_machinelearning_test(self):
        data = DataHandler(data=self.df1, input_data=self.df1[self.x_features], target_data=self.df1[self.target_feature],
                           input_features=self.x_features, target_feature=self.target_feature)
        test_types = self.configdict['Models and Tests to Run']['test_cases']
        model = self.configdict['Models and Tests to Run']['models']
        for test_type in test_types:
            test_class = MASTMLWrapper(configdict=self.configdict).get_machinelearning_test(test_type=test_type,
                                                                                        model=model,
                                                                                        save_path=testdir,
                                                                                        run_test=False,
                                                                                        training_dataset=[data],
                                                                                        testing_dataset=[data],
                                                                                        plot_filter_out=None,
                                                                                        feature_plot_feature=self.target_feature,
                                                                                        data_labels='Initial')
            self.assertIsNotNone(test_class)
        return

if __name__=='__main__':
    unittest.main()