import unittest
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath('../../../'))

from mastml.feature_selectors import NoSelect, EnsembleModelFeatureSelector, PearsonSelector, MASTMLFeatureSelector
from sklearn.ensemble import RandomForestRegressor

class TestSelectors(unittest.TestCase):

    def test_noselect(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(50, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(50,)))
        selector = NoSelect()
        Xselect = selector.evaluate(X=X, y=y)
        self.assertEqual(Xselect.shape, (50, 10))
        return

    def test_ensembleselector(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(50, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(50,)))
        model = RandomForestRegressor()
        selector = EnsembleModelFeatureSelector(model=model, k_features=5)
        Xselect = selector.evaluate(X=X, y=y)
        self.assertEqual(Xselect.shape, (50, 5))
        return

    def test_pearsonselector(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(10, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(10,)))
        selector = PearsonSelector(threshold_between_features=0.3,
                                   threshold_with_target=0.3,
                                   flag_highly_correlated_features=True,
                                   k_features=3)
        selector.fit(X=X, y=y, savepath=os.getcwd())
        Xselect = selector.transform(X=X)
        self.assertEqual(Xselect.shape, (10, 3))
        self.assertTrue(os.path.exists('Features_highly_correlated_with_target.xlsx'))
        self.assertTrue(os.path.exists('Full_correlation_matrix.xlsx'))
        self.assertTrue(os.path.exists('Highly_correlated_features_flagged.xlsx'))
        self.assertTrue(os.path.exists('Highly_correlated_features.xlsx'))
        os.remove('Features_highly_correlated_with_target.xlsx')
        os.remove('Full_correlation_matrix.xlsx')
        os.remove('Highly_correlated_features_flagged.xlsx')
        os.remove('Highly_correlated_features.xlsx')
        return

    def test_mastmlselector(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(10, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(10,)))
        model = RandomForestRegressor()
        selector = MASTMLFeatureSelector(estimator=model, n_features_to_select=2, cv=None, manually_selected_features=[1])
        selector.fit(X=X, y=y, savepath=os.getcwd())
        Xselect = selector.transform(X=X)
        self.assertEqual(Xselect.shape, (10, 2))
        self.assertTrue(os.path.exists(os.path.join(os.getcwd(), 'MASTMLFeatureSelector_data_feature_0.csv')))
        os.remove(os.path.join(os.getcwd(), 'MASTMLFeatureSelector_data_feature_0.csv'))
        return

if __name__=='__main__':
    unittest.main()