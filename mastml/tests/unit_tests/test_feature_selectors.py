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
        Xselect = selector.evaluate(X=X, y=y, savepath=os.getcwd())
        self.assertEqual(Xselect.shape, (50, 10))
        return

    def test_ensembleselector(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(50, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(50,)))
        model = RandomForestRegressor()
        selector = EnsembleModelFeatureSelector(model=model, n_features_to_select=5)
        Xselect = selector.evaluate(X=X, y=y, savepath=os.getcwd())
        self.assertEqual(Xselect.shape, (50, 5))
        self.assertTrue(os.path.exists('EnsembleModelFeatureSelector_feature_importances.xlsx'))
        os.remove('EnsembleModelFeatureSelector_feature_importances.xlsx')
        os.remove('selected_features.txt')
        return

    def test_pearsonselector(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(10, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(10,)))
        selector = PearsonSelector(threshold_between_features=0.3,
                                   threshold_with_target=0.3,
                                   flag_highly_correlated_features=True,
                                   n_features_to_select=3)
        selector.evaluate(X=X, y=y, savepath=os.getcwd())
        Xselect = selector.transform(X=X)
        self.assertEqual(Xselect.shape, (10, 3))
        self.assertTrue(os.path.exists('PearsonSelector_fullcorrelationmatrix.xlsx'))
        self.assertTrue(os.path.exists('PearsonSelector_highlycorrelatedfeatures.xlsx'))
        self.assertTrue(os.path.exists('PearsonSelector_highlycorrelatedfeaturesflagged.xlsx'))
        self.assertTrue(os.path.exists('PearsonSelector_highlycorrelatedwithtarget.xlsx'))
        os.remove('PearsonSelector_fullcorrelationmatrix.xlsx')
        os.remove('PearsonSelector_highlycorrelatedfeatures.xlsx')
        os.remove('PearsonSelector_highlycorrelatedfeaturesflagged.xlsx')
        os.remove('PearsonSelector_highlycorrelatedwithtarget.xlsx')
        os.remove('selected_features.txt')
        return

    def test_mastmlselector(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(10, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(10,)))
        model = RandomForestRegressor()
        selector = MASTMLFeatureSelector(model=model, n_features_to_select=2, cv=None, manually_selected_features=[1])
        selector.evaluate(X=X, y=y, savepath=os.getcwd())
        Xselect = selector.transform(X=X)
        self.assertEqual(Xselect.shape, (10, 2))
        self.assertTrue(os.path.exists(os.path.join(os.getcwd(), 'MASTMLFeatureSelector_featureselection_data.xlsx')))
        os.remove(os.path.join(os.getcwd(), 'MASTMLFeatureSelector_featureselection_data.xlsx'))
        os.remove('selected_features.txt')
        return

if __name__=='__main__':
    unittest.main()