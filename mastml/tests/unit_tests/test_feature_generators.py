import unittest
import numpy as np
import pandas as pd
import os
import sys
import shutil
sys.path.insert(0, os.path.abspath('../../../'))

from mastml.feature_generators import ElementalFeatureGenerator, PolynomialFeatureGenerator, \
    OneHotElementEncoder, MaterialsProjectFeatureGenerator, OneHotGroupGenerator, ElementalFractionGenerator

class TestGenerators(unittest.TestCase):

    def test_elemental(self):
        composition_df = pd.DataFrame({'composition': ['NaCl', 'Al2O3', 'Mg', 'SrTiO3', 'C']})
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(5,5)), columns=['0', '1', '2', '3', '4'])
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(5,)))
        generator = ElementalFeatureGenerator(composition_df=composition_df, feature_types='max')
        Xgenerated, y = generator.evaluate(X=X, y=y, savepath=os.getcwd())
        self.assertEqual(Xgenerated.shape, (5, 92))
        self.assertTrue(os.path.exists(os.path.join(generator.splitdir, 'generated_features.xlsx')))
        shutil.rmtree(generator.splitdir)
        return

    def test_elementfraction(self):
        composition_df = pd.DataFrame({'composition': ['NaCl', 'Al2O3', 'Mg', 'SrTiO3', 'C']})
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(5,5)), columns=['0', '1', '2', '3', '4'])
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(5,)))
        generator = ElementalFractionGenerator(composition_df=composition_df)
        Xgenerated, y = generator.evaluate(X=X, y=y, savepath=os.getcwd())
        self.assertEqual(Xgenerated.shape, (5, 123))
        self.assertTrue(os.path.exists(os.path.join(generator.splitdir, 'generated_features.xlsx')))
        shutil.rmtree(generator.splitdir)
        return

    def test_polynomial(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(5, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(5,)))
        generator = PolynomialFeatureGenerator(features=None, degree=2, include_bias=False)
        Xgenerated, y = generator.evaluate(X=X, y=y, savepath=os.getcwd())
        self.assertEqual(Xgenerated.shape, (5, 75))
        self.assertTrue(os.path.exists(generator.splitdir))
        shutil.rmtree(generator.splitdir)
        return

    def test_onehotgroup(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(5, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(5,)))
        groups = pd.Series(['group1', 'group2' ,'group3', 'group1', 'group2'], name='group')
        generator = OneHotGroupGenerator(groups=groups)
        Xgenerated, y = generator.evaluate(X=X, y=y, savepath=os.getcwd())
        self.assertEqual(Xgenerated.shape, (5, 13))
        self.assertTrue(os.path.exists(generator.splitdir))
        shutil.rmtree(generator.splitdir)
        return

    def test_onehotelement(self):
        composition_df = pd.DataFrame({'composition': ['Al2O3', 'SrTiO3']})
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(2,10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(2,)))
        generator = OneHotElementEncoder(composition_df=composition_df, remove_constant_columns=False)
        Xgenerated, y = generator.evaluate(X=X, y=y)
        self.assertEqual(Xgenerated.shape, (2, 14))
        generator = OneHotElementEncoder(composition_df=composition_df, remove_constant_columns=True)
        Xgenerated, y = generator.evaluate(X=X, y=y)
        self.assertEqual(Xgenerated.shape, (2, 13))
        Xgenerated, y = generator.evaluate(X=X, y=y, savepath=os.getcwd())
        self.assertTrue(os.path.exists(generator.splitdir))
        shutil.rmtree(generator.splitdir)
        return

    #TODO: this will need to be updated with the latest Mat Proj API
    '''
    def test_materialsproject(self):
        composition_df = pd.DataFrame({'composition': ['Al2O3', 'SrTiO3']})
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(2,10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(2,)))
        generator = MaterialsProjectFeatureGenerator(composition_df=composition_df, api_key='TtAHFCrZhQa7cwEy')
        Xgenerated, y = generator.evaluate(X=X, y=y, savepath=os.getcwd())
        self.assertEqual(Xgenerated.shape, (2,31))
        self.assertTrue(os.path.exists(generator.splitdir))
        shutil.rmtree(generator.splitdir)
        return
    '''

if __name__=='__main__':
    unittest.main()