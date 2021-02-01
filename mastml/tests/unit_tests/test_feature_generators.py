import unittest
import numpy as np
import pandas as pd
import os
import sys
import shutil
sys.path.insert(0, os.path.abspath('../../../'))

from mastml.feature_generators import ElementalFeatureGenerator, PolynomialFeatureGenerator, \
    OneHotElementEncoder, MaterialsProjectFeatureGenerator

class TestGenerators(unittest.TestCase):

    def test_elemental(self):
        X = {'composition': ['NaCl', 'Al2O3', 'Mg', 'SrTiO3', 'C']}
        X = pd.DataFrame(X)
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(5,)))
        generator = ElementalFeatureGenerator(composition_feature='composition', feature_types='max')
        Xgenerated, y = generator.evaluate(X=X, y=y, savepath=os.getcwd())
        self.assertEqual(Xgenerated.shape, (5, 87))
        self.assertTrue(os.path.exists(os.path.join(generator.splitdir, 'generated_features.xlsx')))
        shutil.rmtree(generator.splitdir)
        return

    def test_polynomial(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(5, 10)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(5,)))
        generator = PolynomialFeatureGenerator(features=None, degree=2, include_bias=False)
        generator.fit(df=X, y=y)
        Xgenerated = generator.transform(df=X)
        self.assertEqual(Xgenerated.shape, (5, 65))
        return

    def test_onehotelement(self):
        X = {'composition': ['Al2O3', 'SrTiO3']}
        X = pd.DataFrame(X)
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(2,)))
        generator = OneHotElementEncoder(composition_feature='composition', remove_constant_columns=False)
        Xgenerated = generator.fit_transform(X=X, y=y)
        self.assertEqual(Xgenerated.shape, (2, 4))
        generator = OneHotElementEncoder(composition_feature='composition', remove_constant_columns=True)
        Xgenerated = generator.fit_transform(X=X, y=y)
        self.assertEqual(Xgenerated.shape, (2, 3))
        return

    def test_materialsproject(self):
        X = {'composition': ['Al2O3', 'SrTiO3']}
        X = pd.DataFrame(X)
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(2,)))
        generator = MaterialsProjectFeatureGenerator(composition_feature='composition', api_key='TtAHFCrZhQa7cwEy')
        Xgenerated = generator.fit_transform(X=X, y=y)
        self.assertEqual(Xgenerated.shape, (2,21))
        return

if __name__=='__main__':
    unittest.main()