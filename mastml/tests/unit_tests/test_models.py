import unittest
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath('../../../'))

from mastml.models import SklearnModel, EnsembleModel

class TestModels(unittest.TestCase):

    def test_sklearnmodel(self):
        X =  pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(50,5)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(50,)))

        model = SklearnModel(model='LinearRegression')

        model.fit(X=X, y=y)
        ypred = model.predict(X=X, as_frame=True)

        self.assertEqual(ypred.shape, y.shape)
        return

    def test_ensemblemodel(self):
        X =  pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(50,5)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(50,)))

        model = EnsembleModel(model='LinearRegression', n_estimators=10)

        model.fit(X=X, y=y)
        ypred = model.predict(X=X, as_frame=True)

        self.assertEqual(ypred.shape, y.shape)
        return

if __name__=='__main__':
    unittest.main()