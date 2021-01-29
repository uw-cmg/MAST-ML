import unittest
import numpy as np
import pandas as pd
import os
import sys
import shutil
sys.path.insert(0, os.path.abspath('../../../'))

from mastml.preprocessing import SklearnPreprocessor

class TestPreprocessor(unittest.TestCase):

    def test_sklearnpreprocessor(self):
        # Make toy data of random numbers
        X =  pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(10,5)))

        preprocessor = SklearnPreprocessor(preprocessor='StandardScaler', as_frame=True)
        X_scaled = preprocessor.evaluate(X=X, savepath=os.getcwd())

        self.assertEqual(X_scaled.shape[0], 10)
        self.assertTrue(os.path.exists(os.path.join(preprocessor.splitdir, 'data_preprocessed.xlsx')))
        self.assertTrue(os.path.exists(os.path.join(preprocessor.splitdir, 'StandardScaler.pkl')))
        self.preprocessor = preprocessor
        return

    def tearDown(self):
        shutil.rmtree(self.preprocessor.splitdir)
        return

if __name__=='__main__':
    unittest.main()