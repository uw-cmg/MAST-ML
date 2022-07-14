import unittest
import numpy as np
import pandas as pd
import os
import sys
import shutil
sys.path.insert(0, os.path.abspath('../../../'))

from mastml.preprocessing import SklearnPreprocessor, MeanStdevScaler

class TestPreprocessor(unittest.TestCase):

    def test_sklearnpreprocessor(self):
        # Make toy data of random numbers
        X =  pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(10,5)))

        preprocessor = SklearnPreprocessor(preprocessor='StandardScaler', as_frame=True)
        X_scaled = preprocessor.evaluate(X=X, savepath=os.getcwd(), file_name='test')

        self.assertEqual(X_scaled.shape[0], 10)
        self.assertTrue(os.path.exists(os.path.join(preprocessor.savepath, 'data_preprocessed_test.csv')))
        self.assertTrue(os.path.exists(os.path.join(preprocessor.savepath, 'StandardScaler.pkl')))
        os.remove('data_preprocessed_test.csv')
        os.remove('StandardScaler.pkl')

        return

    def test_meanstdevscaler(self):
        # Make toy data of random numbers
        X =  pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(10,5)))

        preprocessor = MeanStdevScaler(mean=0, stdev=2, as_frame=True)
        X_scaled = preprocessor.evaluate(X=X, savepath=os.getcwd(), file_name='test')

        self.assertEqual(X_scaled.shape[0], 10)
        self.assertAlmostEqual(X_scaled[0].std(), 2)
        self.assertTrue(os.path.exists(os.path.join(preprocessor.savepath, 'data_preprocessed_test.csv')))
        self.assertTrue(os.path.exists(os.path.join(preprocessor.savepath, 'MeanStdevScaler.pkl')))
        os.remove('data_preprocessed_test.csv')
        os.remove('MeanStdevScaler.pkl')

        return

if __name__=='__main__':
    unittest.main()