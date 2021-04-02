import unittest
import numpy as np
import pandas as pd
import os
import sys
import shutil
sys.path.insert(0, os.path.abspath('../../../'))

from mastml.data_cleaning import DataCleaning

class TestDataCleaning(unittest.TestCase):

    def test_datacleaning(self):
        X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(50, 10)))
        y_input = pd.Series(np.random.uniform(low=0.0, high=100, size=(50,)))
        y_input.name = 'target'
        X_input = X.mask(np.random.random(X.shape) < 0.1)
        DataCleaning().remove(X=X_input, y=y_input, axis=1)
        DataCleaning().imputation(X=X_input, y=y_input, strategy='mean')
        DataCleaning().ppca(X=X_input, y=y_input)
        cleaner = DataCleaning()
        cleaner.evaluate(X=X_input, y=y_input, method='remove', axis=1, savepath=os.getcwd())
        self.assertTrue(os.path.exists(cleaner.splitdir))
        shutil.rmtree(cleaner.splitdir)
        return

if __name__=='__main__':
    unittest.main()