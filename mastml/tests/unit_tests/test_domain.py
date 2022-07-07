import unittest
import os
from mastml.datasets import LocalDatasets
from mastml.domain import Domain
from sklearn.model_selection import train_test_split

import mastml
try:
    data_path = mastml.__path__._path[0]
except:
    data_path = mastml.__path__[0]

class test_baseline(unittest.TestCase):
    def test_mahalanobis(self):
        target = 'E_regression'

        extra_columns = ['Material compositions 1', 'Material compositions 2','Hop activation barrier']

        d = LocalDatasets(file_path=data_path + '/data/diffusion_data_allfeatures.xlsx',
                          target=target,
                          extra_columns=extra_columns,
                          group_column='Material compositions 1',
                          testdata_columns=None,
                          as_frame=True)

        # Load the data with the load_data() method
        data_dict = d.load_data()

        # Let's assign each data object to its respective name
        X = data_dict['X']
        y = data_dict['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
        domain = Domain()
        domain.distance(X_train, X_test, 'mahalanobis')

        return

if __name__ == '__main__':
    unittest.main()