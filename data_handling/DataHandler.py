import numpy as np
import data_parser
import matplotlib
import matplotlib.pyplot as plt
import copy
from SingleFit import SingleFit
from SingleFit import timeit
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
#from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from multiprocessing import Process,Pool,TimeoutError,Value,Array,Manager
import time
import copy
from DataParser import FeatureNormalization
from DataParser import FeatureIO
__author__ = "Tam Mayeshiba"

class DataHandler():
    """Data handling class
        (Combines old data_parser functionality with new DataParser methods)

    Args:
        dataframe <pandas dataframe>
        Xdata <pandas dataframe>: X data (input data)
        ydata <pandas dataframe>: y data (target data)
        x_features <list of str>: x features (input features)
        y_feature <list of str>: y feature (target feature)
        
        as parsed from DataParser

    Returns:
    Raises:
        ValueError if dataframe is None
    """
    def __init__(self, dataframe=None, Xdata=None, ydata=None, 
                    x_features=None, y_feature=None):
        """Data Handler
            
        Attributes:
            self.data <dataframe>: Main dataframe; all data
            self.input_data <dataframe>: Input data
            self.target_data <dataframe>: Target data
            self.input_features <list of str>: Input features
            self.target_feature <list of str>: Target feature
        """
        if dataframe is None:
            raise ValueError("No dataframe.")
        self.data = copy.deepcopy(dataframe)
        self.input_data = copy.deepcopy(Xdata)
        self.target_data = copy.deepcopy(ydata)
        self.input_features = x_features
        self.target_feature = y_feature
        return
