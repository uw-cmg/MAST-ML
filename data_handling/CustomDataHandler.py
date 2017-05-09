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
__author__ = "Tam"

class CustomDataHandler():
    """Data handling class for creating custom columns.
        New methods may be added.

    Args:
        dataframe <data object>

    Returns:
    Raises:
        ValueError if dataframe is None
    """
    def __init__(self, dataframe=None):
        """Custom data handler
            
        Attributes:
            self.original_dataframe <data object>: Dataframe
            self.df <data object>: Dataframe
        """
        if dataframe is None:
            raise ValueError("No dataframe.")
        self.original_dataframe = copy.deepcopy(dataframe)
        self.df = copy.deepcopy(dataframe)
        return

    def calculate_EffectiveFluence(self, pvalue, flux_feature="",fluence_feature=""):
        fluence = self.df[fluence_feature]
        flux = self.df[flux_feature]
        Norm_fluence = self.df[fluence_feature]
        Norm_flux = self.df[flux_feature]

        EFl = np.log10( fluence*( 3E10 / flux )**pvalue )
        NEFl = np.log10( Norm_fluence*( 3E10 / Norm_flux )**pvalue )    
        
        Nmax = np.max([ np.max(EFl), np.max(NEFl)] )
        Nmin = np.min([ np.min(EFl), np.min(NEFl)] )
        
        N_EFl= (EFl-Nmin)/(Nmax-Nmin) #substitute with a self.featurenorm call?
        

        return N_EFl

    def testing_subtraction(self, param, col1="",col2=""):
        col1_data = self.df[col1]
        col2_data = self.df[col2]
        new_data = col1_data - col2_data + param

        return new_data

