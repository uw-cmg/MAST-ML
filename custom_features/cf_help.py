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
import importlib
__author__ = "Tam Mayeshiba"

def get_custom_feature_data(class_method_str=None,
                        starting_dataframe=None,
                        param_dict=dict(),
                        addl_feature_method_kwargs=dict()):
    """Get custom feature data from a method in one of the classes
        in this subpackage.
        The class must initialize with a dataframe.
        The method must take a dictionary of parameters, with
            numeric keys, as its first argument:
            params[0] = val1
            params[1] = val2 etc.
        The method can then take any number of additional keyword arguments.
        See Testing.py for an example class (Testing) and method (subtraction).
    Args:
        class_method_str <str>: Class and method string, for example, 
                "Testing.subtraction"
        starting_dataframe <pandas dataframe>: Starting dataframe
        param_dict <dict>: Parameter dictionary to be passed to the class method
        addl_feature_method_kwargs <dict>: Keyword dictionary to be passed
                                            to the class method
    """
    class_name = class_method_str.split(".")[0]
    module_name = "custom_features.%s" % class_name
    feature_name = class_method_str.split(".")[1]
    custom_module=importlib.import_module(module_name)
    custom_class = getattr(custom_module, class_name)
    custom_class_instance = custom_class(starting_dataframe)
    new_feature_data = getattr(custom_class_instance, feature_name)(param_dict,
                **addl_feature_method_kwargs)
    return (feature_name, new_feature_data)
