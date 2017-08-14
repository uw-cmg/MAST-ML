import numpy as np
import copy
from FeatureOperations import FeatureNormalization, FeatureIO
__author__ = "Tam Mayeshiba"

class DBTT():
    """Class for creating custom feature columns specifically for the 
        DBTT project.
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

        Each custom feature should take keyword arguments.
        """
        if dataframe is None:
            raise ValueError("No dataframe.")
        self.original_dataframe = copy.deepcopy(dataframe)
        self.df = copy.deepcopy(dataframe)
        return

    def calculate_EffectiveFluence(self, pvalue=0, ref_flux = 3e10, flux_feature="",fluence_feature="", scale_min = 1e17, scale_max = 1e25, **params):
        """Calculate effective fluence
        """
        fluence = self.df[fluence_feature]
        flux = self.df[flux_feature]

        EFl = fluence * (ref_flux / flux) ** pvalue
        EFl = np.log10(EFl)
        fio = FeatureIO(self.df)
        new_df = fio.add_custom_features(["EFl"],EFl)
        fnorm = FeatureNormalization(new_df)
        N_EFl = fnorm.minmax_scale_single_feature("EFl", 
                                            smin = np.log10(scale_min),
                                            smax = np.log10(scale_max))

        return N_EFl

