import numpy as np
import copy
from DataParser import FeatureNormalization
from DataParser import FeatureIO
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

        Each custom feature should take a parameter dictionary, with integers
            as keys starting from zero, followed by keyword arguments.
        """
        if dataframe is None:
            raise ValueError("No dataframe.")
        self.original_dataframe = copy.deepcopy(dataframe)
        self.df = copy.deepcopy(dataframe)
        return

    def calculate_EffectiveFluence(self, params=dict(), ref_flux = 3e10, flux_feature="",fluence_feature="", scale_min = 1e17, scale_max = 1e25):
        """Calculate effective fluence
            params[0]: p value
        """
        pvalue = params[0]
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

    def E900(self, wtP="P",
                    wtNi="Ni",
                    wtMn="Mn",
                    wtCu="Cu",
                    fluence="fluence_n_cm2",
                    tempC="temperature_C",
                    prod_ID="Product ID",
                        ):
        """E900 model, ASTM E900-15
            CC conversion of TTS to delta sigma Y is from ORNL/TM-2006/530
            equation 6-3 and 6-4.
        Args: 
            wtP <str>: feature name for weight percentage of P
            similarly, wtNi, wtMn, wtCu
            fluence <str>: feature name for fluence in n/cm2
            tempC <str>: feature name for temperature in C
            prod_ID <str>: feature name for product ID (plate, weld,
                            forging, etc.)
        """
        p = self.df[wtP]
        ni = self.df[wtNi]
        mn = self.df[wtMn]
        cu = self.df[wtCu]
        tempc = self.df[tempC]
        flu = self.df[fluencestr]
        prod = self.df[prod_ID]

        numvals = len(p)

        fluence_n_m2 = flu * 100.0 * 100.0

        Acol = np.empty(numvals)
        Acol.fill(1.080) #plates as default
        Acol[prod == "P"] = 1.080 #plate
        Acol[prod == "F"] = 1.011 #forging
        Acol[prod == "SRM"] = 1.080 #SRM plate
        Acol[prod == "W"] = 0.919 #weld
        
        Bcol = np.empty(numvals)
        Bcol.fill(0.819) #plates as default
        Bcol[prod == "P"] = 0.819 #plate
        Bcol[prod == "F"] =  0.738 #forging
        Bcol[prod == "SRM"] = 0.819 #SRM plate
        Bcol[prod == "W"] = 0.968 #weld

        tts1 = Acol * (5./9.) * 1.8943 * np.power(10.,-12.) * np.power(fluence_n_m2,0.5695) * np.power(((1.8 * tempc + 32.0)/550.0),-5.47) * np.power((0.09 + p/0.012),0.216) * np.power((1.66 + (np.power(ni,8.54))/0.63),0.39) * np.power((mn/1.36),0.3)
        
        Mcol = Bcol * np.maximum(np.minimum(113.87 * (np.log(fluence_n_m2) - np.log(4.5 * np.power(10.0,20.0))), 612.6), 0.) * np.power(((1.8 * tempc + 32.0)/550.0), -5.45) * np.power((0.1 + p/0.012),-0.098) * np.power((0.168 + np.power(ni,0.58)/0.63),0.73)

        tts2 = (5./9.) * np.maximum(np.minimum(cu, 0.28) - 0.053,0) * Mcol

        tts = tts1 + tts2

        cc = np.empty(numvals)
        cc[prod == "W"] = 0.55 + (1.2e-3) * tts - 1.33e-6 * np.power(tts,2.0)
        cc[prod == "P"] = 0.45 + (1.945e03) * tts - 5.496e-6 * np.power(tts,2.0) + 8.473e-9 * np.power(tts,3.0)
        cc[~self.df[prod].isin("P","W")] = 0.45 + (1.945e03) * tts - 5.496e-6 * np.power(tts,2.0) + 8.473e-9 * np.power(tts,3.0)
        
        ds = tts / cc
        return ds

