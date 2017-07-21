#!/usr/bin/env python
################################
# E900 model for DBTT project
################################
from custom_models.BaseCustomModel import BaseCustomModel
class model(BaseCustomModel):
    def __init__(self, **kwargs):
        """
            self.wtP: feature name for weight percentage of P
            self.wtNi
            self.wtMn
            self.wtCu
            self.fluencestr: feature name for fluence in n/cm2
            self.tempC: feature name for temperature in C
            self.prod_ID: feature name for product ID (plate, weld,
                            forging, etc.)
        """
        self.wtP="P"
        self.wtNi="Ni"
        self.wtMn="Mn"
        self.wtCu="Cu"
        self.fluencestr="fluence_n_cm2"
        self.tempC="temperature_C"
        self.prod_ID="Product ID"
        BaseCustomModel.__init__(**kwargs)
        return

    def predict(self, input_data):
        """E900 model, ASTM E900-15
            CC conversion of TTS to delta sigma Y is from ORNL/TM-2006/530
            equation 6-3 and 6-4.
        """
        p = input_data[wtP]
        ni = input_data[wtNi]
        mn = input_data[wtMn]
        cu = input_data[wtCu]
        tempc = input_data[tempC]
        flu = input_data[fluencestr]
        prod = input_data[prod_ID]

        numvals = len(p)

        fluence_n_m2 = flu * 100.0 * 100.0

        plates = np.array(prod == "P")
        forgings = np.array(prod == "F")
        srm_plates = np.array(prod=="SRM")
        weldings = np.array(prod == "W")

        Acol = np.empty(numvals)
        Acol.fill(1.080) #plates as default
        Acol[plates] = 1.080 #plate
        Acol[forgings] = 1.011 #forging
        Acol[srm_plates] = 1.080 #standard reference material plate
        Acol[weldings] = 0.919 #weld
        
        Bcol = np.empty(numvals)
        Bcol.fill(0.819) #plates as default
        Bcol[plates] = 0.819 #plate
        Bcol[forgings] =  0.738 #forging
        Bcol[srm_plates] = 0.819 #SRM plate
        Bcol[weldings] = 0.968 #weld

        tts1 = Acol * (5./9.) * 1.8943 * np.power(10.,-12.) * np.power(fluence_n_m2,0.5695) * np.power(((1.8 * tempc + 32.0)/550.0),-5.47) * np.power((0.09 + p/0.012),0.216) * np.power((1.66 + (np.power(ni,8.54))/0.63),0.39) * np.power((mn/1.36),0.3)
        
        Mcol = Bcol * np.maximum(np.minimum(113.87 * (np.log(fluence_n_m2) - np.log(4.5 * np.power(10.0,20.0))), 612.6), 0.) * np.power(((1.8 * tempc + 32.0)/550.0), -5.45) * np.power((0.1 + p/0.012),-0.098) * np.power((0.168 + np.power(ni,0.58)/0.63),0.73)

        tts2 = (5./9.) * np.maximum(np.minimum(cu, 0.28) - 0.053,0) * Mcol

        tts = tts1 + tts2

        cc = np.empty(numvals)
        cc[weldings] = 0.55 + (1.2e-3) * tts[weldings] - 1.33e-6 * np.power(tts[weldings],2.0)
        cc[plates] = 0.45 + (1.945e03) * tts[plates] - 5.496e-6 * np.power(tts[plates],2.0) + 8.473e-9 * np.power(tts[plates],3.0)
        others = ~np.any([weldings, plates],axis=0)
        cc[others] = 0.45 + (1.945e03) * tts[others] - 5.496e-6 * np.power(tts[others],2.0) + 8.473e-9 * np.power(tts[others],3.0)
        
        ds = tts / cc
        return ds
