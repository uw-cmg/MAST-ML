#!/usr/bin/env python
################################
# E900 model for DBTT project
################################
from custom_models.BaseCustomModel import BaseCustomModel
import numpy as np
import pandas as pd
import os
class E900model(BaseCustomModel):
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
        BaseCustomModel.__init__(self, **kwargs)
        return

    def predict(self, input_data):
        """E900 model, ASTM E900-15
            CC conversion of TTS to delta sigma Y is from ORNL/TM-2006/530
            equation 6-3 and 6-4.
        """
        p = input_data[self.wtP]
        ni = input_data[self.wtNi]
        mn = input_data[self.wtMn]
        cu = input_data[self.wtCu]
        tempc = input_data[self.tempC]

        pdf = pd.DataFrame(index=input_data.index, columns=['Acol','Bcol','cc'])
        pdf['fluence_n_m2'] = input_data[self.fluencestr] * 100.0 * 100.0

        plates_idx = input_data[input_data[self.prod_ID] == "P"].index
        forgings_idx = input_data[input_data[self.prod_ID] == "F"].index
        srm_plates_idx = input_data[input_data[self.prod_ID] == "SRM"].index
        weldings_idx = input_data[input_data[self.prod_ID] == "W"].index

        for idx in plates_idx:
            pdf.set_value(idx, 'Acol', 1.080)
            pdf.set_value(idx, 'Bcol', 0.819)
        for idx in forgings_idx:
            pdf.set_value(idx, 'Acol', 1.011)
            pdf.set_value(idx, 'Bcol', 0.738)
        for idx in srm_plates_idx:
            pdf.set_value(idx, 'Acol', 1.080) #standard reference material plate
            pdf.set_value(idx, 'Bcol', 0.819)
        for idx in weldings_idx:
            pdf.set_value(idx, 'Acol', 0.919)
            pdf.set_value(idx, 'Bcol', 0.968)

        pdf['tts1'] = pdf.Acol * (5./9.) * 1.8943 * np.power(10.,-12.) * np.power(pdf['fluence_n_m2'],0.5695) * np.power(((1.8 * input_data[self.tempC] + 32.0)/550.0),-5.47) * np.power((0.09 + input_data[self.wtP]/0.012),0.216) * np.power((1.66 + (np.power(input_data[self.wtNi],8.54))/0.63),0.39) * np.power((input_data[self.wtMn]/1.36),0.3)
        
        pdf['Mcol'] = pdf.Bcol * np.maximum(np.minimum(113.87 * (np.log(pdf['fluence_n_m2']) - np.log(4.5 * np.power(10.0,20.0))), 612.6), 0.) * np.power(((1.8 * input_data[self.tempC] + 32.0)/550.0), -5.45) * np.power((0.1 + input_data[self.wtP]/0.012),-0.098) * np.power((0.168 + np.power(input_data[self.wtNi],0.58)/0.63),0.73)

        pdf['tts2'] = (5./9.) * np.maximum(np.minimum(cu, 0.28) - 0.053,0) * pdf.Mcol

        pdf['tts'] = pdf.tts1 + pdf.tts2

        pdf.cc[weldings_idx] = 0.55 + (1.2e-3) * pdf.tts[weldings_idx] - 1.33e-6 * np.power(pdf.tts[weldings_idx],2.0)
        pdf.cc[plates_idx] = 0.45 + (1.945e-3) * pdf.tts[plates_idx] - 5.496e-6 * np.power(pdf.tts[plates_idx],2.0) + 8.473e-9 * np.power(pdf.tts[plates_idx],3.0)
        pdf.cc[srm_plates_idx] = 0.45 + (1.945e-3) * pdf.tts[srm_plates_idx] - 5.496e-6 * np.power(pdf.tts[srm_plates_idx],2.0) + 8.473e-9 * np.power(pdf.tts[srm_plates_idx],3.0)
        
        pdf.cc[forgings_idx] = 0.45 + (1.945e-3) * pdf.tts[forgings_idx] - 5.496e-6 * np.power(pdf.tts[forgings_idx],2.0) + 8.473e-9 * np.power(pdf.tts[forgings_idx],3.0)
        
        pdf['ds'] = pdf['tts'] / pdf['cc']
        pdf = pdf.merge(input_data, left_index=True, right_index=True)
        pdf.to_csv(os.path.join(os.getcwd(),"E900.csv"))
        return pdf['ds']
        
    def get_params(self):
        param_dict=dict()
        param_dict['wtP'] = self.wtP
        param_dict['wtNi'] = self.wtNi
        param_dict['wtMn'] = self.wtMn
        param_dict['wtCu'] = self.wtCu
        param_dict['fluencestr'] = self.fluencestr
        param_dict['tempC'] = self.tempC
        param_dict['prod_ID'] = self.prod_ID
        return param_dict
