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
            self.wtP_feature: feature name for weight percentage of P
            self.wtNi_feature
            self.wtMn_feature
            self.wtCu_feature
            self.fluence_n_cm2_feature: feature name for fluence in n/cm2
            self.temp_C_feature: feature name for temperature in C
            self.product_id_feature: feature name for product ID (plate, weld,
                            forging, etc.)
        """
        self.wtP_feature="P"
        self.wtNi_feature="Ni"
        self.wtMn_feature="Mn"
        self.wtCu_feature="Cu"
        self.fluence_n_cm2_feature="fluence_n_cm2"
        self.temp_C_feature="temperature_C"
        self.product_id_feature="Product ID"
        BaseCustomModel.__init__(self, **kwargs)
        return

    def predict(self, input_data):
        """E900 model, ASTM E900-15
            Cc conversion of TTS to delta sigma Y is from ORNL/TM-2006/530
            equation 6-3 and 6-4.
        """
        # Set up working dataframe
        empty_cols=list()
        empty_cols.append("Acol")
        empty_cols.append("Bcol")
        empty_cols.append("Cc")
        pdf = pd.DataFrame(index=input_data.index, columns=empty_cols)
        pdf = pdf.merge(input_data, left_index=True, right_index=True)
        # Set up shortcut variables 
        wtP = pdf[self.wtP_feature]
        wtNi = pdf[self.wtNi_feature]
        wtMn = pdf[self.wtMn_feature]
        wtCu = pdf[self.wtCu_feature]
        tempC = pdf[self.temp_C_feature]
        product = pdf[self.product_id_feature]
        fluence_n_cm2 = pdf[self.fluence_n_cm2_feature]
        # Convert fluence
        pdf["fluence_n_m2"] = fluence_n_cm2 * 100.0 * 100.0
        # Get product indices
        plates_idx = pdf[product == "P"].index
        forgings_idx = pdf[product == "F"].index
        srm_plates_idx = pdf[product == "SRM"].index #standard ref. matl plates
        welds_idx = pdf[product == "W"].index
        # Set Acol according to product type
        pdf.loc[forgings_idx, 'Acol'] = 1.011
        pdf.loc[plates_idx, 'Acol'] = 1.080
        pdf.loc[srm_plates_idx, 'Acol'] = 1.080
        pdf.loc[welds_idx, 'Acol'] = 0.919
        # Set Bcol according to product type
        pdf.loc[forgings_idx, 'Bcol'] = 0.738
        pdf.loc[plates_idx, 'Bcol'] = 0.819
        pdf.loc[srm_plates_idx, 'Bcol'] = 0.819
        pdf.loc[welds_idx, 'Bcol'] = 0.968
        # Get TTS1
        pdf['TTS1_front'] = pdf.Acol * (5./9.) * 1.8943e-12 * np.power(pdf.fluence_n_m2, 0.5695)
        pdf['TTS1_temp'] = np.power( ((1.8 * tempC + 32.)/550.) , -5.47)
        pdf['TTS1_P'] = np.power( (0.09 + (wtP/0.012)) , 0.216)
        pdf['TTS1_Ni'] = np.power( (1.66 + (np.power(wtNi, 8.54)/0.63)) , 0.39)
        pdf['TTS1_Mn'] = np.power( (wtMn/1.36), 0.3)
        pdf['TTS1'] = pdf.TTS1_front * pdf.TTS1_temp * pdf.TTS1_P * pdf.TTS1_Ni * pdf.TTS1_Mn
        # Get TTS2
        pdf['TTS2_front'] = (5./9.)
        pdf['TTS2_Cu'] = np.maximum(np.minimum(wtCu, 0.28) - 0.053, 0)
        pdf['M_flux'] = np.maximum(np.minimum(113.87 * (np.log(pdf.fluence_n_m2) - np.log(4.5e20)), 612.6), 0)
        pdf['M_temp'] = np.power( ((1.8*tempC + 32.0)/550.) , -5.45)
        pdf['M_P'] = np.power( (0.1 + (wtP/0.012)), -0.098)
        pdf['M_Ni'] = np.power( (0.168 + (np.power(wtNi, 0.58)/0.63)), 0.73)
        pdf['Mcol'] = pdf.Bcol * pdf.M_flux * pdf.M_temp * pdf.M_P * pdf.M_Ni
        pdf['TTS2'] = pdf.TTS2_front * pdf.TTS2_Cu * pdf.Mcol
        # Get TTS
        pdf['TTS'] = pdf.TTS1 + pdf.TTS2
        # Get Cc according to product type
        c_welds = [0.55, 1.2e-3, -1.33e-6, 0.0]
        c_plates = [0.45, 1.945e-3, -5.496e-6, 8.473e-9]
        # Cc default to plates
        pdf.Cc = c_plates[0] + c_plates[1]*pdf.TTS + c_plates[2]*np.power(pdf.TTS, 2.0) + c_plates[3]*np.power(pdf.TTS, 3.0)
        # Cc for welds
        pdf.loc[welds_idx, 'Cc'] = c_welds[0] + c_welds[1]*pdf.TTS[welds_idx] + c_welds[2]*np.power(pdf.TTS[welds_idx], 2.0) + c_welds[3]*np.power(pdf.TTS[welds_idx], 3.0)
        # Get hardening (delta_sigma_y_MPa)
        pdf['delta_sigma_y_MPa'] = pdf.TTS / pdf.Cc
        pdf.to_csv(os.path.join(os.getcwd(),"E900.csv"))
        return pdf['delta_sigma_y_MPa']
        
    def get_params(self):
        param_dict=dict()
        param_dict['wtP_feature'] = self.wtP_feature
        param_dict['wtNi_feature'] = self.wtNi_feature
        param_dict['wtMn_feature'] = self.wtMn_feature
        param_dict['wtCu_feature'] = self.wtCu_feature
        param_dict['fluence_n_cm2_feature'] = self.fluence_n_cm2_feature
        param_dict['temp_C_feature'] = self.temp_C_feature
        param_dict['product_id_feature'] = self.product_id_feature
        return param_dict
