#!/usr/bin/env python
################################
# EONY model for DBTT project
################################
from custom_models.BaseCustomModel import BaseCustomModel
import numpy as np
import pandas as pd
import os
class EONYmodel(BaseCustomModel):
    def __init__(self, **kwargs):
        """
            self.wtP_feature: feature name for weight percentage of P
            self.wtNi_feature
            self.wtMn_feature
            self.wtCu_feature
            self.flux_n_cm2_sec_feature
            self.fluence_n_cm2_feature: feature name for fluence in n/cm2
            self.temp_C_feature: feature name for temperature in C
            self.product_id_feature: feature name for product ID (plate, weld,
                            forging, etc.)
        """
        self.wtP_feature="P"
        self.wtNi_feature="Ni"
        self.wtMn_feature="Mn"
        self.wtCu_feature="Cu"
        self.flux_n_cm2_sec_feature="flux_n_cm2_sec"
        self.fluence_n_cm2_feature="fluence_n_cm2"
        self.temp_C_feature="temperature_C"
        self.product_id_feature="Product ID"
        BaseCustomModel.__init__(self, **kwargs)
        return

    def predict(self, input_data):
        """EONY model,
            E.D. Eason, G.R. Odette, R.K. Nanstad, T. Yamamoto. 
            A physically-based correlation of irradiation-induced 
            transition temperature shifts for RPV steels. 
            Journal of Nuclear Materials 433 (2013) (1-3) 240.
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
        flux_n_cm2_sec = pdf[self.flux_n_cm2_sec_feature]
        # Get effective fluence
        lowflux_idx = pdf[flux_n_cm2_sec < 4.39e10].index
        pdf["eff_fluence"] = fluence_n_cm2
        pdf.loc[lowflux_idx, "eff_fluence"] = fluence_n_cm2[lowflux_idx] * np.power((4.39e10/flux_n_cm2_sec[lowflux_idx]) , 0.259)
        # Convert temp
        pdf["tempF"] = 1.8 * tempC + 32.0
        # Get product indices
        plates_idx = pdf[product == "P"].index
        forgings_idx = pdf[product == "F"].index
        srm_plates_idx = pdf[product == "SRM"].index #standard ref. matl plates
        welds_idx = pdf[product == "W"].index
        # Set Acol according to product type
        pdf.loc[forgings_idx, 'Acol'] = 1.140e-7
        pdf.loc[plates_idx, 'Acol'] = 1.561e-7
        pdf.loc[srm_plates_idx, 'Acol'] = 1.561e-7
        pdf.loc[welds_idx, 'Acol'] = 1.417e-7
        # Set Bcol according to product type
        ## Note that there is a distinction for non-CE mfg vessels and CE mfg
        ## vessels that is skipped here. 
        ## Here, all plates are assumed to be from CE mfg vessels,
        ## but Eason 2013 states there are 181 CE and 128 non-CE points
        pdf.loc[forgings_idx, 'Bcol'] = 102.3
        #pdf.loc[plates_idx, 'Bcol'] = 102.5 #assume non-CE mfg vessels
        pdf.loc[plates_idx, 'Bcol'] = 135.2 #assume CE mfg vessels
        pdf.loc[srm_plates_idx, 'Bcol'] = 128.2
        pdf.loc[welds_idx, 'Bcol'] = 155.0
        # Get MF term
        pdf['MF_temp'] = (1. - 0.001718 * pdf.tempF)
        pdf['MF_PMn'] = (1. + 6.13 * wtP * np.power(wtMn, 2.47))
        pdf['MF_fluence'] = np.sqrt(pdf.eff_fluence)
        pdf['MF'] = pdf.Acol * pdf.MF_temp * pdf.MF_PMn * pdf.MF_fluence
        # Get CRP term
        pdf['CRP_Ni'] = (1. + 3.77 * np.power(wtNi, 1.191))
        # Get Cu_effective
        high_Cu_idx = pdf[wtCu > 0.072].index
        pdf['Cu_eff'] = 0.0 #default to 0
        pdf['Max_Cu_eff'] = 0.301
        pdf.loc[welds_idx, 'Max_Cu_eff'] = 0.243
        pdf.loc[high_Cu_idx, 'Cu_eff'] = np.minimum(wtCu, pdf.Max_Cu_eff)
        # Get CRP f term
        hCu_hP_idx = pdf[(wtCu > 0.072) & (wtP > 0.008)].index
        hCu_lP_idx = pdf[(wtCu > 0.072) & (wtP <= 0.008)].index
        pdf['CRP_f'] = 0.0
        pdf.loc[hCu_lP_idx, 'CRP_f'] = np.power((pdf.Cu_eff-0.072), 0.668)
        pdf.loc[hCu_hP_idx, 'CRP_f'] = np.power( (pdf.Cu_eff - 0.072 + 1.359*(wtP - 0.008)), 0.668)
        # Get CRP g term
        pdf['CRP_g_inner'] = (np.log10(pdf.eff_fluence) + 1.139*pdf.Cu_eff - 0.448*wtNi - 18.120)/0.629
        pdf['CRP_g'] = 0.5 + 0.5*np.tanh(pdf.CRP_g_inner)
        pdf['CRP'] = pdf.Bcol * pdf.CRP_Ni * pdf.CRP_f * pdf.CRP_g
        # Get TTS in degrees Fahrenheit and then Celsius
        pdf['TTS_F'] = pdf.MF + pdf.CRP
        pdf['TTS'] = (pdf.TTS_F - 32.0) / 1.8
        # Get Cc according to product type
        c_welds = [0.55, 1.2e-3, -1.33e-6, 0.0]
        c_plates = [0.45, 1.945e-3, -5.496e-6, 8.473e-9]
        # Cc default to plates
        pdf.Cc = c_plates[0] + c_plates[1]*pdf.TTS + c_plates[2]*np.power(pdf.TTS, 2.0) + c_plates[3]*np.power(pdf.TTS, 3.0)
        # Cc for welds
        pdf.loc[welds_idx, 'Cc'] = c_welds[0] + c_welds[1]*pdf.TTS[welds_idx] + c_welds[2]*np.power(pdf.TTS[welds_idx], 2.0) + c_welds[3]*np.power(pdf.TTS[welds_idx], 3.0)
        # Get hardening (delta_sigma_y_MPa)
        pdf['delta_sigma_y_MPa'] = pdf.TTS / pdf.Cc
        pdf.to_csv(os.path.join(os.getcwd(),"EONY.csv"))
        return pdf['delta_sigma_y_MPa']
        
    def get_params(self):
        param_dict=dict()
        param_dict['wtP_feature'] = self.wtP_feature
        param_dict['wtNi_feature'] = self.wtNi_feature
        param_dict['wtMn_feature'] = self.wtMn_feature
        param_dict['wtCu_feature'] = self.wtCu_feature
        param_dict['flux_n_cm2_sec_feature'] = self.flux_n_cm2_sec_feature
        param_dict['fluence_n_cm2_feature'] = self.fluence_n_cm2_feature
        param_dict['temp_C_feature'] = self.temp_C_feature
        param_dict['product_id_feature'] = self.product_id_feature
        return param_dict
