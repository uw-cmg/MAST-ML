#!/usr/bin/env python
###################
# Data transformations for use with dbtt database
# Tam Mayeshiba 2017-01-27
###################

import numpy as np
import pymongo
import os
import sys

from pymongo import MongoClient
from bson.objectid import ObjectId

dbname="dbtt"

client = MongoClient('localhost', 27017)
db = client[dbname]

def get_time_from_flux_and_fluence(flux, fluence):
    time = float(fluence) / float(flux)
    return time

def get_effective_fluence(flux, fluence, ref_flux, pvalue):
    """Get effective fluence.
        G.R. Odette, T. Yamamoto, and D. Klingensmith, 
        Phil. Mag. 85, 779-797 (2005)
        Args:
            flux <float> in n/cm2/sec
            fluence <float> in n/cm2
            ref_flux <float>: reference flux in n/cm2/sec
            pvalue <float>: p-value for exponent.
        Effective fluence takes the form:
        effective fluence ~= fluence * (ref_flux/flux)^p
        or equivalently:
        flux*effective time ~= flux*time * (ref_flux/flux)^p
    """
    effective_fluence = fluence * np.power((ref_flux/flux),pvalue)
    return effective_fluence

def get_log10(value):
    log10 = np.log10(value)
    return log10

def celsius_to_fahrenheit(temp_C):
    temp_F = 1.8 * (temp_C) + 32.0
    return temp_F

def get_eony_model_tts(flux, time, temp, product_id, wt_dict, ce_manuf=0):
    """From Eason et al., J Nucl. Mater. 433 (2013) 240-254
        http://dx.doi.org/10.1016/j.jnucmat.2012.09.012
        Args:
            flux <float>: flux in n/cm^2/sec
            time <float>: time in seconds
            temp <float>: irradiation temperature in Celsius. 
                        NOTE: temperature will be converted to Fahrenheit
            product_id <str>: P = plate, F = forging, W = weld, SRM = SRM
            wt_dict <dict>: dictionary of floats for weight percents of
                            elements.
            ce_manuf <int>: 1 = CE-manufactured vessel
                            0 = non-CE-manufactured vessel (default)
    """
    #TTS = MF term + CRP term
    
    #First get effective fluence
    ref_flux = 4.39e10 #n/cm^2/sec
    pval = 0.259
    fluence = flux * time
    if flux >= ref_flux: 
        eff_fluence = fluence #n/cm^2
    else:
        eff_fluence = get_effective_fluence(flux, fluence, ref_flux, pval)

    #Then get MF term (Equation 3)
    temp_irrad = celsius_to_fahrenheit(temp)
    
    coeffA = 0.0 #prefactor A
    product_id = product_id.upper()
    if product_id == "F":
        coeffA = 1.140E-7
    elif product_id == "P":
        coeffA = 1.561E-7
    elif product_id == "W":
        coeffA = 1.417E-7
    else:
        coeffA = 0.0 # for SRM? Need to check.

    if not "Mn" in wt_dict.keys():
        wt_Mn = 0.0
    else:
        wt_Mn = wt_dict["Mn"]
    if not "P" in wt_dict.keys():
        wt_P = 0.0
    else:
        wt_P = wt_dict["P"]
    
    MF_pc2 = (1 - (0.001718 * temp_irrad))
    MF_pc3 = (1 + 6.13 * wt_P * np.power(wt_Mn, 2.47))
    MF_pc4 = np.sqrt(eff_fluence)
    MF_term = coeffA * MF_pc2 * MF_pc3 * MF_pc4

    #Then get CRF term
    coeffB = 0.0
    if product_id == "F":
        coeffB = 102.3
    elif product_id == "P":
        if ce_manuf == 0:
            coeffB = 102.5
        elif ce_manuf == 1:
            coeffB = 135.2
        else:
            coeffB = 102.5
    elif product_id == "W":
        coeffB = 155.0
    elif product_id == "SRM":
        coeffB = 128.2
    else:
        coeffB = 0.0 #appropriate to set to zero? Need to check.

    if not "Cu" in wt_dict.keys():
        wt_Cu = 0.0
    else:
        wt_Cu = wt_dict["Cu"]
    if not "Ni" in wt_dict.keys():
        wt_Ni = 0.0
    else:
        wt_Ni = wt_dict["Ni"]

    eff_wt_Cu = 0.0





