#!/usr/bin/env python
###################
# Data transformations for use with dbtt database
# Tam Mayeshiba 2017-01-27
###################

import numpy as np
import os
import sys

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

def fahrenheit_to_celsius(temp_F):
    temp_C = (temp_F - 32.0) / 1.8
    return temp_C

def delta_fahrenheit_to_celsius(delta_temp_F):
    delta_temp_C = delta_temp_F / 1.8
    return delta_temp_C

def get_eony_model_tts(flux, time, temp, product_id, wt_dict, ce_manuf=0, 
                        verbose=0):
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
            verbose <int>: 1 - verbose, 0 - silent
    """
    #TTS = MF term + CRP term
    
    #First get effective fluence
    ref_flux = 4.39E10 #n/cm^2/sec
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
    
    MF_pc2 = (1.0 - (0.001718 * temp_irrad))
    MF_pc3 = (1.0 + 6.13 * wt_P * np.power(wt_Mn, 2.47))
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
    
    max_Cu_eff = 0.301
    if product_id == "W": #how to tell if it is a Linde 80 weld?
        if wt_Ni > 0.50:
            max_Cu_eff = 0.243

    eff_wt_Cu = 0.0
    if wt_Cu <= 0.072:
        eff_wt_Cu = 0.0
    else:
        eff_wt_Cu = min(wt_Cu, max_Cu_eff)
    if (verbose > 0):
        print("Eff_wt_Cu: %3.3f" % eff_wt_Cu)

    f_fn = 0.0
    f_base=0.0
    if wt_Cu <= 0.072: 
        f_fn = 0.0
    else:
        if wt_P <= 0.008:
            f_base = eff_wt_Cu - 0.072
        else:
            f_base = eff_wt_Cu - 0.072 + (1.359*(wt_P-0.008)) 
        f_fn = np.power( f_base , 0.668)
    
    tanh_numerator = np.log10(eff_fluence)+ 1.139*eff_wt_Cu -0.448*wt_Ni -18.120
    g_fn = 0.5 + 0.5 * np.tanh(tanh_numerator / 0.629)

    CRP_term = coeffB*(1.0 + 3.77*np.power(wt_Ni, 1.191)) * f_fn * g_fn
    TTS = MF_term + CRP_term #a difference of degrees Fahrenheit at 30 ft-lbs.
    if (verbose > 0):
        print("mf 2, 3, 4: %3.3f, %3.3f, %3.3f" % (MF_pc2, MF_pc3, MF_pc4))
        print("eff fluence: %3.3e" % eff_fluence)
        print("MF_term: %3.3f" % MF_term)
        print("CRP_term: %3.3f" % CRP_term)
        print("f_base: %3.3f" % f_base)
        print("f_fn: %3.3f" % f_fn)
        print("g_fn: %3.3f" % g_fn)
        print("TTS: %3.3f" % TTS)
    return TTS

def tts_to_delta_sigma_y(tts, degree_type="F", product_id="", verbose=0):
    """Transform transition temperature shift to change in yield stress.
        Primarily from Eqn. 6-3 and Eqn. 6-4 in 
        Eason et al., A Physically Based Correlation of Irradiation-Induced 
            Transition Temperature Shifts for RPV Steels,
            ORNL TM-2006/530
        Args:
            tts <float>: transition temperature shift
            degree_type <str>: F = Fahrenheit
                               C = Celsius
            product_id <str>: P = plate, W = weld, F = forging, SRM=SRM
            verbose <int>: 1 - verbose, 0 - silent
    """
    cw_coeffs = [0.55, 1.2E-3,   -1.33E-6,  0.00]
    cp_coeffs = [0.45, 1.945E-3, -5.496E-6, 8.473E-9]
    if product_id == "P":
        coeffs = list(cp_coeffs)
    else:
        coeffs = list(cw_coeffs) #not sure about forgings
    
    if degree_type.upper() == "F":
        tts_C = delta_fahrenheit_to_celsius(tts)
    else:
        tts_C = tts
    
    C_c = 0.0
    cstr = ""
    for cidx in range(0, len(coeffs)):
        newterm = coeffs[cidx] * np.power(tts_C, float(cidx))
        C_c = C_c + newterm
        cstr = cstr + "%3.3e +" % newterm
    cstr = "%3.3e: " % C_c + cstr[:-1]
    delta_sigma_y = tts_C / C_c #in MPa
    if (verbose > 0):
        print("TTS_degC: %3.3f" % tts_C)
        print("C_c: %s" % cstr)
        print("delta_sigma_Y (MPa): %3.3f" % delta_sigma_y)
    return delta_sigma_y

