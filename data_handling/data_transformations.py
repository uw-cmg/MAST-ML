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

def get_eony_model_tts(flux, time):
    """From Eason et al., J Nucl. Mater. 433 (2013) 240-254
        http://dx.doi.org/10.1016/j.jnucmat.2012.09.012
        Args:
            flux <float>: flux in n/cm^2/sec
            time <float>: time in seconds
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

    #Then get MF term

    #Then get CRF term

