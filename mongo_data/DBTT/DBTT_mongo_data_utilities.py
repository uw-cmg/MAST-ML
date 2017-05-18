#!/usr/bin/env python
##############
# Create analysis spreadsheet.
# Pull values from databases and add additional columns.
# TTM 2017-01-27
##############

import pymongo
import os
import sys
import subprocess 
import time
from bson.objectid import ObjectId
import mongo_data.DBTT.DBTT_data_transformations as dtf
import mongo_data.DBTT.alloy_property_utilities as apu
import mongo_data.mongo_data_utilites as mdu

def add_time_field(db, newcname, verbose=0):
    myfunc = getattr(dtf,"get_time_from_flux_and_fluence")
    records = db[newcname].find()
    for record in records:
        fieldval = myfunc(record["flux_n_cm2_sec"],record["fluence_n_cm2"])
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{"time_sec":fieldval}}
            )
        if verbose > 0:
            print("Updated record %s with time %3f sec." % (record["_id"],fieldval))
    return

def add_alloy_number_field(db, newcname, verbose=0):
    myfunc = getattr(apu,"look_up_name_or_number")
    records = db[newcname].find()
    for record in records:
        fieldval = myfunc(db, record["Alloy"], "name", verbose)
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{"alloy_number":fieldval}}
            )
        if verbose > 0:
            print("Updated record %s with alloy number %i." % (record["_id"],fieldval))
    return


def add_atomic_percent_field(db, newcname, verbose=0):
    """Ignores percentages from Mo and Cr.
        Assumes balance is Fe.
    """
    elemlist=["Cu","Ni","Mn","P","Si","C"]
    records = db[newcname].find()
    for record in records:
        alloy = record["Alloy"]
        outdict = apu.get_atomic_percents(db, alloy, verbose)
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":
                {"at_percent_Cu":outdict["Cu"]['perc_out'],
                "at_percent_Ni":outdict["Ni"]['perc_out'],
                "at_percent_Mn":outdict["Mn"]['perc_out'],
                "at_percent_P":outdict["P"]['perc_out'],
                "at_percent_Si":outdict["Si"]['perc_out'],
                "at_percent_C":outdict["C"]['perc_out']}
                }
            )
        if verbose > 0:
            print("Updated record %s with %s." % (record["_id"],outdict))
    for elem in elemlist:
        mdu.add_minmax_normalization_of_a_field(db, newcname, "at_percent_%s" % elem, 0.0, 1.717)
    return

def add_effective_fluence_field(db, newcname, verbose=0):
    """
        Calculated by fluence*(ref_flux/flux)^p where ref_flux = 3e10 n/cm^2/s and p=0.26
        IVAR has ref_flux of 3e11 n/cm^2/sec (Odette 2005)
        However, LWR conditions have lower flux, so use 3e10 n/cm^2/sec.
    """
    ref_flux=3.0e10 #n/cm^2/sec
    pvalue = 0.26
    myfunc = getattr(dtf,"get_effective_fluence")
    records = db[newcname].find()
    for record in records:
        fieldval = myfunc(flux=record["flux_n_cm2_sec"],
                            fluence=record["fluence_n_cm2"],
                            ref_flux=ref_flux,
                            pvalue=pvalue)
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{"effective_fluence_n_cm2":fieldval}}
            )
        if verbose > 0:
            print("Updated record %s with effective fluence %3.2e n/cm2." % (record["_id"],fieldval))
    return

def add_generic_effective_fluence_field(db, newcname, ref_flux=3e10, pvalue=0.26, verbose=0):
    """
        Calculated by fluence*(ref_flux/flux)^p 
        IVAR has ref_flux of 3e11 n/cm^2/sec (Odette 2005)
        However, LWR conditions have lower flux, so use 3e10 n/cm^2/sec.
        Args:
            ref_flux <float>: reference flux in n/cm^2/sec
            pvalue <float>: p value
        Also adds a log10 field and a log10 min-max normalized field
    """
    myfunc = getattr(dtf,"get_effective_fluence")
    records = db[newcname].find()
    pvalstr = "%i" % (pvalue*100.0)
    newfield = "eff fl 100p=%s" % pvalstr
    for record in records:
        fieldval = myfunc(flux=record["flux_n_cm2_sec"],
                            fluence=record["fluence_n_cm2"],
                            ref_flux=ref_flux,
                            pvalue=pvalue)
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{newfield:fieldval}}
            )
        if verbose > 0:
            print("Updated record %s with %s %3.2e n/cm2." % (record["_id"],newfield,fieldval))
    mdu.add_log10_of_a_field(db, newcname, newfield)
    #add_minmax_normalization_of_a_field(db, newcname, "log(%s)" % newfield)
    return


def add_product_type_columns(db, newcname, verbose=0):
    records = db[newcname].find()
    for record in records:
        pid = record['product_id'].strip().upper()
        plate=0
        weld=0
        SRM=0
        forging=0
        if pid == "P":
            plate=1
        elif pid == "W":
            weld=1
        elif pid == "SRM":
            SRM=1
        elif pid == "F":
            forging=1
        else:
            raise ValueError("Category for product id %s not found." % pid)
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":
                {"isPlate":plate,
                "isWeld":weld,
                "isForging":forging,
                "isSRM":SRM}
            }
            )
        if verbose > 0:
            print("Updated record %s with %s %3.3f." % (record["_id"],newfield,fieldval))

    return

def add_eony_field(db, newcname, verbose=0):
    """Add the EONY 2013 model
    """
    records = db[newcname].find()
    for record in records:
        flux = record["flux_n_cm2_sec"]
        time = record["time_sec"]
        temp = record["temperature_C"]
        product_id = record["product_id"]
        wt_dict=dict()
        for element in ["Mn","Ni","Cu","P","Si","C"]:
            wt_dict[element] = record["wt_percent_%s" % element]
        ce_manuf= 0 #how to determine?
        eony_tts = dtf.get_eony_model_tts(flux, time, temp, product_id,
                                            wt_dict, ce_manuf, verbose=1)
        eony_delta_sigma_y = dtf.tts_to_delta_sigma_y(eony_tts, "F", product_id, verbose=1)
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{"EONY_delta_sigma_y":eony_delta_sigma_y}}
            )
        if verbose > 0:
            print("Alloy %s flux %s fluence %s" % (record["Alloy"],flux,record["fluence_n_cm2"]))
            print("Updated record %s with %s %3.3f." % (record["_id"],"EONY_delta_sigma_y", eony_delta_sigma_y))
    return

if __name__=="__main__":
    print("Use from DataImportAndExport.py. Exiting.")
    sys.exit()
