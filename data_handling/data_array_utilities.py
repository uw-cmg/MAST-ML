#!/usr/bin/env python
##############
# Utilities for handling parsed data from data_parser
# TTM 2017-03-31
##############

import pymongo
import os
import sys
import subprocess 
import time
from bson.objectid import ObjectId
import data_handling.data_transformations as dtf
import data_handling.alloy_property_utilities as apu
import numpy as np
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
    add_log10_of_a_field(db, newcname, newfield)
    #add_minmax_normalization_of_a_field(db, newcname, "log(%s)" % newfield)
    return


def add_log10_of_a_field(pdatalist=list(), fieldname=""):
    """Add the base-10 logarithm of a field as a new field
    """
    newfield="log(%s)" % origfield
    myfunc = getattr(dtf,"get_log10")
    records = db[newcname].find()
    for record in records:
        fieldval = myfunc(record[origfield])
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{newfield:fieldval}}
            )
        if verbose > 0:
            print("Updated record %s with %s %3.3f." % (record["_id"],newfield,fieldval))
    return

def find_field_number(pdatalist=list(), fieldname=""):
    fieldnumbers = list()
    return fieldnumbers

def add_minmax_normalization_of_a_field(db, newcname, origfield, setmin=None, setmax=None, verbose=0, collectionlist=list()):
    """Add the normalization of a field based on the min and max values
        Normalization is given as X_new = (X - X_min)/(X_max - X_min)
        For elemental compositions, max atomic percent is taken as 1.717 At%
            for Mn.
        Args:
            db
            newcname
            origfield <str>: Original field name
            setmin <float or None>: Set minimum directly
            setmax <float or None>: Set maximum directly
            verbose <int>: 0 - silent (default)
                           1 - verbose
            collectionlist <list of str>: list of multiple
                            collection names for normalization
                            over multiple collections
                            empty list only works on collection newcname
    """
    newfield="N(%s)" % origfield

    if len(collectionlist) == 0:
        collectionlist.append(newcname)

    cmins = list()
    cmaxes = list()
    if (setmin == None) or (setmax == None):
        for collection in collectionlist:
            cminval = None
            cmaxval = None
            if (verbose > 0):
                print("Using collection %s" % collection)
            agname = "nonignore_%s" % collection
            db[collection].aggregate([
                        {"$match":{"ignore":{"$ne":1}}},
                        {"$group":{"_id":None,
                            "fieldminval":{"$min":"$%s" % origfield},
                            "fieldmaxval":{"$max":"$%s" % origfield}}},
                        {"$out":agname}
                        ]) #get nonignore records
            for record in db[agname].find(): #one record from aggregation
                cminval = record['fieldminval']
                cmaxval = record['fieldmaxval']
                if verbose > 0:
                    print(record)
            if not (cminval == None):
                cmins.append(cminval)
            if not (cmaxval == None):
                cmaxes.append(cmaxval)
            db[agname].drop()
    if setmin == None:    
        minval = min(cmins)
    else:
        minval = setmin
    if setmax == None:
        maxval = max(cmaxes)
    else:
        maxval = setmax
    print("Min: %3.3f" % minval)
    print("Max: %3.3f" % maxval)
    records = db[newcname].find()
    for record in records:
        if (minval == maxval): #data is flat
            fieldval = 0.0
        else:
            fieldval = ( record[origfield] - minval ) / (maxval - minval)
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{newfield:fieldval}}
            )
        if verbose > 0:
            print("Updated record %s with %s %3.3f." % (record["_id"],newfield,fieldval))
    return
def printtest(pdatalist=list(), header=""):
    for dataset in pdatalist:
        print("%s" % header, dataset)
    return

def handler(pdatalist=list(), methodname="", **kwargs):
    import parsed_data_utilities
    if hasattr(parsed_data_utilities, methodname):
        method_to_call = getattr(parsed_data_utilities, methodname)
        result = method_to_call(pdatalist, **kwargs)
    return
if __name__=="__main__":
    testdata1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(testdata1)
    testdata2 = np.array([[1,3,5],[2,4,6],[3,5,7],[4,6,8]])
    kwargs=dict()
    kwargs['header'] = "Dataset: "
    handler(list([testdata1,testdata2]),"printtest", **kwargs)
    sys.exit()
