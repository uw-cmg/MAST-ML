#!/usr/bin/env python
##############
# Collect alloy property utilities here
# TTM 2017-02-03
##############

import pymongo
import os
import sys
import time
import data_handling.percent_converter as pconv

#Info specific to the alloys database
cname_alloy = "alloys"
alias_cols = ["alias_1"] #only one column right now

def get_standardized_alloy_name(db, alloy, verbose=0):
    """Standardize alloy names
        CD LWR data has some names that do not match.
        Args:
            alloy <str>: Name to look up
    """
    std_names = db[cname_alloy].distinct("Alloy")
    if alloy in std_names:
        return alloy
    for alias_col in alias_cols:
        results = db[cname_alloy].find({alias_col:alloy})
        for result in results: #only finds first match
            std_name = result["Alloy"]
            return std_name
    return None

def get_atomic_percents(db, alloy, verbose=0):
    """Ignores percentages from Mo and Cr.
        Assumes balance is Fe.
        Args:
            alloy <str>: Alloy name
    """
    elemlist=["Cu","Ni","Mn","P","Si","C"]
    results = db[cname_alloy].find({"Alloy":alloy})
    for result in results:
        if verbose > 0:
            print("Found alloy %s" % alloy)
        compdict=dict()
        for elem in elemlist:
            try:
                wt = float(result['wt_percent_%s' % elem])
            except (ValueError,TypeError):
                wt = 0.0
            compdict[elem] = wt
        compstr=""
        for elem in elemlist:
            compstr += "%s %s," % (elem, compdict[elem])
        compstr = compstr[:-1] #remove last comma
        outdict = pconv.main(compstr,"weight",verbose)
        if (verbose > 0):
            print("output dict: %s" % outdict)
        return outdict
    raise ValueError("Could not get composition for alloy %s" % alloy)
    return

def look_up_name_or_number(db, istr="",itype="name", verbose=0):
    """Look up alloy name or number.
        Args:
            istr <str or int>: input value
            itype <str>: input type: alloy "name" or "number"
        Returns:
            <str or int>: alloy number or name
    """
    if itype == "name":
        ilookup = "Alloy"
        oreturn = "alloy_number"
    elif itype == "number":
        ilookup = "alloy_number"
        oreturn = "Alloy"
    else:
        print("Invalid entry: %s should be 'name' or 'number'" % itype)
        return None
    results = db[cname_alloy].find({ilookup:istr})
    olist = list()
    for result in results:
        if verbose > 0:
            print(result)
            print(result[oreturn])
        olist.append(result[oreturn])
    if len(olist) == 1:
        return olist[0]
    return olist

if __name__=="__main__":
    print("Use from DataImportAndExport.py. Exiting.")
    sys.exit()
