#!/usr/bin/env python
##############
# Collect alloy property utilities here
# TTM 2017-02-03
##############

import pymongo
import os
import sys
import time
import percent_converter

#Info specific to the alloys database
cname_alloy = "alloys"
alias_cols = ["alias_1"] #only one column right now

def standardize_alloy_names(db, newcname, verbose=0):
    """Standardize alloy names
        CD LWR data has some names that do not match.
    """
    std_names = db[cname_alloy].distinct("Alloy")
    records = db[newcname].find()
    for record in records:
        alloy = record["Alloy"]
        if not (alloy in std_names):
            found = 0
            for alias_col in alias_cols:
                if found == 1:
                    break
                results = db[cname_alloy].find({alias_col:alloy})
                for result in results: #only finds first match
                    std_name = result["Alloy"]
                    record["old_Alloy_alias"] = alloy
                    record["Alloy"] = std_name
                    found = 1
                    if verbose > 0:
                        print("Adjusted id %s name %s to %s." % (record["_id"], alloy, std_name))
                    break
    return

def add_atomic_percent_field(db, newcname, verbose=0):
    """Ignores percentages from Mo and Cr.
        Assumes balance is Fe.
    """
    elemlist=["Cu","Ni","Mn","P","Si","C"]
    records = db[newcname].find()
    for record in records:
        compdict=dict()
        for elem in elemlist:
            try:
                wt = float(record['wt_percent_%s' % elem])
            except (ValueError,TypeError):
                wt = 0.0
            compdict[elem] = wt
        compstr=""
        for elem in elemlist:
            compstr += "%s %s," % (elem, compdict[elem])
        compstr = compstr[:-1] #remove last comma
        outdict = percent_converter.main(compstr,"weight",0)
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
        add_minmax_normalization_of_a_field(db, newcname, "at_percent_%s" % elem, 0.0, 1.717)
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
    results = db['alloys'].find({ilookup:istr})
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
    print("Warning: use through DataImportAndExport.py, not on its own")
    from pymongo import MongoClient
    dbname="dbtt"
    client = MongoClient('localhost', 27017)
    db = client[dbname]
