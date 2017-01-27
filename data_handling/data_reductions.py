#!/usr/bin/env python
##############
# Flag some data IDs to be ignored.
# Update some experimental temperatures.
# Changes correspond for example to DBTT_Data7.csv to DBTT_Data8.csv differences
# TTM 2017-01-24
##############

import pymongo
import os
import sys

from pymongo import MongoClient
from bson.objectid import ObjectId

dbname="dbtt"

client = MongoClient('localhost', 27017)
db = client[dbname]

def get_alloy_removal_ids(cname, verbose=1):
    """Removal of certain alloys from database
    """
    alloylist=list()
    alloylist.append(41)
    id_list = list()
    reason_list=list()
    for alloynum in alloylist:
        aname = look_up_name_or_number(alloynum, "number")
        results = db[cname].find({'Alloy':aname},{"_id":1,"Alloy":1})
        for result in results:
            id_list.append(result['_id'])
            reason_list.append("Not considering alloy %s" % aname)
    if verbose > 0:
        for iidx in range(0, len(id_list)):
            print("%s: %s" % (id_list[iidx], reason_list[iidx]))
    return [id_list, reason_list]

def update_experimental_temperatures(cname, verbose=1):
    """Update temperatures for a handful of experimental points
        whose reported temperature in the original data sheet 
        for ucsb ivar plus were incorrect.
            Alloys 6, 34, 35, 36, 37, 38 at fluence of 1.10e21 n/cm2, 
            flux of 2.30e14 n/cm2/sec should all be at 
            Temperature = 320 degrees C instead of 290 degrees C.
            Their CD temperature, however, remains at 290. 
    """ 
    id_list=list()
    orig_temp_list=list()
    flux=2.30e14 #n/cm2/sec
    fluence=1.10e21 #n/cm2
    newtemp=320 #degrees C
    num_list=[6,34,35,36,37,38]
    for num in num_list:
        aname = look_up_name_or_number(num, "number")
        results = db[cname].find({"Alloy":aname,
                        "flux_n_cm2_sec":flux,
                        "fluence_n_cm2":fluence})
        for result in results:
            id_list.append(result["_id"])
            orig_temp_list.append(result["temperature_C"])
    #need to do actual temperature modification
    for modidx in range(0, len(id_list)):
        db[cname].update_one(
            {"_id":ObjectId(id_list[modidx])},
            {
                "$set":{
                    "original_reported_temperature_C": orig_temp_list[modidx],
                    "temperature_C": newtemp
                }
            }
        )
        if verbose > 0:
            print("%s: temperature updated" % id_list[modidx])
    return id_list

def look_up_name_or_number(istr="",itype="name", verbose=0):
    """Look up alloy name or number.
        Args:
            istr <str or int>: input value
            itype <str>: input type: alloy "name" or "number"
        Returns:
            <str or int>: alloy number or name
    """
    if itype == "name":
        ilookup = "alloy_name"
        oreturn = "alloy_number"
    elif itype == "number":
        ilookup = "alloy_number"
        oreturn = "alloy_name"
    else:
        print("Invalid entry: %s should be 'name' or 'number'" % itype)
        return None
    results = db['alloykey'].find({ilookup:istr})
    olist = list()
    for result in results:
        if verbose > 0:
            print(result)
            print(result[oreturn])
        olist.append(result[oreturn])
    if len(olist) == 1:
        return olist[0]
    return olist

def get_duplicate_conditions(cname, verbose=0):
    ddict=dict()
    id_list = list()
    pipeline=[
        {"$group": {"_id": {"Alloy":"$Alloy","flux_n_cm2_sec":"$flux_n_cm2_sec","fluence_n_cm2":"$fluence_n_cm2","temperature_C":"$temperature_C"},
                "count":{"$sum":1}}},
        { "$match": {"count": {"$gt" : 1}}}
        ]
    results = db[cname].aggregate(pipeline)
    condlist=list()
    for result in results:
        if verbose > 0:
            print(result)
        condlist.append(result['_id'])
    return condlist

def get_duplicate_ids_to_remove(cname, dupcheck="delta_sigma_y_MPa", verbose=1):
    """Get duplicate IDs to remove.
        True duplicates, remove the second copy.
        Where delta_sigma_y differs, remove the duplicate where
            conditions are least like the rest of the set.
            For eight such pairs, this happens to mean removing the
            smaller delta_sigma_y.
    """
    condlist = get_duplicate_conditions(cname)
    id_list=list()
    reason_list=list()
    for condition in condlist:
        if (verbose > 1):
            print(condition)
        rlist = list(db[cname].find(condition)) #should be two of each
        if not (len(rlist) == 2):
            raise ValueError("not exactly two duplicates for condition %s" % condition)
        if verbose > 1:
            print("Duplicate records for condition:")
            print(rlist[0])
            print(rlist[1])
        rval0 = rlist[0][dupcheck]
        rval1 = rlist[1][dupcheck]
        if rval0 == rval1:
            if verbose > 1:
                print("True duplicates.")
            id_list.append(rlist[1]['_id'])
            reason_list.append("Duplicate alloy, conditions, and delta_sigma_y.")
        else:
            if verbose > 1:
                print ("Not true duplicates.")
            if rval0 < rval1:
                id_list.append(rlist[0]['_id'])
            else:
                id_list.append(rlist[1]['_id'])
            reason_list.append("Duplicate alloy and conditions; remove lower delta_sigma_y.")
    if verbose > 0:
        for iidx in range(0, len(id_list)):
            print("%s: %s" % (id_list[iidx], reason_list[iidx]))
    return [id_list, reason_list]

def flag_for_ignore(cname, id_list, reason_list, verbose=1):
    for fidx in range(0, len(id_list)):
        flagid = id_list[fidx]
        db[cname].update_one(
            {"_id":ObjectId(flagid)},
            {
                "$set":{
                    "ignore":1,
                    "ignore_reason":reason_list[fidx]
                }
            }
        )
        if verbose > 0:
            print("Updated record %s" % flagid)
    return

def main_exptivar(cname="ucsbivarplus",verbose=1):
    [id_list, reason_list] = get_alloy_removal_ids(cname)
    flag_for_ignore(cname, id_list, reason_list)
    print(len(id_list))
    [id_list, reason_list] = get_duplicate_ids_to_remove(cname)
    flag_for_ignore(cname, id_list, reason_list)
    print(len(id_list))
    update_experimental_temperatures(cname)
    return

def main_cdivar(cname="cdivar2017",verbose=1):
    [id_list, reason_list] = get_alloy_removal_ids(cname)
    flag_for_ignore(cname, id_list, reason_list)
    print(len(id_list))
    [id_list, reason_list] = get_duplicate_ids_to_remove(cname,"CD_delta_sigma_y_MPa") 
    flag_for_ignore(cname, id_list, reason_list)
    print(len(id_list))
    #   #the same number of duplicate ids should be removed as in the
    #   #experimental case; where CD was run, both the true duplicates and
    #   #the duplicates with differing delta_sigma_y would register as
    #   #true duplicates, and one ID from each pair will be removed.
    return

if __name__=="__main__":
    main_exptivar()
    main_cdivar("cdivar2017")
    main_cdivar("cdivar2016")
