#!/usr/bin/env python
##############
# get lists of id's to later filter out of collection
# corresponds for example to DBTT_Data7.csv to DBTT_Data8.csv differences
# TTM 2017-01-24
##############

import pymongo
import os
import sys

from pymongo import MongoClient

dbname="dbtt"
cname="ucsbivarplus" #later will be the collated collection

client = MongoClient('localhost', 27017)
db = client[dbname]

def get_LO_ids(verbose=1):
    id_list = list()
    results = db[cname].find({'Alloy':"LO"},{"_id":1,"Alloy":1})
    for result in results:
        id_list.append(result['_id'])
        if verbose > 0:
            print("%s,%s" % (result['_id'],result["Alloy"]))
    return id_list

def update_experimental_temperatures(verbose=1):
    """Update temperatures for a handful of experimental points
        whose reported temperature in the original data sheet 
        for ucsb ivar plus were incorrect.
            Alloys 6, 34, 35, 36, 37, 38 at fluence of 1.10e21 n/cm2, 
            flux of 2.30e14 n/cm2/sec should all be at 
            Temperature = 320 degrees C instead of 290 degrees C.
            Their CD temperature, however, remains at 290. 
    """



    return

def look_up_name_or_number(istr="",itype="name", verbose=1):
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

def get_duplicate_conditions(verbose=1):
    ddict=dict()
    id_list = list()
    pipeline=[
        {"$group": {"_id": {"Alloy":"$Alloy","flux_n_cm2_sec":"$flux_n_cm2_sec","fluence_n_cm2":"$fluence_n_cm2","temperature_C":"$temperature_C"},
                "count":{"$sum":1}}},
        #{"$group": {"_id": "Alloy":"$Alloy","$flux_n_cm2_sec","$fluence_n_cm2","count": {"$sum": 1}}},
        { "$match": {"count": {"$gt" : 1}}}
        ]
    results = db[cname].aggregate(pipeline)
    condlist=list()
    for result in results:
        if verbose > 0:
            print(result)
        condlist.append(result['_id'])
    return condlist

def get_duplicate_ids_to_remove(verbose=1):
    """Get duplicate IDs to remove.
        True duplicates, remove the second copy.
        Where delta_sigma_y differs, remove the duplicate where
            conditions are least like the rest of the set.
    """
    condlist = get_duplicate_conditions(verbose)
    id_list=list()
    for condition in condlist:
        if (verbose > 0):
            print(condition)
        records = db[cname].find(condition) #should be two of each
        comp_list=list()
        d_sig_list=list()
        for record in records:
            if (verbose > 1):
                print(record)
            comp_list.append(record["_id"])
            d_sig_list.append(record["delta_sigma_y_MPa"])
        if len(comp_list) > 2:
            raise ValueError("more than one duplicate for condition %s" % condition)
        if d_sig_list[0] == d_sig_list[1]:
            if verbose > 0:
                print("True duplicates.")
            id_list.append(comp_list[1]) #flag only the second duplicate
        else:
            if verbose > 0:
                print("Unknown. Check %s" % comp_list)
    if verbose > 0:
        for id_item in id_list:
            print(id_item)
    return id_list

if __name__=="__main__":
    get_LO_ids(verbose=1)
    get_duplicate_ids_to_remove()
    result = look_up_name_or_number(58,"number")
    print("Result: %s" % result)
    result = look_up_name_or_number("62W","name")
    print("Result: %s" % result)
