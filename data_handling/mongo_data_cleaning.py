#!/usr/bin/env python
##############
# Data cleaning
#   - update some experimental temperatures
#   - standardize names where other names have been used
#   - keep only one of each true duplicate pair
#   - keep results from duplicate-condition experiments where conditions
#     are similar to the rest of the dataset
# Changes correspond for example to DBTT_Data7.csv to DBTT_Data8.csv differences
# TTM 2017-01-24
##############

import pymongo
import os
import sys
import data_handling.alloy_property_utilities as apu
import data_handling.data_transformations as dtr

from bson.objectid import ObjectId

def get_field_condition_to_remove(db, cname, fieldname, fieldval, verbose=1):
    """Removal of certain field condition
        Args:
            db <mongo DB>: Mongo client object (database)
            cname <str>: collection name
            fieldname <str>: field name
            fieldval <usually float>: field value
    """
    id_list = list()
    reason_list=list()
    records = db[cname].find({fieldname: fieldval})
    for record in records:
        id_list.append(record['_id'])
        reason_list.append("Not considering %s value %s" % (fieldname,fieldval))
    if verbose > 0:
        for iidx in range(0, len(id_list)):
            print("%s: %s" % (id_list[iidx], reason_list[iidx]))
    return [id_list, reason_list]


def flag_for_ignore(db, cname, id_list, reason_list, verbose=1):
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





if __name__=="__main__":
    print("Use from DataImportAndExport.py. Exiting.")
    sys.exit()
    from pymongo import MongoClient
    dbname="dbtt"
    client = MongoClient('localhost', 27017)
    db = client[dbname]
