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

def get_duplicates_without_matching_delta_sigma(verbose=1):
    ddict=dict()
    id_list = list()
    pipeline=[
        {"$group": {"_id": {"Alloy":"$Alloy","flux_n_cm2_sec":"$flux_n_cm2_sec","fluence_n_cm2":"$fluence_n_cm2"},
                "count":{"$sum":1}}},
        #{"$group": {"_id": "Alloy":"$Alloy","$flux_n_cm2_sec","$fluence_n_cm2","count": {"$sum": 1}}},
        { "$match": {"count": {"$gt" : 1}}}
        ]
    mylist=list(db[cname].aggregate(pipeline))
    print(mylist)

    return id_list

if __name__=="__main__":
    get_LO_ids(verbose=1)
    get_duplicates_without_matching_delta_sigma()
