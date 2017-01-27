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
from pymongo import MongoClient
from bson.objectid import ObjectId
import data_transformations as dtf
dbname="dbtt"

client = MongoClient('localhost', 27017)
db = client[dbname]

def get_nonignore_records(cname):
    results = db[cname].find({"ignore":{"$ne":1}})
    return results

def transfer_nonignore_records(fromcname, newcname, verbose=1):
    nonignore_records = get_nonignore_records(fromcname)
    nict = 0
    for nirecord in nonignore_records:
        db[newcname].insert(nirecord)
        nict = nict + 1
    if (verbose > 0):
        print("Transferred %i records." % nict)
    return

def match_and_add_records(newcname, records, matchlist=list(), matchas=list(), transferlist=list(), transferas=list(), verbose=1):
    """Match records to the collection and add certain fields
        Args:
            newcname <str>: Collection name to which to add records
            records <Cursor>: records to match
            matchlist <list of str>: fields on which to match values
            matchas <list of str>: corresponding match fields in the new collection
            transferlist <list of str>: field names to transfer
            transferas <list of str>: field names to rename upon transfer,
                    in the same order listed in transferlist
    """
    uct=0
    if len(matchlist) == 0:
        raise ValueError("Matchlist cannot be empty. Must match on some fields.")
    for record in records:
        matchdict=dict()
        for midx in range(0, len(matchlist)):
            matchdict[matchas[midx]] = record[matchlist[midx]]
        setdict=dict()
        setdict["$set"]=dict()
        for tidx in range(0, len(transferlist)):
            setdict["$set"][transferas[tidx]] = record[transferlist[tidx]]
        updated = db[newcname].update(matchdict, setdict)
        if updated['updatedExisting'] == True:
            uct = uct + 1
        if (verbose > 1):
            print("Updated: %s" % updated)
    if (verbose > 0):
        print("Total %i records updated." % uct)
    return

def list_all_fields(cname, verbose=0):
    """
        List all fields in a collection. Make more sophisticated later.
    """
    fieldlist=list()
    records = db[cname].find()
    for record in records:
        for key in record.keys():
            if not key in fieldlist:
                fieldlist.append(key)
    if verbose > 0:
        for field in fieldlist:
            print(field)
    return fieldlist

def export_spreadsheet(newcname="", fieldlist=list()):
    if len(fieldlist) == 0:
        fieldlist=list_all_fields(newcname)
    fieldstr=""
    for field in fieldlist:
        fieldstr = fieldstr + field + ","
    estr = "mongoexport"
    estr += " --db=%s" % dbname
    estr += " --collection=%s" % newcname
    estr += " --out=%s_%s.csv" % (newcname, time.strftime("%Y%m%d_%H%M%S"))
    estr += " --type=csv"
    estr += ' --fields="%s"' % fieldstr
    eproc = subprocess.Popen(estr, shell=True,
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
    print(eproc.communicate())
    eproc.wait()
    return

def add_time_field(newcname, verbose=0):
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


def main_ivar(newcname=""):
    transfer_nonignore_records("ucsbivarplus",newcname)
    print("Transferred experimental IVAR records.")
    cd_records = get_nonignore_records("cdivar2016") #change to 2017 later
    match_and_add_records(newcname, cd_records, 
        matchlist=["Alloy","flux_n_cm2_sec","fluence_n_cm2","temperature_C"],
        matchas=["Alloy","flux_n_cm2_sec","fluence_n_cm2","temperature_C"],
        transferlist= ["temperature_C","CD_delta_sigma_y_MPa"],
        transferas = ["CD_temperature_C","CD_delta_sigma_y_MPa"])
    print("Updated with CD IVAR temperature matches.")
    cd_records.rewind()
    match_and_add_records(newcname, cd_records, 
        matchlist=["Alloy","flux_n_cm2_sec","fluence_n_cm2","temperature_C"],
        matchas=["Alloy","flux_n_cm2_sec","fluence_n_cm2","original_reported_temperature_C"],
        transferlist= ["temperature_C","CD_delta_sigma_y_MPa"],
        transferas = ["CD_temperature_C","CD_delta_sigma_y_MPa"])
    print("Updated with CD IVAR temperature mismatches.")
    export_spreadsheet(newcname)
    return

def main_addfields(newcname=""):
    add_time_field(newcname)
    return

if __name__=="__main__":
    if len(sys.argv) > 1:
        newcname = sys.argv[1]
    else:
        newcname = "test_1"
    #main_ivar(newcname)
    main_addfields(newcname)
    #export_spreadsheet(newcname)
