#!/usr/bin/env python
###################
# Data import and export for the dbtt database
# Tam Mayeshiba 2017-02-03
#
# This script is intended to be run on a local computer in order to
# generate .csv files.
#
# Prerequisites:
# 1. Must have mongodb installed and running.
#    Visit https://docs.mongodb.com/manual/administration/install-community/
# 2. Must have starting import csv files available.
#
###################
import numpy as np
import pymongo
import os
import sys
import traceback
import subprocess
import data_cleaning as dclean
import create_analysis_spreadsheets as cas

from pymongo import MongoClient
from bson.objectid import ObjectId

#Set up paths and names
dbpath = "../../../data/DBTT_mongo"
importpath = "../../../data/DBTT_mongo/imports_201702"
db_base="dbtt"
db = "" #will be set by script
ivarpluscname = "ivarplus_data"
#The following csv files should be in $importpath
cbasic=dict()
cbasic["alloys"] = "alloy_properties.csv"
cbasic["cd_ivar_2017"]="CD_IVAR_Hardening_2017-1_with_ivar_columns_reduced.csv"
cbasic["cd_ivar_2016"]="CD_IVAR_Hardening_clean_2016.csv"
cbasic["cd_lwr_2017"]="lwr_cd_2017_reduced_for_import.csv"
cbasic["ucsb_ivar_and_ivarplus"]="ucsb_ivar_and_ivarplus.csv"

def get_mongo_client():
    """Check connection and get mongo client
        Based on http://stackoverflow.com/questions/30539183/how-do-you-check-if-the-client-for-a-mongodb-instance-is-valid
    """
    timeout = 500 # milliseconds
    try:
        client = MongoClient('localhost', 27017, serverSelectionTimeoutMS = timeout)
        client.server_info() # Force connection check
    except pymongo.errors.ServerSelectionTimeoutError as err:
        traceback.print_exc()
        print(err)
        print("")
        print("Check to see if mongodb is actually running. Exiting.")
        print("")
        sys.exit(-1)
    return client

def get_unique_name(client, db_base, nmax=100):
    """Get a unique database name.
    """
    dbs = client.database_names()
    for idx in range(0, nmax):
        name_try = db_base + "_" + str(idx).zfill(2)
        if not (name_try in dbs):
            print("Using database name: %s" % name_try)
            return name_try
    print("Must drop or rename some databases.")
    print("Maximum of %i databases named with %s are present." % (nmax,db_base))
    print("Exiting.")
    sys.exit(-1)
    return None

def import_initial_collections(db, cbasic):
    """Import initial collections for use in creating specialized collections.
    """
    for cname in cbasic.keys():
        print("Attempting import for %s" % cname)
        fullpath = os.path.join(importpath, cbasic[cname])
        istr = "mongoimport --file=%s --headerline --db=%s --collection=%s --type=csv" % (fullpath, db.name, cname)
        iproc = subprocess.Popen(istr,shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
        iproc.wait()
        print(iproc.communicate())
    print("Collections created:")
    for cname in db.collection_names():
        print(cname)
    return

def clean_ivar_basic(db, cname, verbose=1):
    [id_list, reason_list] = dclean.get_alloy_removal_ids(db, cname)
    dclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    [id_list, reason_list] = dclean.get_duplicate_ids_to_remove(db, cname)
    dclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    dclean.update_experimental_temperatures(db, cname)
    return
def create_ivar_for_gkrr_hyperparam(db, cname, fromcname, verbose=1):
    tempname = "%s_temp" % cname
    cas.transfer_nonignore_records(db, fromcname, tempname, verbose)
    [id_list, reason_list] = dclean.get_field_condition_to_remove(db, tempname,
                            "dataset","IVAR+")
    dclean.flag_for_ignore(db, tempname, id_list, reason_list)
    print(len(id_list))
    cas.transfer_nonignore_records(db, tempname, cname, verbose)
    cas.export_spreadsheet(db, cname, "../../../data/DBTT_mongo/data_exports/")
    return
    

def create_spreadsheets(db):
    #IVAR
    ivarcname="ucsb_ivar_and_ivarplus"
    cas.main_ivar(db, ivarcname)
    cas.main_addfields(db, ivarcname)
    cas.export_spreadsheet(ivarcname, "../../../data/DBTT_mongo/data_exports/")
    #LWR
    lwrcname = "cd_lwr_2017"
    cas.main_lwr(lwrcname)
    cas.lwr_addfields(lwrcname)
    cas.export_spreadsheet(lwrcname, "../../../data/DBTT_mongo/data_exports/")
    return

if __name__ == "__main__":
    client = get_mongo_client()
    dbname = get_unique_name(client, db_base)
    db = client[dbname]
    import_initial_collections(db, cbasic)
    clean_ivar_basic(db, "ucsb_ivar_and_ivarplus")
    create_ivar_for_gkrr_hyperparam(db, "ucsb_ivar_hyperparam","ucsb_ivar_and_ivarplus")
    sys.exit()
sys.exit()
