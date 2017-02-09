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
    print("")
    return

def clean_ivar_basic(db, cname, verbose=1):
    [id_list, reason_list] = dclean.get_alloy_removal_ids(db, cname, [41])
    dclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    [id_list, reason_list] = dclean.get_duplicate_ids_to_remove(db, cname)
    dclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    dclean.update_experimental_temperatures(db, cname)
    return

def add_standard_fields(db, cname, verbose=0):
    """Add fields that are standard to most analysis
    """
    cas.add_atomic_percent_field(db, cname, verbose=0)
    cas.add_log10_of_a_field(db, cname,"fluence_n_cm2")
    cas.add_log10_of_a_field(db, cname,"flux_n_cm2_sec")
    cas.add_minmax_normalization_of_a_field(db, cname, "log(fluence_n_cm2)")
    cas.add_minmax_normalization_of_a_field(db, cname, "log(flux_n_cm2_sec)")
    cas.add_minmax_normalization_of_a_field(db, cname, "temperature_C")
    cas.add_generic_effective_fluence_field(db, cname, 3e10, 0.26)
    cas.add_stddev_normalization_of_a_field(db, cname, "delta_sigma_y_MPa")
    return

def create_expt_ivar(db, cname, fromcname, verbose=1):
    """Create IVAR and IVAR+ spreadsheet
    """
    cas.transfer_nonignore_records(db, fromcname, cname, verbose)
    add_standard_fields(db, cname)
    cas.export_spreadsheet(db, cname, "../../../data/DBTT_mongo/data_exports/")
    return

def create_cd_ivar(db, cname, fromcname, fromcdname, verbose=1):
    """Create IVAR and IVAR+ spreadsheet for CD data
        Get conditions from experimental IVAR; will match to CD
        data and replace delta_sigma_y_MPa with CD's data
    """
    cas.transfer_nonignore_records(db, fromcname, cname, verbose)
    cas.remove_field(db, cname, "delta_sigma_y_MPa") #will replace with CD data
    add_cd(db, cname, fromcdname)
    add_standard_fields(db, cname)
    cas.export_spreadsheet(db, cname, "../../../data/DBTT_mongo/data_exports/")
    return


def create_ivar_for_fullfit(db, cname, fromcname, verbose=1):
    """Create IVAR-only spreadsheet for
        optimizing hyperparameters
    """
    #remove IVAR+
    tempname = "%s_temp" % cname
    cas.transfer_nonignore_records(db, fromcname, tempname, verbose)
    [id_list, reason_list] = dclean.get_field_condition_to_remove(db, tempname,
                            "dataset","IVAR+")
    dclean.flag_for_ignore(db, tempname, id_list, reason_list)
    print(len(id_list))
    cas.transfer_nonignore_records(db, tempname, cname, verbose)
    cas.export_spreadsheet(db, cname, "../../../data/DBTT_mongo/data_exports/")
    return

def clean_lwr(db, cname, verbose=1):
    dclean.standardize_alloy_names(db, cname)
    [id_list, reason_list] = dclean.get_alloy_removal_ids(db, cname,[41])
    dclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    
    [id_list, reason_list] = dclean.get_empty_flux_or_fluence_removal_ids(db, cname)
    dclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    
    [id_list, reason_list] = dclean.get_short_time_removal_ids(db,cname, 30e6)
    dclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    
    [id_list, reason_list] = dclean.get_field_condition_to_remove(db,cname,
                                "CD_delta_sigma_y_MPa","")
    dclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    return

def create_lwr(db, cname, fromcname, verbose=1):
    """Create LWR condition spreadsheet
    """
    cas.transfer_nonignore_records(db, fromcname, cname, verbose)
    #Additional cleaning. Flux and fluence must be present for all records.
    dclean.standardize_flux_and_fluence(db, cname)
    cas.rename_field(db, cname, "CD_delta_sigma_y_MPa", "delta_sigma_y_MPa")
    cas.add_basic_field(db, cname, "temperature_C", 290.0) # all at 290
    add_standard_fields(db, cname)
    cas.export_spreadsheet(db, cname, "../../../data/DBTT_mongo/data_exports/")
    return

def add_cd(db, cname, cdname, verbose=1):
    """Match CD records to expt IVAR conditions and replace delta_sigma_y_MPa
    """
    cd_records = cas.get_nonignore_records(db, cdname) 
    cas.match_and_add_records(db, cname, cd_records, 
        matchlist=["Alloy","flux_n_cm2_sec","fluence_n_cm2","temperature_C"],
        matchas=["Alloy","flux_n_cm2_sec","fluence_n_cm2","temperature_C"],
        transferlist= ["CD_delta_sigma_y_MPa"],
        transferas = ["delta_sigma_y_MPa"])
    print("Updated with condition and temperature matches from %s." % cdname)
    cd_records.rewind()
    cas.match_and_add_records(db, cname, cd_records, 
        matchlist=["Alloy","flux_n_cm2_sec","fluence_n_cm2","temperature_C"],
        matchas=["Alloy","flux_n_cm2_sec","fluence_n_cm2","original_reported_temperature_C"],
        transferlist= ["CD_delta_sigma_y_MPa"],
        transferas = ["delta_sigma_y_MPa"])
    print("Updated with condition and old temperature matches from %s." % cdname)
    return

if __name__ == "__main__":
    client = get_mongo_client()
    dbname = get_unique_name(client, db_base)
    db = client[dbname]
    import_initial_collections(db, cbasic)
    clean_ivar_basic(db, "ucsb_ivar_and_ivarplus")
    create_expt_ivar(db, "expt_ivar", "ucsb_ivar_and_ivarplus", verbose=0)
    create_cd_ivar(db, "cd1_ivar", "expt_ivar", "cd_ivar_2016")
    create_cd_ivar(db, "cd2_ivar", "expt_ivar", "cd_ivar_2017")
    create_ivar_for_fullfit(db, "expt_ivaronly", "expt_ivar")
    create_ivar_for_fullfit(db, "cd1_ivaronly", "cd1_ivar")
    create_ivar_for_fullfit(db, "cd2_ivaronly", "cd2_ivar")
    clean_lwr(db, "cd_lwr_2017")
    create_lwr(db, "cd2_lwr", "cd_lwr_2017")
    sys.exit()
sys.exit()
