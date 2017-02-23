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
import data_handling.data_cleaning as dclean
import data_handling.create_analysis_spreadsheets as cas
import data_handling.data_verification as dver
import data_handling.alloy_property_utilities as apu
import time
from pymongo import MongoClient
from bson.objectid import ObjectId


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

def import_initial_collections(db, cbasic, importpath):
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

def filter_temperatures_ivar_basic(db, cname, verbose=1):
    [id_list, reason_list] = dclean.get_field_condition_to_remove(db,cname,
                                "temperature_C",270)
    dclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    [id_list, reason_list] = dclean.get_field_condition_to_remove(db,cname,
                                "temperature_C",310)
    dclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    [id_list, reason_list] = dclean.get_field_condition_to_remove(db,cname,
                                "temperature_C",320)
    dclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    return

def add_standard_fields(db, cname, verbose=0):
    """Add fields that are standard to most analysis
    """
    cas.add_alloy_number_field(db, cname, verbose=0)
    cas.add_atomic_percent_field(db, cname, verbose=0)
    cas.add_log10_of_a_field(db, cname,"fluence_n_cm2")
    cas.add_log10_of_a_field(db, cname,"flux_n_cm2_sec")
    cas.add_generic_effective_fluence_field(db, cname, 3e10, 0.26)
    cas.add_generic_effective_fluence_field(db, cname, 3e10, 0.10)
    cas.add_generic_effective_fluence_field(db, cname, 3e10, 0.20)
    return

def add_normalized_fields(db, cname, clist=list(), verbose=0):
    cas.add_minmax_normalization_of_a_field(db, cname, "log(fluence_n_cm2)",
            verbose=verbose, collectionlist = clist)
    cas.add_minmax_normalization_of_a_field(db, cname, "log(flux_n_cm2_sec)",
            verbose=verbose, collectionlist = clist)
    cas.add_minmax_normalization_of_a_field(db, cname, "temperature_C",
            verbose=verbose, collectionlist = clist)
    cas.add_minmax_normalization_of_a_field(db, cname, "log(eff fl 100p=26)",
            verbose=verbose, collectionlist = clist)
    cas.add_minmax_normalization_of_a_field(db, cname, "log(eff fl 100p=20)",
            verbose=verbose, collectionlist = clist)
    cas.add_minmax_normalization_of_a_field(db, cname, "log(eff fl 100p=10)",
            verbose=verbose, collectionlist = clist)
    cas.add_stddev_normalization_of_a_field(db, cname, "delta_sigma_y_MPa",
            verbose = verbose, collectionlist = clist)
    return

def create_expt_ivar(db, cname, fromcname, verbose=1):
    """Create IVAR and IVAR+ spreadsheet
    """
    cas.transfer_nonignore_records(db, fromcname, cname, verbose)
    add_standard_fields(db, cname)
    return

def prefilter_ivar_for_cd1(db, cname, fromcname, verbose=1):
    tempname = "%s_temp" % cname
    cas.transfer_nonignore_records(db, fromcname, tempname, verbose)
    [id_list, reason_list] = dclean.get_alloy_removal_ids(db, tempname, 
                                [41,1,2,8,14,29])
    dclean.flag_for_ignore(db, tempname, id_list, reason_list)
    print(len(id_list))
    cas.transfer_nonignore_records(db, tempname, cname, verbose)
    db.drop_collection(tempname)
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
    return

def clean_lwr(db, cname, verbose=1):
    dclean.standardize_alloy_names(db, cname)
    [id_list, reason_list] = dclean.get_alloy_removal_ids(db, cname,[41])
    dclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    
    [id_list, reason_list] = dclean.get_empty_flux_or_fluence_removal_ids(db, cname)
    dclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    
    [id_list, reason_list] = dclean.get_short_time_removal_ids(db,cname, 3e6)
    dclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    
    [id_list, reason_list] = dclean.get_field_condition_to_remove(db,cname,
                                "CD_delta_sigma_y_MPa","")
    dclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    return

def clean_cd1_lwr(db, cname):
    [id_list, reason_list] = dclean.get_alloy_removal_ids(db, cname,[14,29])
    dclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    [id_list, reason_list] = dclean.get_field_condition_to_remove(db,cname,
                                "temperature_C",270)
    dclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    [id_list, reason_list] = dclean.get_field_condition_to_remove(db,cname,
                                "temperature_C",310)
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
    if not "temperature_C" in cas.list_all_fields(db, cname):
        cas.add_basic_field(db, cname, "temperature_C", 290.0) # all at 290
    add_standard_fields(db, cname)
    return

def reformat_lwr(db, cname, fromcname, verbose=1):
    """Reformat CD LWR 2016 where each record has a number of
        columns for each alloy number
    """
    alloy_numbers = apu.get_alloy_numbers(db)
    fields = cas.list_all_fields(db, fromcname)
    transferfields = list(fields)
    transferfields.remove("_id") # do not copy over ID from previous db
    for alloy_num in alloy_numbers:
        if str(alloy_num) in transferfields: #filter out alloy numbers
            transferfields.remove(str(alloy_num))
    records = db[fromcname].find()
    for record in records:
        for alloy_num in alloy_numbers:
            idict=dict()
            for tfield in transferfields:
                idict[tfield] = record[tfield]
            try: 
                dsyval = float(record["%i" % alloy_num])
            except (ValueError, KeyError): #might be Err!, blank, or not exist
                continue
            idict["delta_sigma_y_MPa"] = dsyval
            alloy_name = apu.look_up_name_or_number(db,alloy_num,"number")
            idict["Alloy"] = alloy_name
            db[cname].insert_one(idict)
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

def create_standard_conditions(db, cname, ref_flux=3e10, clist=list(), verbose=0):
    #ref_flux in n/cm2/sec
    min_sec = 3e6
    max_sec = 5e9
    second_range = np.logspace(np.log10(min_sec), np.log10(max_sec), 500)
    alloys = apu.get_alloy_names(db)
    for alloy in alloys:
        for time_sec in second_range:
            fluence = ref_flux * time_sec
            db[cname].insert_one({"Alloy": alloy,
                                "time_sec": time_sec,
                                "fluence_n_cm2": fluence,
                                "flux_n_cm2_sec": ref_flux})
    cas.add_basic_field(db, cname, "temperature_C", 290.0)
    cas.add_alloy_number_field(db, cname, verbose=0)
    cas.add_atomic_percent_field(db, cname, verbose=0)
    cas.add_log10_of_a_field(db, cname,"fluence_n_cm2")
    cas.add_log10_of_a_field(db, cname,"flux_n_cm2_sec")
    cas.add_generic_effective_fluence_field(db, cname, 3e10, 0.26)
    cas.add_generic_effective_fluence_field(db, cname, 3e10, 0.10)
    cas.add_generic_effective_fluence_field(db, cname, 3e10, 0.20)
    cas.add_minmax_normalization_of_a_field(db, cname, "temperature_C",
            verbose=verbose, collectionlist = clist)
    cas.add_minmax_normalization_of_a_field(db, cname, "log(fluence_n_cm2)",
            verbose=verbose, collectionlist = clist)
    cas.add_minmax_normalization_of_a_field(db, cname, "log(flux_n_cm2_sec)",
            verbose=verbose, collectionlist = clist)
    cas.add_minmax_normalization_of_a_field(db, cname, "log(eff fl 100p=26)",
            verbose=verbose, collectionlist = clist)
    cas.add_minmax_normalization_of_a_field(db, cname, "log(eff fl 100p=20)",
            verbose=verbose, collectionlist = clist)
    cas.add_minmax_normalization_of_a_field(db, cname, "log(eff fl 100p=10)",
            verbose=verbose, collectionlist = clist)
    return

def main(importpath):
    dirpath = os.path.dirname(importpath)
    db_base="dbtt"
    cbasic=dict()
    cbasic["alloys"] = "alloy_properties.csv"
    cbasic["cd_ivar_2017"]="CD_IVAR_Hardening_2017-1_with_ivar_columns_reduced.csv"
    cbasic["cd_ivar_2016"]="CD_IVAR_Hardening_clean_2016.csv"
    cbasic["cd_lwr_2017"]="lwr_cd_2017_reduced_for_import.csv"
    cbasic["ucsb_ivar_and_ivarplus"]="ucsb_ivar_and_ivarplus.csv"
    cbasic["cd_lwr_2016_bynum"]="CDTemp_CD_lwr_2016_raw.csv"
    cbasic["atr2_2016"]="atr2_data.csv"
    client = get_mongo_client()
    dbname = get_unique_name(client, db_base)
    exportfolder = "data_exports_%s_%s" %(dbname,time.strftime("%Y%m%d_%H%M%S"))
    exportpath = os.path.join(dirpath, exportfolder)
    db = client[dbname]
    #import initial collections
    import_initial_collections(db, cbasic, importpath)
    #create ancillary databases and spreadsheets
    clean_ivar_basic(db, "ucsb_ivar_and_ivarplus")
    cas.transfer_nonignore_records(db, "ucsb_ivar_and_ivarplus","expt_ivar")
    add_standard_fields(db, "expt_ivar")
    #
    prefilter_ivar_for_cd1(db, "cd1_ivar_pre", "ucsb_ivar_and_ivarplus")
    create_cd_ivar(db, "cd1_ivar", "cd1_ivar_pre", "cd_ivar_2016")
    #
    create_cd_ivar(db, "cd2_ivar", "expt_ivar", "cd_ivar_2017")
    #
    clean_lwr(db, "cd_lwr_2017")
    create_lwr(db, "cd2_lwr", "cd_lwr_2017")
    #
    reformat_lwr(db, "cd_lwr_2016", "cd_lwr_2016_bynum")
    clean_cd1_lwr(db, "cd_lwr_2016")
    clean_lwr(db, "cd_lwr_2016")
    create_lwr(db, "cd1_lwr", "cd_lwr_2016")
    #
    cas.transfer_nonignore_records(db, "atr2_2016","expt_atr2")
    cas.rename_field(db,"expt_atr2","alloy name", "Alloy")
    dclean.standardize_alloy_names(db,"expt_atr2")
    cas.add_basic_field(db, "expt_atr2", "dataset", "ATR2")
    cas.add_time_field(db, "expt_atr2")
    #cas.transfer_nonignore_records(db, "expt_ivar","expt_atr2")
    add_standard_fields(db, "expt_atr2")
    #
    add_normalized_fields(db, "expt_ivar", ["expt_ivar","expt_atr2","cd1_lwr"])
    add_normalized_fields(db, "cd1_ivar", ["cd1_ivar","cd1_lwr"])
    add_normalized_fields(db, "cd1_lwr", ["cd1_ivar","cd1_lwr"])
    add_normalized_fields(db, "cd2_ivar", ["cd2_ivar","cd2_lwr"])
    add_normalized_fields(db, "cd2_lwr", ["cd2_ivar","cd2_lwr"])
    add_normalized_fields(db, "expt_atr2", ["expt_ivar","expt_atr2","cd1_lwr"])
    #
    create_standard_conditions(db, "lwr_std_expt",3e10,["expt_ivar","expt_atr2","cd1_lwr"])
    create_standard_conditions(db, "atr2_std_expt",3.64e12,["expt_ivar","expt_atr2","cd1_lwr"])
    create_standard_conditions(db, "lwr_std_cd1",3e10,["cd1_ivar","cd1_lwr"])
    #
    cas.export_spreadsheet(db, "expt_ivar", exportpath)
    cas.export_spreadsheet(db, "cd1_ivar", exportpath)
    cas.export_spreadsheet(db, "cd2_ivar", exportpath)
    cas.export_spreadsheet(db, "cd2_lwr", exportpath)
    cas.export_spreadsheet(db, "cd1_lwr", exportpath)
    cas.export_spreadsheet(db, "expt_atr2", exportpath)
    cas.export_spreadsheet(db, "lwr_std_expt", exportpath)
    cas.export_spreadsheet(db, "lwr_std_cd1", exportpath)
    #verify data
    clist=["expt_ivar","cd1_ivar","cd2_ivar","cd1_lwr","cd2_lwr","expt_atr2"]
    dver.make_per_alloy_plots(db, clist, "%s/verification_plots" % exportpath) 
    #Additional to-do
    ##
    return exportpath

if __name__ == "__main__":
    importpath = "../../../data/DBTT_mongo/imports_201702"
    importpath = os.path.abspath(importpath)
    exportpath = main(importpath)
    print("Files in %s" % exportpath)
    sys.exit()
