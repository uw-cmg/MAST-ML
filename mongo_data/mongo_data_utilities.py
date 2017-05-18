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
from bson.objectid import ObjectId
import data_handling.data_transformations as dtf
import data_handling.alloy_property_utilities as apu

def get_simplesearch_records(db, cname, fieldstring, comparestring, value):
    results = db[cname].find({fieldstring:{comparestring:value}})
    return results

def get_nonignore_records(db, cname):
    results = get_simplesearch_records(db, cname, "ignore","$ne", 1)
    return results

def transfer_nonignore_records(db, fromcname, newcname, verbose=1):
    transfer_simplesearch_records(db, fromcname, newcname, "ignore",
                        "$ne", 1, verbose)
    return

def transfer_ignore_records(db, fromcname, newcname, verbose=1):
    transfer_simplesearch_records(db, fromcname, newcname, "ignore",
                        "$eq", 1, verbose)
    return


def transfer_simplesearch_records(db, fromcname, newcname, fieldstring,
                    comparestring, value, verbose=1):
    records = get_simplesearch_records(db, fromcname, fieldstring,
                    comparestring, value)
    nict = 0
    for record in records:
        db[newcname].insert(record)
        nict = nict + 1
    if (verbose > 0):
        print("Transferred %i records." % nict)
    return

def match_and_add_records(db, newcname, records, matchlist=list(), matchas=list(), transferlist=list(), transferas=list(), matchmulti=False, verbose=1):
    """Match records to the collection and add certain fields
        Args:
            newcname <str>: Collection name to which to add records
            records <Cursor>: records to match
            matchlist <list of str>: fields on which to match values
            matchas <list of str>: corresponding match fields in the new collection
            transferlist <list of str>: field names to transfer
            transferas <list of str>: field names to rename upon transfer,
                    in the same order listed in transferlist
            matchmulti <bool>: True = yes, False = no (default)
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
        updated = db[newcname].update(matchdict, setdict, multi=matchmulti)
        if updated['updatedExisting'] == True:
            uct = uct + 1
        if (verbose > 1):
            print("Updated: %s" % updated)
    if (verbose > 0):
        print("Total %i records updated." % uct)
    return

def list_all_fields(db, cname, verbose=0):
    """
        List all fields in a collection. Make more sophisticated later.
    """
    fieldlist=list()
    records = db[cname].find()
    for record in records:
        for key in record.keys():
            if not key in fieldlist:
                fieldlist.append(key)
    fieldlist.sort() #sort in place
    if verbose > 0:
        for field in fieldlist:
            print(field)
    return fieldlist

def export_spreadsheet(db, newcname="", prepath="", fieldlist=list()):
    if len(fieldlist) == 0:
        fieldlist=list_all_fields(db, newcname)
    fieldstr=""
    for field in fieldlist:
        fieldstr = fieldstr + field + ","
    
    #outputpath = "%s_%s.csv" % (newcname, time.strftime("%Y%m%d_%H%M%S"))
    outputpath = "%s.csv" % newcname
    if not (prepath == ""):
        outputpath = os.path.join(prepath, outputpath)

    estr = "mongoexport"
    estr += " --db=%s" % db.name
    estr += " --collection=%s" % newcname
    estr += " --out=%s" % outputpath
    estr += " --type=csv"
    estr += ' --fields="%s"' % fieldstr
    eproc = subprocess.Popen(estr, shell=True,
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
    print(eproc.communicate())
    eproc.wait()
    return









def add_log10_of_a_field(db, newcname, origfield, verbose=0):
    """Add the base-10 logarithm of a field as a new field
    """
    newfield="log(%s)" % origfield
    myfunc = getattr(dtf,"get_log10")
    records = db[newcname].find()
    for record in records:
        fieldval = myfunc(record[origfield])
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{newfield:fieldval}}
            )
        if verbose > 0:
            print("Updated record %s with %s %3.3f." % (record["_id"],newfield,fieldval))
    return

def add_stddev_normalization_of_a_field(db, newcname, origfield, verbose=0, collectionlist = list()):
    """Add the normalization of a field based on the mean and standard dev.
        Normalization is given as X_new = (X-X_mean)/(Xstd dev)
            inefficient; needs modification
    """
    newfield="N(%s)" % origfield
    tempcname = "temp_%s" % time.time()

    if len(collectionlist) == 0:
        collectionlist.append(newcname)

    for collection in collectionlist:
        if (verbose > 0):
            print("Using collection %s" % collection)
        results = db[collection].find({"ignore":{"$ne":1}},{"_id":0, origfield:1})
        for result in results:
            db[tempcname].insert({origfield:result[origfield]})

    stddevagg = db[tempcname].aggregate([
        {"$group":{"_id":None,"fieldstddev":{"$stdDevPop":"$%s" % origfield}}}
    ])
    for record in stddevagg:
        stddev = record['fieldstddev']
        print(record)
    avgagg = db[tempcname].aggregate([
        {"$group":{"_id":None,"fieldavg":{"$avg":"$%s" % origfield}}}
    ])
    for record in avgagg:
        mean = record['fieldavg']
        print(record)
    print("Mean: %3.3f" % mean)
    print("StdDev: %3.3f" % stddev)
    records = db[newcname].find()
    for record in records:
        fieldval = ( record[origfield] - mean ) / (stddev)
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{newfield:fieldval}}
            )
        if verbose > 0:
            print("Updated record %s with %s %3.3f." % (record["_id"],newfield,fieldval))
    db[tempcname].drop()
    return

def add_minmax_normalization_of_a_field(db, newcname, origfield, setmin=None, setmax=None, verbose=0, collectionlist=list()):
    """Add the normalization of a field based on the min and max values
        Normalization is given as X_new = (X - X_min)/(X_max - X_min)
        For elemental compositions, max atomic percent is taken as 1.717 At%
            for Mn.
        Args:
            db
            newcname
            origfield <str>: Original field name
            setmin <float or None>: Set minimum directly
            setmax <float or None>: Set maximum directly
            verbose <int>: 0 - silent (default)
                           1 - verbose
            collectionlist <list of str>: list of multiple
                            collection names for normalization
                            over multiple collections
                            empty list only works on collection newcname
    """
    newfield="N(%s)" % origfield

    if len(collectionlist) == 0:
        collectionlist.append(newcname)

    cmins = list()
    cmaxes = list()
    if (setmin == None) or (setmax == None):
        for collection in collectionlist:
            cminval = None
            cmaxval = None
            if (verbose > 0):
                print("Using collection %s" % collection)
            agname = "nonignore_%s" % collection
            db[collection].aggregate([
                        {"$match":{"ignore":{"$ne":1}}},
                        {"$group":{"_id":None,
                            "fieldminval":{"$min":"$%s" % origfield},
                            "fieldmaxval":{"$max":"$%s" % origfield}}},
                        {"$out":agname}
                        ]) #get nonignore records
            for record in db[agname].find(): #one record from aggregation
                cminval = record['fieldminval']
                cmaxval = record['fieldmaxval']
                if verbose > 0:
                    print(record)
            if not (cminval == None):
                cmins.append(cminval)
            if not (cmaxval == None):
                cmaxes.append(cmaxval)
            db[agname].drop()
    if setmin == None:    
        minval = min(cmins)
    else:
        minval = setmin
    if setmax == None:
        maxval = max(cmaxes)
    else:
        maxval = setmax
    print("Min: %3.3f" % minval)
    print("Max: %3.3f" % maxval)
    records = db[newcname].find()
    for record in records:
        if (minval == maxval): #data is flat
            fieldval = 0.0
        else:
            fieldval = ( record[origfield] - minval ) / (maxval - minval)
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{newfield:fieldval}}
            )
        if verbose > 0:
            print("Updated record %s with %s %3.3f." % (record["_id"],newfield,fieldval))
    return

def add_basic_field(db, newcname, fieldname,fieldval, verbose=1):
    db[newcname].update_many({},{"$set": {fieldname: fieldval}})
    if verbose > 0:
        print("Updated collection %s field %s with value %s." % (newcname, fieldname, fieldval))
    return

def remove_field(db, newcname, fieldname, verbose=1):
    db[newcname].update_many({}, {"$unset": {fieldname:1}})
    if verbose > 0:
        print("Removed field %s from all records in collection %s" % (fieldname, newcname))
    return

def rename_field(db, newcname, oldfieldname, newfieldname, verbose=1):
    db[newcname].update_many({}, {"$rename": {oldfieldname: newfieldname}})
    if verbose > 0:
        print("Updated field name %s to %s in collection %s" % (oldfieldname,
                newfieldname, newcname))
    return

def duplicate_string_field_as_numeric(db, newcname, oldfieldname, newfieldname, subdict=dict(), verbose=0):
    """Duplicate a string field as a numeric field
        Args:
            oldfieldname <str>: old field name
            newfieldname <str>: new field name
            subdict <dict>: substitution dictionary, with
                            key as the old field value (string) and
                            value as the new field value (numeric)
                        e.g. {"high":3,"medium":2,"low":1}
    """
    if not subdict: #empty dictionary
        allvals = db[newcname].distinct(oldfieldname)
        for aidx in range(0, len(allvals)):
            subdict[allvals[aidx]] = aidx  
    records = db[newcname].find()
    for record in records:
        newval = ""
        oldval = record[oldfieldname]
        if oldval in subdict.keys():
            newval = subdict[oldval]
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{newfieldname:newval}}
            )
        if verbose > 0:
            print("Updated record %s with %s %3.3f." % (record["_id"],newfieldname, newval))
    return

def duplicate_time_field_as_numeric(db, newcname, oldfieldname, newfieldname, formatstr="%m/%d/%y %H:%M", verbose=0):
    """Duplicate a time field as a numeric field
        Args:
            oldfieldname <str>: old field name
            newfieldname <str>: new field name
            formatstr <str>: Format string to interpret the given time field.
                                See python documentation.
    """
    records = db[newcname].find()
    for record in records:
        oldval = record[oldfieldname]
        try:
            oldstruct = time.strptime(oldval, formatstr)
            newtime = time.mktime(oldstruct)
        except ValueError:
            print("Error updating record %s, old value %s. Setting to blank." % (record["_id"],oldval))
            newtime = ""
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{newfieldname:newtime}}
            )
        if verbose > 0:
            print("Updated record %s with %s %3.3f." % (record["_id"],newfieldname, newval))
    return

if __name__=="__main__":
    print("Use from DataImportAndExport.py. Exiting.")
    sys.exit()
