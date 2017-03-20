#!/usr/bin/env python
###################
# Mongo db utilities
# Tam Mayeshiba 2017-03-20
#
# Prerequisites:
# 1. Must have mongodb installed and running.
#    Visit https://docs.mongodb.com/manual/administration/install-community/
#
###################
import numpy as np
import pymongo
import os
import sys
import traceback
import subprocess
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

