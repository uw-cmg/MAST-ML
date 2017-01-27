#!/usr/bin/env python
###################
# Data transformations for use with dbtt database
# Tam Mayeshiba 2017-01-27
###################

import numpy
import pymongo
import os
import sys

from pymongo import MongoClient
from bson.objectid import ObjectId

dbname="dbtt"

client = MongoClient('localhost', 27017)
db = client[dbname]

def get_time_from_flux_and_fluence(flux, fluence):
    time = float(fluence) / float(flux)
    return time
