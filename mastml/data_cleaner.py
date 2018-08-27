"""
Module for cleaning dataframes (e.g. removing NaN, imputation, etc.)
"""

import pandas as pd
import numpy as np
import logging
log = logging.getLogger('mastml')

def remove(df, axis):
    # TODO: add cleaning for y data (remove rows, and need to remove rows from other df's as well
    #df_nan = df[pd.isnull(df)]
    #nan_indices = df_nan.index
    #print(nan_indices)
    df = df.dropna(axis=axis, how='any')
    return df