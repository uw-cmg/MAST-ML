"""
Place for misc functions that are needed for making legos
Actual legos don't go in here.
"""

from functools import wraps
import pandas as pd

def dataframify(transform):
    " Decorator to make a transformer's transform method work on dataframes. "
    @wraps(transform)
    def new_transform(self, df):
        arr = transform(self, df.values)
        try:
            return pd.DataFrame(arr, columns=df.columns, index=df.index)
        except ValueError:
            return pd.DataFrame(arr, index=df.index)
    return new_transform
