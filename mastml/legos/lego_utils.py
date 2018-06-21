"""
Place for misc functions that are needed for making legos
Actual legos don't go in here.
"""

from functools import wraps
import pandas as pd

def dataframify(transform):
    """
    Decorator to make a transformer's transform method work on dataframes
    Assumes columns will be preserved
    """
    @wraps(transform)
    def new_transform(self, df):
        arr = transform(self, df.values)
        return pd.DataFrame(arr, columns=df.columns, index=df.index)
    return new_transform

def dataframify_selector(transform):
    " Special dataframify which preserves column names for feature selectors "
    @wraps(transform)
    def new_transform(self, df):
        return df[df.columns[self.get_support(indices=True)]]
    return new_transform

def variance_threshold_selector(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]
