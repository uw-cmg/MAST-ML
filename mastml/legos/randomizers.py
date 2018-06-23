"""
Not integrated yet.
A collection of classes for randomizing X-y matchings
Needs to be part of main conf file.
"""

import pandas as pd
import numpy as np

class Randomizer():
    """
    Lego which randomizes X-y pairings 
    Inherently problematic: transform only operates on X.
    """

    def __init__(self):
        pass

    def fit(self, df, y=None):
        self.shuffler = np.random.permutation(df.shape[0])
        self.reverse_shuffler = np.zeros(df.shape[0], dtype=int)
        self.reverse_shuffler[self.shuffler] = np.arange(df.shape[0])
        return self

    def transform(self, df):
        # A new dataframe is needed to throw out the old indices:
        return pd.DataFrame(df.values[self.shuffler], columns=df.columns)

    def inverse_transform(self, df):
        return pd.DataFrame(df.values[self.reverse_shuffler], columns=df.columns)
