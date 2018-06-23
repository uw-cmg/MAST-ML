"""
Not integrated yet.
A collection of classes for randomizing X-y matchings
Needs to be part of main conf file.
"""

import pandas as pd
import numpy as np

class Randomizer():

    def __init__(self, y_feature_name):
        self.y_feature_name = y_feature_name
        
    def fit(self, df, y=None):
        self.shuffler = np.random.permutation(df.shape[0])
        self.reverse_shuffler = np.zeros(df.shape[0], dtype=int)
        self.reverse_shuffler[self.shuffler] = np.arange(df.shape[0])
        return self
    
    def transform(self, df):
        just_X = df.drop(columns=self.y_feature_name)
        new_y = df[self.y_feature_name].copy()
        new_y[:] = new_y[self.shuffler]
        return pd.concat([just_X, new_y], axis=1)

    def inverse_transform(self, df):
        just_X = df.drop(columns=self.y_feature_name)
        new_y = df[self.y_feature_name].copy()
        new_y[:] = new_y[self.reverse_shuffler]
        return pd.concat([just_X, new_y], axis=1)
