#!/usr/bin/env python
#TTM make random data for testing
import numpy as np
import pandas as pd
import os
from DataOperations import DataParser
from FeatureOperations import FeatureIO, FeatureNormalization

class MakeRandomData():
    def __init__(self, 
                save_path=None,
                random_seed=None):
        self.save_path = save_path
        self.random_state = np.random.RandomState(int(random_seed))
        self.dataframe = None
        return

    def run(self):
        self.make_data()
        self.print_data()
        return

    def print_data(self):
        ocsvname = os.path.join(self.save_path, "make_random_test_data.csv")
        self.dataframe.to_csv(ocsvname)
        return

    def make_data(self):
        n_samples, n_features = 100, 5
        y = self.random_state.randn(n_samples)
        X = self.random_state.randn(n_samples, n_features)
       
        nidx = np.arange(0, n_samples)
        self.dataframe = pd.DataFrame(index=nidx)
        num_cat = self.random_state.randint(0,4,n_samples)
        cats=['A','B','C','D']
        str_cat = [cats[nc] for nc in num_cat]
        time = nidx * np.pi/8.0
        sine_feature = np.sin(time) + X[:,0] #add noise
        linear_feature = 100*time + 30.0 + X[:,1] #add noise
        y_feature = np.sin(time) + y/10.0
        y_feature_error = X[:,3]/X[:,4]/100.0 #add random error
        d_cols =dict()
        d_cols["num_idx"] = nidx
        d_cols["num_cat"] = num_cat
        d_cols["str_cat"] = str_cat
        d_cols["time"] = time
        d_cols["sine_feature"] = sine_feature
        d_cols["linear_feature"] = linear_feature
        d_cols["y_feature"] = y_feature
        d_cols["y_feature_error"] = y_feature_error
        cols = list(d_cols.keys())
        cols.sort()
        for col in cols:
            fio = FeatureIO(self.dataframe)
            self.dataframe = fio.add_custom_features([col], d_cols[col])
        fnorm = FeatureNormalization(self.dataframe)
        N_sine_feature = fnorm.minmax_scale_single_feature("sine_feature")
        N_linear_feature = fnorm.minmax_scale_single_feature("linear_feature")
        fio = FeatureIO(self.dataframe)
        self.dataframe = fio.add_custom_features(["N_sine_feature"], N_sine_feature)
        fio = FeatureIO(self.dataframe)
        self.dataframe = fio.add_custom_features(["N_linear_feature"], N_sine_feature)
        return

if __name__ == "__main__":
    mymd = MakeRandomData(save_path = os.getcwd(), random_seed=0)
    mymd.run()
