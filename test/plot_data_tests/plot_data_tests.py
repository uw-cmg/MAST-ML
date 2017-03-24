#!/usr/bin/env python

import plot_data.plot_csv as plotcsv

kwargs=dict()
kwargs['xlabel'] = "Time"
kwargs['ylabel'] = "Sine feature"
kwargs['yerrfield'] = "sine_error"
plotcsv.single("../random_data/random_test_data.csv","time","sine_feature",**kwargs)
