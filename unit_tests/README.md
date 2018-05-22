# initial observations


`MASTML_unit_tests.py:13` is the only place that has the name of a config file:
```python
        self.configfile = 'test_unittest_fullrun.conf'
```

# tree
.
├── DataOperations_unit_tests.py
├── FeatureGeneration_unit_tests.py
├── (REF MISSING FILE) FeatureOperations_unit_tests.py
├── FeatureSelection_unit_tests.py
├── input_with_univariate_feature_selection_O_pband_center_regression.csv
├── MASTML_config_files
│   ├── magpiedata
│   │   └── magpie_elementdata
│   │       ├── AtomicNumber.table
│   │       ├── AtomicRadii.table
│   │       ├── AtomicVolume.table
│   │       ├── ...
│   │       └── valence.table
│   ├── (UNUSED) mastmlinputvalidationnames.conf
│   └── (UNUSED) mastmlinputvalidationtypes.conf
├── MASTMLInitializer_unit_tests.py
├── (IMPORTANT) MASTML_unit_tests.py
├── testcsv1constant.csv
├── testcsv1.csv
├── testcsv1featureselection.csv
├── testcsv1matproj.csv
├── testcsv2.csv
├── testcsv_fullrun.csv
├── (UNUSED) test_unittest_dataoperations.conf
├── (UNUSED) test_unittest_featuregeneration.conf
└── (IMPORTANT) test_unittest_fullrun.conf

3 directories, 108 files
