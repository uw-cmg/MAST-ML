#! /bin/bash -v
python3 -m mastml.mastml tests/conf/regression_basic.conf tests/csv/regression_data.csv -o results/basic_regression
python3 -m mastml.mastml tests/conf/pca_multiname.conf tests/csv/3d_utility.csv -o results/multiple_names
python3 -m mastml.mastml tests/conf/pca_basic.conf tests/csv/3d_utility.csv -o results/pca_basic
python3 -m mastml.mastml tests/conf/classification.conf tests/csv/three_clusters.csv -o results/classification
python3 -m mastml.mastml tests/conf/regression.conf tests/csv/boston_housing.csv -o results/regression
python3 -m mastml.mastml tests/conf/feature_gen.conf tests/csv/feature_generation.csv -o results/generation
python3 -m mastml.mastml tests/conf/advanced_feature_selection_classification.conf tests/csv/classification.csv -o results/advanced_feature_classification
python3 -m mastml.mastml tests/conf/advanced_feature_selection_regression.conf tests/csv/regression_data.csv -o results/advanced_feature_regression
