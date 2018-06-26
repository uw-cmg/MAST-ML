#! /bin/bash
python -m mastml.mastml tests/conf/regression_basic.conf tests/csv/regression_data.csv -o results/basic_regression
python -m mastml.mastml tests/conf/multiple_names.conf tests/csv/3d_utility.csv -o results/multiple_names
python -m mastml.mastml tests/conf/classification.conf tests/csv/three_clusters.csv -o results/classification
python -m mastml.mastml tests/conf/regression.conf tests/csv/boston_housing.csv -o results/regression
python -m mastml.mastml tests/conf/feature-gen.conf tests/csv/feature-gen.csv -o results/generation
