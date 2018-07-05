#! /bin/bash -v
pwd
python3 -m mastml.mastml $1 tests/conf/keras_classifier.conf tests/csv/classification.csv -o results/keras_classifier
python3 -m mastml.mastml $1 tests/conf/advanced_feature_selection_classification.conf tests/csv/classification.csv -o results/advanced_feature_classification
python3 -m mastml.mastml $1 tests/conf/advanced_feature_selection_regression.conf tests/csv/regression_data.csv -o results/advanced_feature_regression
python3 -m mastml.mastml $1 tests/conf/classification.conf tests/csv/three_clusters.csv -o results/classification
python3 -m mastml.mastml $1 tests/conf/feature_gen.conf tests/csv/feature_generation.csv -o results/feature_gen
python3 -m mastml.mastml $1 tests/conf/just_generation.conf tests/csv/feature_generation.csv -o results/just_generation
python3 -m mastml.mastml $1 tests/conf/just_model.conf tests/csv/mnist_short.csv -o results/just_model
python3 -m mastml.mastml $1 tests/conf/just_model_no_plots.conf tests/csv/mnist_short.csv -o results/just_model_no_plots
python3 -m mastml.mastml $1 tests/conf/just_normalization.conf tests/csv/three_clusters.csv -o results/just_normalization
python3 -m mastml.mastml $1 tests/conf/just_selection.conf tests/csv/3d_utility.csv -o results/just_selection
python3 -m mastml.mastml $1 tests/conf/pca_basic.conf tests/csv/3d_utility.csv -o results/pca_basic
python3 -m mastml.mastml $1 tests/conf/pca_multiname.conf tests/csv/3d_utility.csv -o results/pca_multiname
python3 -m mastml.mastml $1 tests/conf/regression.conf tests/csv/boston_housing.csv -o results/regression
python3 -m mastml.mastml $1 tests/conf/regression_basic.conf tests/csv/regression_data.csv -o results/regression_basic
python3 -m mastml.mastml $1 tests/conf/sfs_basic.conf tests/csv/3d_utility.csv -o results/sfs_basic
python3 -m mastml.mastml $1 tests/conf/grouped.conf tests/csv/grouped.csv -o results/grouped
python3 -m mastml.search.search $1 mastml/search/settings.conf mastml/search/mnist_short.csv -o results/search
