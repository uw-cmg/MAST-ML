from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

from mastml.mad.ml import splitters, feature_selectors, domain, domain_assessment
from mastml.mad.datasets import load_data, statistics
from mastml.mad.plots import parity, calibration

import numpy as np
import math


def main():
    '''
    Test ml workflow
    '''

    seed = 14987

    # Load data
    data = load_data.diffusion()
    df = data['frame']
    X = data['data']
    y = data['target']
    d = data['class_name']

    # Feature scaling and feature selection
    scale = StandardScaler()
    selector = feature_selectors.no_selection()

    # Splitter for inner folds
    inner_split = RepeatedKFold(n_splits=5, n_repeats=1)

    # Splitters for outer folds (<name>, <splitter>)
    outer_splitters = []
    rand_split = ('random', RepeatedKFold(n_splits=5, n_repeats=1))
    chem_split = ('chemical', splitters.BootstrappedLeaveOneGroupOut(
                                                                     n_repeats=1,
                                                                     groups=d
                                                                     ))
    outer_splitters.append(rand_split)
    outer_splitters.append(chem_split)
    
    for i in [2, 10]:

        # Cluster Splits
        top_split = splitters.BootstrappedClusterSplit(
                                                       KMeans,
                                                       n_repeats=5,
                                                       n_clusters=i
                                                       )

        outer_splitters.append(('kmeans_{}'.format(i), top_split))

    # Random forest regression with grid search
    grid = {}
    model = RandomForestRegressor()
    grid['model__n_estimators'] = [100]
    grid['model__max_features'] = [None]
    grid['model__max_depth'] = [None]
    pipe = Pipeline(steps=[
                           ('scaler', scale),
                           ('select', selector),
                           ('model', model)
                           ])
    gs_model = GridSearchCV(pipe, grid, cv=inner_split)

    # Evaluate and note effect of outer split sampling
    for i in outer_splitters:
        splits = domain.builder(
                                gs_model,
                                X,
                                y,
                                d,
                                i[1],
                                i[0],
                                seed=seed,
                                )

        splits.assess_domain()  # Do ML
        splits.aggregate()  # combine all of the ml data
        statistics.folds(i[0])  # Gather statistics from data
        parity.make_plots(i[0], 'gpr_std')
        calibration.make_plots(i[0], 'stdcal', 'gpr_std')

    domain_assessment.calc()


if __name__ == '__main__':
    main()

