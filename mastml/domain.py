'''
This module contains a collection of routines to perform domain evaluations
'''
import os
import re
import itertools
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, Matern

from sklearn.model_selection import GridSearchCV
from sklearn.cluster import AgglomerativeClustering
from madml.splitters import BootstrappedLeaveClusterOut
from madml.models import dissimilarity
from madml.models import calibration
from madml.assess import nested_cv
from madml.models import combine

class Domain():

    def __init__(self, check_type, preprocessor=None, model=None, params=None, path=None):
        self.check_type = check_type  # The type of domain check
        self.params = params
        self.model = model
        self.preprocessor = preprocessor
        self.path = path

    # Convert a string to elements
    def convert_string_to_elements(self, x):
        x = re.sub('[^a-zA-Z]+', '', x)
        x = Composition(x)
        x = x.chemical_system
        x = x.split('-')
        return x

    # Check if all elements from a reference are the same as another case
    def compare_elements(self, ref, x):

        # Check if any reference material in another observation
        condition = [True if i in ref else False for i in x]

        if all(condition):
            return 1
        elif any(condition):
            return 0
        else:
            return -1

    def fit(self, X_train=None, y_train=None):

        # Data here is the chemical groups
        if self.check_type == 'elemental':

            chem_ref = X_train.apply(self.convert_string_to_elements)

            # Merge training cases to check if each test case is within
            self.chem_ref = set(itertools.chain.from_iterable(chem_ref))

        # Data here is features and target variable
        elif self.check_type == 'gpr':

            self.pipe = Pipeline([
                                  ('scaler', StandardScaler()),
                                  ('model', GaussianProcessRegressor(kernel=ConstantKernel()*Matern()+WhiteKernel(), n_restarts_optimizer=10)),
                                  ])

            splitter = RepeatedKFold(n_repeats=1, n_splits=5)
            stds = []
            for tr_index, te_index in splitter.split(X_train.values):

                self.pipe.fit(
                              X_train.values[tr_index],
                              y_train.values[tr_index]
                              )

                _, std = self.pipe.predict(
                                           X_train.values[te_index],
                                           return_std=True
                                           )

                stds = np.concatenate((stds, std), axis=None)

            self.max_std = max(stds)  # STD, ouch. Better get a check up

            # Train one more time with all training data
            self.pipe.fit(
                          X_train.values,
                          y_train.values
                          )

        elif self.check_type == 'feature_range':
            self.features = X_train.columns.tolist()
            self.feature_mins = dict()
            self.feature_maxes = dict()
            for f in self.features:
                self.feature_mins[f] = min(X_train[f])
                self.feature_maxes[f] = max(X_train[f])

        elif self.check_type == 'madml':

            # The machine learning pipeline
            pipe = Pipeline(steps=[
                                   ('scaler', self.preprocessor.preprocessor),
                                   ('model', self.model.model),
                                   ])

            # Need GridSearchCV object for compatability
            gs_model = GridSearchCV(
                                    pipe,
                                    {},
                                    cv=((slice(None), slice(None)),),
                                    )

            ds_model = dissimilarity(dis='kde')
            if 'bandwidth' in self.params:
                ds_model.bandwidth= self.params['bandwidth']
            if 'kernel' in self.params:
                ds_model.kernel = self.params['kernel']

            # Uncertainty estimation model
            if 'uq_coeffs' in self.params:
                uq_model = calibration(params=self.params['uq_coeffs'])
            elif 'uq_function' in self.params:
                uq_model = calibration(
                                             uq_func=self.params['uq_function'],
                                             prior=True
                                             )
            else:
                uq_model = calibration(params=[0.0, 1.0])  # UQ model

            # Types of sampling to test
            splits = [('fit', RepeatedKFold(n_repeats=self.params['n_repeats']))]

            # Boostrap, cluster data, and generate splits
            if 'n_clusters' in self.params:
                n_clusters = self.params['n_clusters']
            else:
                n_clusters = [2, 3]

            for i in n_clusters:

                # Cluster Splits
                top_split = BootstrappedLeaveClusterOut(
                                                        AgglomerativeClustering,
                                                        n_repeats=self.params['n_repeats'],
                                                        n_clusters=i
                                                        )

                splits.append(('agglo_{}'.format(i), top_split))

            # Fit models
            model = combine(gs_model, ds_model, uq_model, splits)

            # Arguments may have different names in MADML implementation
            if 'bins' in self.params:
                model.bins = self.params['bins']
            if 'gt_rmse' in self.params:
                model.gt_rmse = self.params['gt_rmse']
            if 'gt_area' in self.params:
                model.gt_area = self.params['gt_area']

            cv = nested_cv(model=model,
                           X=X_train.values,
                           y=y_train.values.ravel(),
                           g=np.array(['None']*y_train.shape[0]),
                           splitters=splits,
                           )

            _, __, self.madml_model = cv.test(save_outer_folds=os.path.join(self.path, 'madml_domain'))

    def predict(self, X_test, *args, **kwargs):
        domains = dict()
        if self.check_type == 'elemental':

            chem_test = X_test.apply(self.convert_string_to_elements)

            domain_vals = []
            for i in chem_test:
                d = self.compare_elements(self.chem_ref, i)
                domain_vals.append(d)
            domains['domain_elemental'] = domain_vals

        elif self.check_type == 'gpr':

            _, std = self.pipe.predict(X_test.values, return_std=True)
            domain_vals = std <= self.max_std
            domain_vals = [1 if i == True else -1 for i in domain_vals]
            domains['domain_gpr'] = domain_vals

        elif self.check_type == 'feature_range':
            features_inside_training_range = list()
            features_outside_training_full = list()
            for i, x in X_test.iterrows():
                num_OK = len(self.features)
                features_outside_training = ''
                for f in self.features:
                    is_OK = True
                    if x[f] < self.feature_mins[f]:
                        is_OK = False
                        # features_outside_training.append(f)
                        features_outside_training += str(f) + ', '
                    if x[f] > self.feature_maxes[f]:
                        is_OK = False
                        # features_outside_training.append(f)
                        features_outside_training += str(f) + ', '
                    if is_OK == False:
                        num_OK -= 1
                    # if is_OK == True:
                    #    features_outside_training.append('n/a')
                features_inside_training_range.append(num_OK)
                features_outside_training_full.append(features_outside_training)
            domains['domain_feature_range_numfeatsinside'] = features_inside_training_range
            domains['domain_feature_range_featsoutside'] = features_outside_training_full

        elif self.check_type == 'madml':

            if "madml_thresholds" in kwargs.keys():
                t = kwargs['madml_thresholds']
                th = []
                for i in t:
                    i = ['dist', i[0], i[1]]

                    if i[1] == 'residual':
                        i[1] = 'id'
                    elif i[1] == 'uncertainty':
                        i[1] = 'id_bin'

                    th.append(i)

                domains = self.madml_model.predict(X_test, th)
            else:
                domains = self.madml_model.predict(X_test)

            return domains

        domains = pd.DataFrame(domains)

        return domains