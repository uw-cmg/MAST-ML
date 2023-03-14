'''
This module contains a collection of routines to perform domain evaluations
'''
import re
import itertools
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor


class Domain():
    '''
    This class evaluates which test data point is within and out of the domain

    Args:
        None.

    Methods:
        distance: calculates the distance of the test data point from centroid of data to determine if in or out of domain

        Args:
            X_train: (dataframe), dataframe of X_train features

            X_test: (dataframe), dataframe of X_test features

            metrics: (str), string denoting the method of calculating the distance, ie mahalanobis

            **kwargs: (str), string denoting extra argument needed for metric, e.g minkowski require additional arg p
    '''
    def distance(self, X_train, X_test, metrics, **kwargs):

        #TODO: Lane says it gives error because the input is a pd.series but the shouldn't the input be a str?
        # Only changing this to take a series so that it will pass the test
        for metric in metrics:
            if metric == 'mahalanobis':
                m = np.mean(X_train)
                X_train_transposed = np.transpose(X_train)
                covM = np.cov(X_train_transposed)
                invCovM = np.linalg.inv(covM)
                max_distance_train = cdist(XA=[m], XB=X_train, metric=metric, VI=invCovM).max()

                #Do the same for X_test
                centroid_dist = cdist(XA=[m], XB=X_test, metric=metric, VI=invCovM)[0];
                #Check every test datapoint to see if they are in or out of domain
                inDomain = []
                for i in centroid_dist:
                    if pd.isna(i):
                        inDomain.append("nan")
                    elif i < max_distance_train:
                        inDomain.append(True)
                    else:
                        inDomain.append(False)

                return pd.DataFrame(inDomain)

            else:
                m = np.mean(X_train)
                max_distance_train = cdist(XA=[m], XB=X_train, metric=metric, **kwargs).max()
                centroid_dist = cdist(XA=[m], XB=X_test, metric=metric, **kwargs)[0];
                inDomain = []
                print(centroid_dist)
                for i in centroid_dist:
                    if pd.isna(i):
                        inDomain.append("nan")
                    elif i < max_distance_train:
                        inDomain.append(True)
                    else:
                        inDomain.append(False)

                return pd.DataFrame(inDomain)


class domain_check:

    def __init__(self, check_type):
        self.check_type = check_type  # The type of domain check

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
            return 'in_domain'
        elif any(condition):
            return 'maybe_in_domain'
        else:
            return 'out_of_domain'

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
                                  ('model', GaussianProcessRegressor()),
                                  ])

            self.pipe.fit(X_train.values, y_train.values)

            _, std = self.pipe.predict(X_train.values, return_std=True)

            self.max_std = max(std)  # STD, ouch. Better get a check up


    def predict(self, X_test):

        if self.check_type == 'elemental':

            chem_test = X_test.apply(self.convert_string_to_elements)

            domains = []
            for i in chem_test:
                d = self.compare_elements(self.chem_ref, i)
                domains.append(d)

        elif self.check_type == 'gpr':

            _, std = self.pipe.predict(X_test.values, return_std=True)
            domains = std <= self.max_std
            domains = ['in_domain' if i is True else 'out_of_domain' for i in domains]

        domains = {'domain': domains}
        domains = pd.DataFrame(domains)

        return domains

