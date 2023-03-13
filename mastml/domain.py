'''
This module contains a collection of routines to perform domain evaluations
'''
import re
import itertools
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.spatial.distance import cdist


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
    def __init__(self, ref, check_type):
        self.ref = ref  # The reference indexes
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

    def check(self, test, groups=None):
        if self.check_type == 'elemental':

            chem_ref = groups[self.ref]
            chem_test = groups[test]

            chem_ref = chem_ref.apply(self.convert_string_to_elements)
            chem_test = chem_test.apply(self.convert_string_to_elements)

            # Merge training cases to check if each test case is within
            chem_ref = set(itertools.chain.from_iterable(chem_ref))

            domains = []
            for i in chem_test:
                d = self.compare_elements(chem_ref, i)
                domains.append(d)

        return domains

