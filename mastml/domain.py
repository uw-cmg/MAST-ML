'''
This module contains a collection of routines to perform domain evaluations
'''
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

class Domain():

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

    def predict(self, X_test):
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
            domain_vals = ['in_domain' if i == True else 'out_of_domain' for i in domain_vals]
            domains['domain_gpr'] = domain_vals

        domains = pd.DataFrame(domains)

        return domains
