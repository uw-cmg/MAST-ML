from mad.functions import parallel, llh, poly
from mastml.error_analysis import CorrectionFactors
from mad.ml import distances

from sklearn.base import clone

import pandas as pd
import numpy as np
import random
import dill
import glob
import os

import warnings
warnings.filterwarnings('ignore')


class uq_func_model:

    def train(self, std, y, y_pred):
        params = CorrectionFactors(y-y_pred, std)
        params = params.nll()
        self.params = params[::-1]
        self.uq_func = poly

    def predict(self, std):
        return self.uq_func(self.params, std)


class dist_func_model:

    def train(self, X, y=None):
        self.dist_func = lambda x: distances.distance(X, x, y)

    def predict(self, X):
        return self.dist_func(X)


class builder:
    '''
    Class to use the ingredients of splits to build a model and assessment.
    '''

    def __init__(
                 self,
                 pipe,
                 X,
                 y,
                 d,
                 splitter,
                 save,
                 seed=1,
                 ):
        '''
        inputs:
            pipe = The machine learning pipeline.
            X = The features.
            y = The target variable.
            d = The domain for each case.
            splitter = The test set splitter.
            splitters = The splitting oject to create 2 layers.
            save = The directory to save splits.
            seed = The seed option for reproducibility.
        '''

        # Setting seed for reproducibility
        np.random.seed(seed)
        np.random.RandomState(seed)
        random.seed(seed)

        self.pipe = pipe
        self.X = X
        self.y = y
        self.d = d
        self.splitter = splitter

        # Output directory creation
        self.save = save

    def assess_domain(self):
        '''
        Asses the model through nested CV with a domain layer.
        '''

        o = np.array(range(self.X.shape[0]))  # Tracking cases.

        # Setup saving directory.
        save = os.path.join(self.save, 'splits')
        os.makedirs(save, exist_ok=True)

        # Make all of the train and test splits.
        test_count = 0  # In domain count
        splits = []
        for i in self.splitter.split(self.X, self.y, self.d):

            tr_index = np.array(i[0])  # The train.
            te_index = np.array(i[1])  # The test.

            tr_te = (
                     tr_index,
                     te_index,
                     test_count,
                     )

            splits.append(tr_te)

            test_count += 1  # Increment in domain count

        # Do nested CV
        parallel(
                 self.nestedcv,
                 splits,
                 X=self.X,
                 y=self.y,
                 d=self.d,
                 pipe=self.pipe,
                 save=save,
                 )

    def nestedcv(
                 self,
                 indexes,
                 X,
                 y,
                 d,
                 pipe,
                 save,
                 ):
        '''
        A class for nesetd cross validation.

        inputs:
            indexes = The in domain test and training indexes.
            X = The feature set.
            y = The target variable.
            d = The class.
            pipe = The machine learning pipe.
            save = The saving directory.
            uq_coeffs_start = The starting coefficients for UQ polynomial.

        outputs:
            df = The dataframe for all evaluation.
        '''

        # Split indexes and spit count
        tr, te, test_count = indexes

        X_train, X_test = X[tr], X[te]
        y_train, y_test = y[tr], y[te]
        d_train, d_test = d[tr], d[te]

        # Fit the model on training data in domain.
        self.pipe.fit(X_train, y_train)

        # Grab model critical information for assessment
        pipe_best = pipe.best_estimator_
        pipe_best_scaler = pipe_best.named_steps['scaler']
        pipe_best_select = pipe_best.named_steps['select']
        pipe_best_model = pipe_best.named_steps['model']

        if 'manifold' in pipe_best.named_steps:
            pipe_best_manifold = pipe_best.named_steps['manifold']

        # Grab model specific details
        model_type = pipe_best_model.__class__.__name__

        # Feature transformations
        X_train_trans = pipe_best_scaler.transform(X_train)
        X_test_trans = pipe_best_scaler.transform(X_test)

        if 'manifold' in pipe_best.named_steps:
            X_train_trans = pipe_best_manifold.transform(X_train)
            X_test_trans = pipe_best_manifold.transform(X_test)

        # Feature selection
        X_train_select = pipe_best_select.transform(X_train_trans)
        X_test_select = pipe_best_select.transform(X_test_trans)

        # Setup distance model
        dists = dist_func_model()
        dists.train(X_train_select, y_train)

        # Calculate distances after feature transformations from ML workflow.
        df_te = dists.predict(X_test_select)

        # If model is ensemble regressor (need to update varialbe name)
        ensemble_methods = [
                            'RandomForestRegressor',
                            'BaggingRegressor',
                            'GradientBoostingRegressor',
                            'GaussianProcessRegressor'
                            ]

        if model_type in ensemble_methods:

            # Train and test on inner CV
            std_cv = []
            d_cv = []
            y_cv = []
            y_cv_pred = []
            y_cv_indx = []
            df_tr = []
            for train_index, test_index in pipe.cv.split(
                                                         X_train_select,
                                                         y_train,
                                                         d_train
                                                         ):

                model = clone(pipe_best_model)

                X_cv_train = X_train_select[train_index]
                X_cv_test = X_train_select[test_index]

                y_cv_train = y_train[train_index]
                y_cv_test = y_train[test_index]

                model.fit(X_cv_train, y_cv_train)

                if model_type == 'GaussianProcessRegressor':
                    _, std = model.predict(X_cv_test, return_std=True)
                else:
                    std = []
                    for i in model.estimators_:
                        if model_type == 'GradientBoostingRegressor':
                            i = i[0]
                        std.append(i.predict(X_cv_test))

                    std = np.std(std, axis=0)

                dists_cv = dist_func_model()
                dists_cv.train(X_cv_train, y_cv_train)

                std_cv = np.append(std_cv, std)
                d_cv = np.append(d_cv, d_train[test_index])
                y_cv = np.append(y_cv, y_cv_test)
                y_cv_pred = np.append(y_cv_pred, model.predict(X_cv_test))
                y_cv_indx = np.append(y_cv_indx, tr[test_index])
                df_tr.append(pd.DataFrame(dists_cv.predict(X_cv_test)))

            df_tr = pd.concat(df_tr)

            # Calibration
            uq_func = uq_func_model()
            uq_func.train(std_cv, y_cv, y_cv_pred)

            # Nested prediction for left out data
            y_test_pred = pipe_best.predict(X_test)

            # Ensemble predictions with correct feature set
            if model_type == 'GaussianProcessRegressor':
                _, std_test = pipe_best_model.predict(
                                                      X_test_select,
                                                      return_std=True
                                                      )

            else:
                pipe_estimators = pipe_best_model.estimators_
                std_test = []
                for i in pipe_estimators:

                    if model_type == 'GradientBoostingRegressor':
                        i = i[0]

                    std_test.append(i.predict(X_test_select))

                std_test = np.std(std_test, axis=0)

            stdcal_cv = uq_func.predict(std_cv)
            stdcal_test = uq_func.predict(std_test)

            # Grab standard deviations.
            df_tr['std'] = std_cv
            df_te['std'] = std_test

            # Grab calibrated standard deviations.
            df_tr['stdcal'] = stdcal_cv
            df_te['stdcal'] = stdcal_test

        else:
            raise Exception('Only ensemble models supported.')

        # Assign domain.
        df_tr['in_domain'] = ['tr']*std_cv.shape[0]
        df_te['in_domain'] = ['te']*X_test.shape[0]

        # Grab indexes of tests.
        df_tr['index'] = y_cv_indx
        df_te['index'] = te

        # Grab the domain of tests.
        df_tr['domain'] = d_cv
        df_te['domain'] = d_test

        # Grab the true target variables of test.
        df_tr['y'] = y_cv
        df_te['y'] = y_test

        # Grab the predictions of tests.
        df_tr['y_pred'] = y_cv_pred
        df_te['y_pred'] = y_test_pred

        # Calculate the negative log likelihoods
        df_tr['nllh'] = -llh(
                             std_cv,
                             y_cv-y_cv_pred,
                             uq_func.params,
                             uq_func.uq_func
                             )
        df_te['nllh'] = -llh(
                             std_test,
                             y_test-y_test_pred,
                             uq_func.params,
                             uq_func.uq_func
                             )

        df_tr = pd.DataFrame(df_tr)
        df_te = pd.DataFrame(df_te)

        df = pd.concat([df_tr, df_te])

        # Assign values that should be the same
        df['test_count'] = test_count
        df['run'] = save.replace('/splits', '')

        dfname = 'split_{}.csv'.format(test_count)
        modelname = 'model_{}.joblib'.format(test_count)
        uqname = 'uqfunc_{}.joblib'.format(test_count)
        distname = 'distfunc_{}.joblib'.format(test_count)

        dfname = os.path.join(save, dfname)
        modelname = os.path.join(save, modelname)
        uqname = os.path.join(save, uqname)
        distname = os.path.join(save, distname)

        df.to_csv(dfname, index=False)
        dill.dump(pipe, open(modelname, 'wb'))
        dill.dump(uq_func, open(uqname, 'wb'))
        dill.dump(dists, open(distname, 'wb'))

    def aggregate(self):
        '''
        Gather all data from domain analysis.
        '''

        files = glob.glob(self.save+'/splits/split_*')

        df = parallel(pd.read_csv, files)
        df = pd.concat(df)

        name = os.path.join(self.save, 'aggregate')
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, 'data.csv')
        df.to_csv(name, index=False)

        return df
