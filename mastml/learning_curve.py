"""
This module contains methods to construct learning curves, which evaluate some cross-validation performance metric (e.g. RMSE)
as a function of amount of training data (i.e. a sample learning curve) or as a function of the number of features used
in the fitting (i.e. a feature learning curve).
"""

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold

from mastml.metrics import Metrics
from mastml.feature_selectors import SklearnFeatureSelector
from mastml.plots import Line

class LearningCurve():
    '''

    '''
    def __init__(self):
        pass

    def evaluate(self, model, X, y, savepath=None, groups=None, train_sizes=None, cv=None, scoring=None, selector=None,
                            make_plot=True):
        if savepath is None:
            savepath = os.getcwd()
        splitdir = self._setup_savedir(savepath=savepath)
        self.data_learning_curve(model=model,
                                 X=X,
                                 y=y,
                                 savepath=splitdir,
                                 groups=groups,
                                 train_sizes=train_sizes,
                                 cv=cv,
                                 scoring=scoring,
                                 make_plot=make_plot)
        self.feature_learning_curve(model=model,
                                    X=X,
                                    y=y,
                                    savepath=splitdir,
                                    groups=groups,
                                    cv=cv,
                                    scoring=scoring,
                                    selector=selector,
                                    make_plot=make_plot)
        return

    def data_learning_curve(self, model, X, y, savepath=None, groups=None, train_sizes=None, cv=None, scoring=None,
                            make_plot=True):
        """
        Method that calculates data used to plot a sample learning curve, e.g. the RMSE of a cross-validation routine using a
        specified model and a given fraction of the total training data

        Args:
            X: (numpy array), array of X data values

            y: (numpy array), array of y data values

            estimator: (scikit-learn model object), a scikit-learn model used for fitting

            cv: (scikit-learn cross validation object), a scikit-learn cross validation object to construct train/test splits

            scoring: (scikit-learn metric object), a scikit-learn metric to use as a scorer

            Xgroups: (list), list of row indices corresponding to each group

        Returns:
            train_sizes: (numpy array), array of fractions of training data used in learning curve

            train_mean: (numpy array), array of means of training data scores for each training data fraction

            test_mean: (numpy array), array of means of testing data scores for each training data fraction

            train_stdev: (numpy array), array of standard deviations of training data scores for each training data fraction

            test_stdev: (numpy array), array of standard deviations of testing data scores for each training data fraction

        """
        if savepath is None:
            savepath = os.getcwd()
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 5)
        if cv is None:
            cv = 5
        if model.__class__.__name__=='SklearnModel':
            model = model.model
        metrics = Metrics(metrics_list=None)._metric_zoo()
        if scoring is None:
            score_name = 'mean_absolute_error'
            scoring = make_scorer(metrics['mean_absolute_error'][1], greater_is_better=True) #Note using True b/c if False then sklearn multiplies by -1
        else:
            score_name = scoring
            scoring = make_scorer(metrics[scoring][1], greater_is_better=True) #Note using True b/c if False then sklearn multiplies by -1

        train_sizes, train_scores, valid_scores = learning_curve(estimator=model,
                                                                 X=X,
                                                                 y=y,
                                                                 train_sizes=train_sizes,
                                                                 scoring=scoring,
                                                                 cv=cv,
                                                                 groups=groups)

        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(valid_scores, axis=1)
        train_stdev = np.std(train_scores, axis=1)
        test_stdev = np.std(valid_scores, axis=1)

        datadict = {"train_sizes": train_sizes,
                    "train_mean": train_mean,
                    "train_std": train_stdev,
                    "test_mean": test_mean,
                    "test_std": test_stdev}
        pd.DataFrame().from_dict(data=datadict).to_excel(os.path.join(savepath, 'data_learning_curve.xlsx'), index=False)

        if make_plot is True:
            Line().plot_learning_curve(train_sizes=train_sizes,
                                       train_mean=train_mean,
                                       test_mean=test_mean,
                                       train_stdev=train_stdev,
                                       test_stdev=test_stdev,
                                       learning_curve_type='data_learning_curve',
                                       score_name=score_name,
                                       savepath=savepath)

        return

    def feature_learning_curve(self, model, X, y, savepath=None, groups=None, cv=None, scoring=None, selector=None, make_plot=True):
        """
        Method that calculates data used to plot a feature learning curve, e.g. the RMSE of a cross-validation routine using a
        specified model and a given number of features

        Args:
            X: (numpy array), array of X data values

            y: (numpy array), array of y data values

            estimator: (scikit-learn model object), a scikit-learn model used for fitting

            cv: (scikit-learn cross validation object), a scikit-learn cross validation object to construct train/test splits

            scoring: (scikit-learn metric object), a scikit-learn metric to use as a scorer

            selector_name: (str), name of a scikit-learn or MAST-ML feature selection routine

            n_features_to_select: (int), total number of features to select, i.e. stopping criterion for number of features

            Xgroups: (list), list of row indices corresponding to each group

        Returns:
            train_sizes: (numpy array), array of fractions of training data used in learning curve

            train_mean: (numpy array), array of means of training data scores for each number of features

            test_mean: (numpy array), array of means of testing data scores for each number of features

            train_stdev: (numpy array), array of standard deviations of training data scores for each number of features

            test_stdev: (numpy array), array of standard deviations of testing data scores for each number of features

        """

        if savepath is None:
            savepath = os.getcwd()
        if cv is None:
            cv = KFold(n_splits=5, shuffle=True)
        if model.__class__.__name__ == 'SklearnModel':
            model = model.model

        splits = cv.split(X, y, groups)
        train_inds = list()
        test_inds = list()
        for train, test in splits:
            train_inds.append(train)
            test_inds.append(test)

        metrics = Metrics(metrics_list=None)._metric_zoo()
        if scoring is None:
            score_name = 'mean_absolute_error'
            scoring = make_scorer(metrics['mean_absolute_error'][1], greater_is_better=True) #Note using True b/c if False then sklearn multiplies by -1
        else:
            score_name = scoring
            scoring = make_scorer(metrics[scoring][1], greater_is_better=True) #Note using True b/c if False then sklearn multiplies by -1

        if selector is None:
            selector_name = 'SequentialFeatureSelector'
            selector = SklearnFeatureSelector(selector='SequentialFeatureSelector',
                                              estimator=model,
                                              n_features_to_select=X.shape[1],
                                              scoring=scoring,
                                              cv=cv)
        else:
            try:
                selector_name = selector.selector.__class__.__name__
            except:
                selector_name = selector.__class__.__name__

        train_mean = list()
        train_stdev = list()
        test_mean = list()
        test_stdev = list()

        if selector_name == 'RFE':
            print("Using RFE as feature selector does not support a custom CV or grouping scheme. Your learning "
                        "curve will be generated properly, but will not use the custom CV or grouping scheme")
            try:
                Xnew = selector.fit(X=X, y=y).transform(X=X)
            except RuntimeError:
                print("You have specified an estimator for RFE that does not have a coef_ or feature_importances_ attribute. "
                          "Acceptable models to use with RFE include: LinearRegression, Lasso, SVR, DecisionTreeRegressor, "
                          "RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, etc.")
                sys.exit()
        elif selector_name == 'SelectKBest':
            print("Using SelectKBest as feature selector does not support a custom estimator model, CV or grouping scheme. "
                        "Your learning curve will be generated properly, but will not use the custom model, CV or grouping scheme")
            Xnew = selector.fit(X=X, y=y).transform(X=X)
        else:
            try:
                Xnew = selector.fit(X=X, y=y).transform(X=X)
            except:
                print("You have specified an invalid selector_name for learning curve. Either leave blank to use the default"
                      " SequentialFeatureSelector or use one of SelectKBest, RFE, SequentialFeatureSelector, MASTMLFeatureSelector")
                sys.exit()

        # save selected features for each iteration to text file
        with open(os.path.join(savepath, 'selected_features.txt'), 'w') as f:
            features_selected = Xnew.columns.tolist()
            for feature in features_selected:
                f.write(feature+'\n')

        train_scores = dict()
        test_scores = dict()
        train_sizes = list()
        y = np.array(y)
        for n_features in range(len(features_selected)):
            train_sizes.append(n_features+1)
            Xnew_subset = Xnew.iloc[:, 0:n_features+1]

            cv_number = 1
            Xnew_subset = np.array(Xnew_subset)
            if n_features+1 == 1:
                Xnew_subset.reshape(-1, 1)

            for trains, tests in zip(train_inds, test_inds):
                model = model.fit(Xnew_subset[trains], y[trains])
                train_vals = model.predict(Xnew_subset[trains])
                test_vals = model.predict(Xnew_subset[tests])
                train_scores[cv_number] = scoring._score_func(train_vals, y[trains])
                test_scores[cv_number] = scoring._score_func(test_vals, y[tests])
                cv_number += 1
            train_mean.append(np.mean(list(train_scores.values())))
            train_stdev.append(np.std(list(train_scores.values())))
            test_mean.append(np.mean(list(test_scores.values())))
            test_stdev.append(np.std(list(test_scores.values())))

        train_sizes = np.array(train_sizes)
        train_mean = np.array(train_mean)
        train_stdev = np.array(train_stdev)
        test_mean = np.array(test_mean)
        test_stdev = np.array(test_stdev)

        datadict = {"train_sizes": train_sizes,
                    "features selected": features_selected,
                    "train_mean": train_mean,
                    "train_std": train_stdev,
                    "test_mean": test_mean,
                    "test_std": test_stdev}
        pd.DataFrame().from_dict(data=datadict).to_excel(os.path.join(savepath, 'feature_learning_curve.xlsx'), index=False)

        if make_plot is True:
            Line().plot_learning_curve(train_sizes=train_sizes,
                                       train_mean=train_mean,
                                       test_mean=test_mean,
                                       train_stdev=train_stdev,
                                       test_stdev=test_stdev,
                                       learning_curve_type='feature_learning_curve',
                                       score_name=score_name,
                                       savepath=savepath)

        return

    def _setup_savedir(self, savepath):
        now = datetime.now()
        dirname = 'LearningCurve'
        dirname = f"{dirname}_{now.month:02d}_{now.day:02d}" \
                        f"_{now.hour:02d}_{now.minute:02d}_{now.second:02d}"
        if savepath == None:
            splitdir = os.getcwd()
        else:
            splitdir = os.path.join(savepath, dirname)
        if not os.path.exists(splitdir):
            os.mkdir(splitdir)
        self.splitdir = splitdir
        return splitdir
