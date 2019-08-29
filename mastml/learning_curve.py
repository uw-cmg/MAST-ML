"""
This module contains methods to construct learning curves, which evaluate some cross-validation performance metric (e.g. RMSE)
as a function of amount of training data (i.e. a sample learning curve) or as a function of the number of features used
in the fitting (i.e. a feature learning curve).
"""

import numpy as np
import pandas as pd
import warnings
import logging
import os

from sklearn.model_selection import learning_curve
from sklearn.feature_selection import f_regression

from mastml.legos import feature_selectors as fs

# Ignore the harmless warning about the gelsd driver on mac.
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")

log = logging.getLogger('mastml')

def sample_learning_curve(X, y, estimator, cv, scoring, Xgroups=None):
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

    train_sizes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    if Xgroups.shape[0] > 0:
        Xgroups = np.array(Xgroups).reshape(-1, )
    else:
        Xgroups = np.zeros(len(y))

    train_sizes, train_scores, valid_scores = learning_curve(estimator=estimator, X=X, y=y, train_sizes=train_sizes,
                                                             scoring=scoring, cv=cv, groups=Xgroups)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(valid_scores, axis=1)
    train_stdev = np.std(train_scores, axis=1)
    test_stdev = np.std(valid_scores, axis=1)

    return train_sizes, train_mean, test_mean, train_stdev, test_stdev

def feature_learning_curve(X, y, estimator, cv, scoring, selector_name, savepath, n_features_to_select=None, Xgroups=None):
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
    if Xgroups.shape[0] > 0:
        Xgroups = np.array(Xgroups).reshape(-1, )
    else:
        Xgroups = np.zeros(len(y))
    train_mean = list()
    train_stdev = list()
    test_mean = list()
    test_stdev = list()
    if not n_features_to_select:
        n_features_to_select = X.shape[1].tolist()
    train_sizes = range(n_features_to_select)
    train_sizes = [1+f for f in train_sizes]
    features_selected = list()
    n_features = list()
    for feature in train_sizes:
        n_features.append(feature)
        if selector_name == 'RFE':
            log.warning("Using RFE as feature selector does not support a custom CV or grouping scheme. Your learning"
                        "curve will be generated properly, but will not use the custom CV or grouping scheme")
            try:
                Xnew = fs.name_to_constructor[selector_name](estimator, feature).fit(X, y).transform(X)
            except RuntimeError:
                log.error("You have specified an estimator for RFE that does not have a coef_ or feature_importances_ attribute. "
                          "Acceptable models to use with RFE include: LinearRegression, Lasso, SVR, DecisionTreeRegressor, "
                          "RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, etc.")
        elif selector_name == 'SelectKBest':
            log.warning("Using SelectKBest as feature selector does not support a custom estimator model, CV or grouping scheme. "
                        "Your learning curve will be generated properly, but will not use the custom model, CV or grouping scheme")
            Xnew = fs.name_to_constructor[selector_name](f_regression, feature).fit(X, y).transform(X)
        elif selector_name == 'MASTMLFeatureSelector':
            Xnew = fs.name_to_constructor[selector_name](estimator, feature, cv).fit(X, y, pd.DataFrame(Xgroups)).transform(X)
        elif selector_name == 'SequentialFeatureSelector':
            log.warning("Using SequentialFeatureSelector as feature selector does not support a custom CV or grouping scheme. "
                        "Your learning curve will be generated properly, but will not use the custom CV or grouping scheme")
            Xnew = fs.name_to_constructor[selector_name](estimator, feature).fit(pd.DataFrame(X), pd.DataFrame(y)).transform(X)
        elif selector_name == None:
            log.warning("A selector name for learning curve calculation was not found. Defaulting to using the "
                        "MASTMLFeatureSelector for learning curve")
            Xnew = fs.name_to_constructor["MASTMLFeatureSelector"](estimator, feature, cv).fit(X, y, pd.DataFrame(Xgroups)).transform(X)
        else:
            log.error("You have specified an invalid selector_name for learning curve. Either leave blank to use the default"
                      " MASTMLFeatureSelector or use one of SelectKBest, RFE, SequentialFeatureSelector, MASTMLFeatureSelector")
            exit()
        # Need to use arrays to avoid indexing issues when leaving out validation data
        features_selected.append(Xnew.columns.tolist())

        Xnew = np.array(Xnew)
        y = np.array(y)
        Xgroups = np.array(Xgroups)
        cv_number=1
        train_scores = dict()
        test_scores = dict()
        for trains, tests in cv.split(Xnew, y, Xgroups):
            model = estimator.fit(Xnew[trains], y[trains])
            train_vals = model.predict(Xnew[trains])
            test_vals = model.predict(Xnew[tests])
            train_scores[cv_number] = scoring._score_func(train_vals, y[trains])
            test_scores[cv_number] = scoring._score_func(test_vals, y[tests])
            cv_number += 1
        train_mean.append(np.mean(list(train_scores.values())))
        train_stdev.append(np.std(list(train_scores.values())))
        test_mean.append(np.mean(list(test_scores.values())))
        test_stdev.append(np.std(list(test_scores.values())))

    datadict = {"n_features": n_features, "features selected": features_selected}
    pd.DataFrame().from_dict(data=datadict).to_csv(os.path.join(savepath, 'features_selected_in_learning_curve.csv'))

    return np.array(train_sizes), np.array(train_mean), np.array(test_mean), np.array(train_stdev), np.array(test_stdev)