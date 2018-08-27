import numpy as np
import pandas as pd
import warnings
import logging

from sklearn.model_selection import learning_curve
from sklearn.feature_selection import f_regression

from mastml.legos import feature_selectors as fs

# Ignore the harmless warning about the gelsd driver on mac.
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")

log = logging.getLogger('mastml')

def sample_learning_curve(X, y, estimator, cv, scoring, Xgroups=None):
    if Xgroups is not None:
        Xgroups = np.array(Xgroups).reshape(-1, )
    train_sizes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    train_sizes, train_scores, valid_scores = learning_curve(estimator=estimator, X=X, y=y, train_sizes=train_sizes,
                                                             scoring=scoring, cv=cv, groups=Xgroups)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(valid_scores, axis=1)
    train_stdev = np.std(train_scores, axis=1)
    test_stdev = np.std(valid_scores, axis=1)

    return train_sizes, train_mean, test_mean, train_stdev, test_stdev

def feature_learning_curve(X, y, estimator, cv, scoring, selector_name, n_features_to_select=None, Xgroups=None):
    if Xgroups is not None:
        Xgroups = np.array(Xgroups).reshape(-1, )
    train_mean = list()
    train_stdev = list()
    test_mean = list()
    test_stdev = list()
    if not n_features_to_select:
        n_features_to_select = X.shape[1].tolist()
    train_sizes = range(n_features_to_select)
    train_sizes = [1+f for f in train_sizes]
    for feature in train_sizes:
        if selector_name == 'RFE':
            log.warning("Using RFE as feature selector does not support a custom CV or grouping scheme. Your learning"
                        "curve will be generated properly, but will not use the custom CV or grouping scheme")
            try:
                Xnew = fs.name_to_constructor[selector_name](estimator, feature+1).fit(X, y).transform(X)
            except RuntimeError:
                log.error("You have specified an estimator for RFE that does not have a coef_ or feature_importances_ attribute. "
                          "Acceptable models to use with RFE include: LinearRegression, Lasso, SVR, DecisionTreeRegressor, "
                          "RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, etc.")
        elif selector_name == 'SelectKBest':
            log.warning("Using SelectKBest as feature selector does not support a custom estimator model, CV or grouping scheme. "
                        "Your learning curve will be generated properly, but will not use the custom model, CV or grouping scheme")
            Xnew = fs.name_to_constructor[selector_name](f_regression, feature+1).fit(X, y).transform(X)
        elif selector_name == 'MASTMLFeatureSelector':
            Xnew = fs.name_to_constructor[selector_name](estimator, feature+1, cv).fit(X, y, pd.DataFrame(Xgroups)).transform(X)
        elif selector_name == 'SequentialFeatureSelector':
            log.warning("Using SequentialFeatureSelector as feature selector does not support a custom CV or grouping scheme. "
                        "Your learning curve will be generated properly, but will not use the custom CV or grouping scheme")
            Xnew = fs.name_to_constructor[selector_name](estimator, feature+1).fit(pd.DataFrame(X), pd.DataFrame(y)).transform(X)
        elif selector_name == None:
            log.warning("A selector name for learning curve calculation was not found. Defaulting to using the "
                        "MASTMLFeatureSelector for learning curve")
            Xnew = fs.name_to_constructor["MASTMLFeatureSelector"](estimator, feature+1, cv).fit(X, y, pd.DataFrame(Xgroups)).transform(X)
        else:
            log.error("You have specified an invalid selector_name for learning curve. Either leave blank to use the default"
                      " MASTMLFeatureSelector or use one of SelectKBest, RFE, SequentialFeatureSelector, MASTMLFeatureSelector")
            exit()
        # Need to use arrays to avoid indexing issues when leaving out validation data
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
    return np.array(train_sizes), np.array(train_mean), np.array(test_mean), np.array(train_stdev), np.array(test_stdev)