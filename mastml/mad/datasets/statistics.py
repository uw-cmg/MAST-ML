from sklearn import metrics
from pathlib import Path

import pandas as pd
import numpy as np
import os

from mad.functions import parallel


def eval_reg_metrics(groups):
    '''
    Evaluate standard regression prediction metrics.
    '''

    group, df = groups

    y = df['y']
    y_pred = df['y_pred']

    rmse = metrics.mean_squared_error(y, y_pred)**0.5
    rmse_sig = rmse/np.std(y)
    mae = metrics.mean_absolute_error(y, y_pred)
    r2 = metrics.r2_score(y, y_pred)

    in_domain, test_count, domain = group

    results = {}
    results['in_domain'] = in_domain
    results['domain'] = domain
    results['test_count'] = test_count
    results[r'$RMSE$'] = rmse
    results[r'$RMSE/\sigma$'] = rmse_sig
    results[r'$MAE$'] = mae
    results[r'$R^{2}$'] = r2

    return results


def group_metrics(df, cols):
    '''
    Get the metrics statistics.
    '''

    groups = df.groupby(cols, dropna=False)
    mets = parallel(eval_reg_metrics, groups)
    mets = pd.DataFrame(mets)

    return mets


def stats(df, cols):
    '''
    Get the statistic of a dataframe.
    '''

    groups = df.groupby(cols)
    mean = groups.mean().add_suffix('_mean')
    sem = groups.sem().add_suffix('_sem')
    count = groups.count().add_suffix('_count')
    df = mean.merge(sem, on=cols)
    df = df.merge(count, on=cols)
    df = df.reset_index()

    return df


def folds_opperation(save, file_name):
    '''
    Save the true values, predicted values, distances, and model error.

    inputs:
        save = The directory to save and where split data are.
    '''

    save = os.path.join(save, 'aggregate')

    # Load
    df_path = os.path.join(save, file_name)
    df = pd.read_csv(df_path)

    # Get statistics
    dfstats = stats(df, [
                         'index',
                         'in_domain',
                         'domain'
                         ])

    # Get metrics
    mets = group_metrics(df, [
                              'in_domain',
                              'test_count',
                              'domain',
                              ])

    metsstats = mets.drop(['test_count'], axis=1)
    metsstats = stats(metsstats, [
                                  'in_domain',
                                  'domain',
                                  ])

    # Save data
    dfstats.to_csv(
                   os.path.join(save, 'data_stats.csv'),
                   index=False
                   )
    mets.to_csv(
                os.path.join(save, 'metrics.csv'),
                index=False
                )
    metsstats.to_csv(
                     os.path.join(save, 'metrics_stats.csv'),
                     index=False
                     )


def folds(save):
    '''
    Save the true values, predicted values, distances, and model error.

    inputs:
        save = The directory to save and where split data are.
    '''

    folds_opperation(save, 'data.csv')
