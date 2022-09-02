from pathos.multiprocessing import ProcessingPool as Pool
from matplotlib import pyplot as pl

from functools import partial
from sklearn import metrics
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import numpy as np

import sys
import os


def parallel(func, x, *args, **kwargs):
    '''
    Run some function in parallel.

    inputs:
        func = The function to apply.
        x = The list to iterate on.
        args = Additional arguemnts.
        kwargs = Additional key word arguments.

    outputs:
        data = A list of independent calculations.
    '''

    part_func = partial(func, *args, **kwargs)
    cores = os.cpu_count()
    with Pool(cores) as pool:
        data = list(tqdm(
                         pool.imap(part_func, x),
                         total=len(x),
                         file=sys.stdout
                         ))

    return data


def find(where, pattern):
    '''
    Function to find matching files.

    inputs:
        where = The top diretory to look for files.
        pattern = The pattern for files to look for.

    outputs:
        paths = A list of paths of files found.
    '''

    paths = list(map(str, Path(where).rglob(pattern)))

    return paths


def quantiles(groups, bins=100, std_y=1, control='points'):
    '''
    Get quantile values.
    '''

    group, values = groups

    if control == 'points':
        subbins = values.shape[0]//bins  # Number of quantiles
    elif control == 'quantiles':
        subbins = bins

    data = {
            'set': group,
            'rmse': [],
            'stdcal': [],
            'score': [],
            'errs': [],
            }

    if bins > 1:
        values['bin'] = pd.qcut(values['stdcal'], subbins)  # Quantile split

        for subgroup, subvalues in values.groupby('bin'):

            # RMSE
            err = metrics.mean_squared_error(
                                             subvalues['y'],
                                             subvalues['y_pred']
                                             )**0.5

            # Mean calibrated STDEV
            un = subvalues['stdcal'].mean()

            # Mean of dissimilarity metric
            dis = subvalues['score'].mean()

            # Normalization
            err /= std_y
            un /= std_y

            # Error in error
            err_in_errs = abs(err-un)

            data['rmse'].append(err)
            data['stdcal'].append(un)
            data['score'].append(dis)
            data['errs'].append(err_in_errs)

    else:
        err = abs(values['y']-values['y_pred'])
        un = values['stdcal']/std_y  # Normalized
        dis = values['gpr_std']/std_y  # Normalized
        err_in_errs = abs(err-un)

        data['rmse'] = err.tolist()
        data['stdcal'] = un.tolist()
        data['score'] = dis.tolist()
        data['errs'] = err_in_errs.tolist()

    return data


def ground_truth(data):
    '''
    Get the ground truth from data.

    inputs:
        data = A list of calibration data dictionaries.

    outputs:
        ecut = The ground truth as the max errror in error from CV.
        stdc = Calibrated STDEV point deviating from ideal.
    '''

    ecut = []
    stdc = []
    for i in data:
        if i['set'] == 'tr':
            ecut += i['errs']
            stdc += i['stdcal']
        else:
            continue

    # Return cutoffs as percentiles
    ecut = np.percentile(ecut, 95.45)
    stdc = np.percentile(stdc, 95.45)

    return ecut, stdc


def domain_pred(data, ecut, stdc):
    '''
    Define in domain based on ground truth.

    inputs:
        data = A list of calibration data dictionaries.
        ecut = Max error in error cutoff from CV.
        stdc = Calibrated STDEV point deviating from ideal.
    '''

    for i in data:

        in_domain = []
        for j, k in zip(i['errs'], i['stdcal']):

            if (j < ecut) and (k < stdc):
                in_domain.append(1)
            else:
                in_domain.append(0)

        i['in_domain'] = in_domain


def plot_calibration(data, stdc, ecut, save):
    '''
    Plot the calibration data.

    inputs:
        data = A list of calibration data dictionaries.
        stdc = Calibrated STDEV point deviating from ideal.
        ecut = Max error in error cutoff from CV.
        save = The directory to save.

    outputs:
        fig = The figure object.
        ax = The axis object.
    '''

    fig, ax = pl.subplots(1, 2)
    for i in data:
        if 'te' == i['set']:
            j = 1
        elif 'tr' == i['set']:
            j = 0
        else:
            continue

        x = np.array(i['stdcal'])
        y = np.array(i['rmse'])

        indx = np.array(i['errs']) <= ecut

        prop = ax[j].scatter(x[indx], y[indx], marker='.')
        ax[j].scatter(
                      x[~indx],
                      y[~indx],
                      facecolors='none',
                      edgecolors=prop.get_edgecolor()
                      )

    ax[0].set_title('Train')
    ax[1].set_title('Test')
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    for i in range(2):
        ax[i].set_xlabel(r'$\sigma_{cal}/\sigma_y$')
        ax[i].set_ylabel(r'$RMSE/\sigma_y$')

        lims = [*ax[i].get_xlim(), *ax[i].get_ylim()]
        ax[i].axline([min(lims)]*2, [max(lims)]*2, linestyle=':', color='k')
        ax[i].set_aspect('equal')
        ax[i].axvline(stdc, linestyle=':', color='r', label=r'$\sigma_{c}$')

    ax[0].legend()

    # Make same sale
    ax[0].set_xlim(ax[1].get_xlim())
    ax[0].set_ylim(ax[1].get_ylim())

    fig.savefig(os.path.join(save, 'calibration.png'))

    return fig, ax


def plot_score(data, stdc, ecut, save):
    '''
    Plot the error in erros versus the dissimilarity metric.

    inputs:
        data = A list of calibration data dictionaries.
        stdc = Calibrated STDEV point deviating from ideal.
        ecut = Max error in error cutoff from CV.
        save = The directory to save.

    outputs:
        fig = The figure object.
        ax = The axis object.
    '''

    fig, ax = pl.subplots(1, 2)
    for i in data:
        if 'te' == i['set']:
            j = 1
        elif 'tr' == i['set']:
            j = 0
        else:
            continue

        x = np.array(i['score'])
        y = np.array(i['errs'])

        indx = np.array(i['stdcal']) <= stdc

        prop = ax[j].scatter(x[indx], y[indx], marker='.')
        ax[j].scatter(
                      x[~indx],
                      y[~indx],
                      facecolors='none',
                      edgecolors=prop.get_edgecolor()
                      )

    ax[0].set_title('Train')
    ax[1].set_title('Test')
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    for i in range(2):
        ax[i].set_ylabel(r'$|RMSE-\sigma_{cal}|/\sigma_{y}$')
        ax[i].set_xlabel(r'GPR $\sigma$')
        ax[i].axhline(ecut, linestyle=':', color='r', label=r'$E_{c}$')

    ax[0].legend()

    # Make same sale
    ax[0].set_xlim(ax[1].get_xlim())
    ax[0].set_ylim(ax[1].get_ylim())

    fig.savefig(os.path.join(save, 'err_in_errs.png'))

    return fig, ax


def pr_sub(y_true, recall, precision, thresholds, title, save):
    '''
    Precision recall curve.
    '''

    baseline = sum(y_true)/len(y_true)
    auc_score = metrics.auc(recall, precision)
    auc_baseline = auc_score-baseline

    num = 2*recall*precision
    den = recall+precision
    f1_scores = np.divide(num, den, out=np.zeros_like(den), where=(den != 0))
    max_f1 = np.max(f1_scores)
    max_f1_thresh = thresholds[np.argmax(f1_scores)]

    label = 'AUC={:.2f}\n'.format(auc_score)
    label += r'$\Delta$AUC='+'{:.2f}\n'.format(auc_baseline)
    label += 'Max F1={:.2f}\n'.format(max_f1)
    label += 'Max F1 Threshold={:.2f}'.format(max_f1_thresh)

    fig, ax = pl.subplots()
    ax.set_title(title)
    ax.plot(recall, precision, label=label)
    ax.axhline(
               baseline,
               color='r',
               linestyle=':',
               label='Baseline={:.2f}'.format(baseline)
               )
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()

    fig.savefig(save)


def plot_pr(data, save):
    '''
    Plot precision recall curve.

    inputs:
        data = A list of calibration data dictionaries.
        save = The directory to save.

    outputs:
        fig = The figure object.
        ax = The axis object.
    '''

    # Hold all data needed for PR
    pr = {
          'te_id_domain': [],
          'te_id_score': [],
          'te_od_score': [],
          'tr_id_domain': [],
          'tr_id_score': [],
          'tr_od_score': [],
          }

    for i in data:
        if i['set'] == 'te':
            pr['te_id_domain'] += i['in_domain']
            pr['te_od_score'] += i['score']

        elif i['set'] == 'tr':
            pr['tr_id_domain'] += i['in_domain']
            pr['tr_od_score'] += i['score']

        else:
            continue

    # Score for ID score (just need small values to be large and vice versa)
    pr['te_id_score'] = [np.exp(-i) for i in pr['te_od_score']]
    pr['tr_id_score'] = [np.exp(-i) for i in pr['tr_od_score']]

    # Switch class labels for OD
    pr['te_od_domain'] = [1 if i == 0 else 0 for i in pr['te_id_domain']]
    pr['tr_od_domain'] = [1 if i == 0 else 0 for i in pr['tr_id_domain']]

    # Compute precision recall
    te_id_pr = metrics.precision_recall_curve(
                                              pr['te_id_domain'],
                                              pr['te_id_score'],
                                              )

    te_od_pr = metrics.precision_recall_curve(
                                              pr['te_od_domain'],
                                              pr['te_od_score'],
                                              )

    tr_id_pr = metrics.precision_recall_curve(
                                              pr['tr_id_domain'],
                                              pr['tr_id_score'],
                                              )

    tr_od_pr = metrics.precision_recall_curve(
                                              pr['tr_od_domain'],
                                              pr['tr_od_score'],
                                              )

    # Convert thresholds back to GPR STDEV
    te_id_pr = list(te_id_pr)
    tr_id_pr = list(tr_id_pr)
    te_id_pr[2] = np.log(1/te_id_pr[2])
    tr_id_pr[2] = np.log(1/tr_id_pr[2])

    pr_sub(
           pr['te_od_domain'],
           te_od_pr[1],
           te_od_pr[0],
           te_od_pr[2],
           'Test OD as +',
           os.path.join(save, 'pr_od_test.png')
           )
    pr_sub(
           pr['tr_od_domain'],
           tr_od_pr[1],
           tr_od_pr[0],
           tr_od_pr[2],
           'Train OD as +',
           os.path.join(save, 'pr_od_train.png')
           )
    pr_sub(
           pr['te_id_domain'],
           te_id_pr[1],
           te_id_pr[0],
           te_id_pr[2],
           'Test ID as +',
           os.path.join(save, 'pr_id_test.png')
           )
    pr_sub(
           pr['tr_id_domain'],
           tr_id_pr[1],
           tr_id_pr[0],
           tr_id_pr[2],
           'Train ID as +',
           os.path.join(save, 'pr_id_train.png')
           )


def make_plots(save, xaxis, dist):
    '''
    The general workflow.
    '''

    bins = 10
    control = 'quantiles'

    df = pd.read_csv(os.path.join(save, 'aggregate/data.csv'))
    df = df.sort_values(by=['stdcal', 'y_pred', dist])

    # Get values from training CV
    df_tr = df[df['in_domain'] == 'tr']
    std_y = df_tr['y'].std()

    df['score'] = df[dist]

    # Grab quantile data for each set of runs
    group = df.groupby(['in_domain'], sort=False)
    data = parallel(quantiles, group, bins=bins, std_y=std_y, control=control)

    # Grab ground truth from validation sets and assign class
    ecut, stdc = ground_truth(data)
    domain_pred(data, ecut, stdc)

    # Make plots
    save = [
            save,
            'aggregate',
            'plots',
            'total',
            'calibration',
            ]

    save = os.path.join(*save)
    os.makedirs(save, exist_ok=True)

    plot_calibration(data, stdc, ecut, save)
    plot_score(data, stdc, ecut, save)
    plot_pr(data, save)
