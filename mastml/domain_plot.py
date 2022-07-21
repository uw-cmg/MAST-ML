from pathos.multiprocessing import ProcessingPool as Pool
from matplotlib import pyplot as pl

from functools import partial
from sklearn import metrics
from cycler import cycler
from pathlib import Path
from tqdm import tqdm
from sys import argv

import pandas as pd
import numpy as np

import matplotlib
import json
import sys
import os

# Font styles
font = {'font.size': 15, 'lines.markersize': 10}
matplotlib.rcParams.update(font)

# To make sure runs are plot with same symbols and colors 
cc = (
      cycler(color=[
                    'tab:blue',
                    'tab:red',
                    'tab:green',
                    'tab:orange',
                    'tab:purple',
                    'tab:brown',
                    'tab:olive',
                    'slategray',
                    'chartreuse',
                    'moccasin',
                    'darkcyan',
                    'rosybrown',
                    'salmon',
                    ]
             ) *
      cycler(marker=[
                     '.',
                     ]
             )
      )


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
    #paths = [i for i in paths if ('random' in i) or ('chemical' in i)]

    return paths


def load(path):
    '''
    Load data from path.

    inputs:
        path = The path to csv file.

    outputs:
        df = A pandas dataframe.
    '''

    stdcal = pd.read_csv(path)
    stdcal.columns = ['stdcal']

    dist = pd.read_csv(path.replace(
                                    'model_errors_leaveout_calibrated.csv',
                                    'dist_train_to_leaveout.csv'
                                    ))


    y = pd.read_csv(path.replace(
                                 'model_errors_leaveout_calibrated.csv',
                                 'y_leaveout.csv'
                                 ))
    y.columns = ['y']

    y_pred = pd.read_csv(path.replace(
                                      'model_errors_leaveout_calibrated.csv',
                                      'y_pred_leaveout.csv'
                                      ))

    y_pred.columns = ['y_pred']


    df = pd.concat([dist, stdcal, y, y_pred], axis=1)
    print(df)

    return df


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
            'run': group[0],
            'set': group[1],
            'domain': group[2],
            'color': group[3],
            'marker': group[4],
            'rmse': [],
            'stdcal': [],
            'score': [],
            'errs': [],
            'count': [],
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

            # Counts of each bin
            count = subvalues['stdcal'].count()

            # Normalization
            err /= std_y
            un /= std_y

            # Error in error
            err_in_errs = abs(err-un)

            data['rmse'].append(err)
            data['stdcal'].append(un)
            data['score'].append(dis)
            data['errs'].append(err_in_errs)
            data['count'].append(count)

    else:
        err = abs(values['y']-values['y_pred'])/std_y
        un = values['stdcal']/std_y  # Normalized
        dis = values['score']  # Normalized
        err_in_errs = abs(err-un)

        data['rmse'] = err.tolist()
        data['stdcal'] = un.tolist()
        data['score'] = dis.tolist()
        data['errs'] = err_in_errs.tolist()
        data['count'] = len(err)*[1]

    return data


def ground_truth(data, perc_stdc, perc_ecut):
    '''
    Get the ground truth from data.

    inputs:
        data = A list of calibration data dictionaries.
        perc_stdc = The percentile cutoff for calbirated uncertainties.
        perc_ecut = The percentile cutoff for error in errors.

    outputs:
        ecut = The ground truth as the max errror in error from CV.
        stdc = Calibrated STDEV point deviating from ideal.
    '''

    ecut = []
    stdc = []
    counts = []
    for i in data:
        if i['set'] == 'tr':
            ecut += i['errs']
            stdc += i['stdcal']
            counts += i['count']
        else:
            continue

    # Weigh from quantile points
    ecut_mod = []
    stdc_mod = []
    for i, j, k in zip(ecut, stdc, counts):
        ecut_mod += [i]*k
        stdc_mod += [j]*k

    # Return cutoffs as percentiles
    stdc = np.percentile(stdc_mod, perc_stdc)
    ecut = np.percentile(ecut_mod, perc_ecut)

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


def cdf_parity(x, ax, fig_data):
    '''
    Plot the quantile quantile plot for cummulative distributions.

    inputs:
        x = The residuals normalized by the calibrated uncertainties.
    '''

    n = len(x)
    z = np.random.normal(0, 1, n)  # Standard normal distribution

    # Need sorting
    x = sorted(x)
    z = sorted(z)

    # Cummulative fractions
    frac = np.arange(n)/n

    # Interpolation to compare cdf
    eval_points = sorted(list(set(x+z)))
    y_pred = np.interp(eval_points, x, frac)  # Predicted
    y = np.interp(eval_points, z, frac)  # Standard Normal

    y_pred = y_pred.tolist()
    y = y.tolist()

    ax.plot(
            y,
            y_pred,
            zorder=0,
            label='Data'
            )

    # Line of best fit
    ax.plot(
            [0, 1],
            [0, 1],
            label=r'$y=\hat{y}$ Line',
            color='k',
            linestyle=':',
            zorder=1
            )

    ax.set_aspect('equal')

    ax.set_ylabel('Predicted CDF')
    ax.set_xlabel('Standard Normal CDF')

    fig_data['y_pred'] = y_pred
    fig_data['y'] = y


def plot_qq(df):
    '''
    Plot the quantile quantile plot for calibrated uncertainties.

    inputs:
        df = Accumulated data from all of the nested CV.
    '''

    fig, ax = pl.subplots(1, 2)
    fig_data = {}
    for group, values in df.groupby(['in_domain', 'run', 'domain']):

        # Make absolute residuals normalized by model uncertainties
        stdcal = values['stdcal'].values
        y = values['y']
        y_pred = values['y_pred']
        res = y-y_pred
        x = res/stdcal

        group_name = '_'.join(map(str, group))
        fig_data[group_name] = {}
        fig_data[group_name]['in_domain'] = group[0]
        fig_data[group_name]['run'] = group[1]
        fig_data[group_name]['domain'] = group[2]
        if group[0] == 'tr':
            cdf_parity(x, ax[0], fig_data[group_name])
        elif group[0] == 'te':
            cdf_parity(x, ax[1], fig_data[group_name])

    ax[0].set_title('Train')
    ax[1].set_title('Test')
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    fig.tight_layout()

    save = 'figures/calibration/'
    os.makedirs(save, exist_ok=True)
    fig.savefig(save+'qq.png')
    pl.close('all')

    jsonfile = '{}.json'.format(save)
    with open(save+'qq.json', 'w') as handle:
        json.dump(fig_data, handle)


def plot_calibration(data, stdc, ecut):
    '''
    Plot the calibration data.

    inputs:
        data = A list of calibration data dictionaries.
        stdc = Calibrated STDEV point deviating from ideal.
        ecut = Max error in error cutoff from CV.

    outputs:
        fig = The figure object.
        ax = The axis object.
    '''

    fig_data = []
    fig, ax = pl.subplots(1, 2)
    for i in data:
        if 'te' == i['set']:
            j = 1
        elif 'tr' == i['set']:
            j = 0
        else:
            continue

        marker = i['marker']
        color = i['color']

        x = np.array(i['stdcal'])
        y = np.array(i['rmse'])

        indx = np.array(i['errs']) <= ecut

        ax[j].scatter(x[indx], y[indx], color=color, marker=marker)
        ax[j].scatter(
                      x[~indx],
                      y[~indx],
                      facecolors='none',
                      edgecolors=color,
                      marker=marker,
                      )

        # For plot data export
        fig_data.append({
                         'set': i['set'],
                         'run': i['run'],
                         'xopen': x[~indx].tolist(),
                         'yopen': y[~indx].tolist(),
                         'xclosed': x[indx].tolist(),
                         'yclosed': y[indx].tolist()
                         })

    # Make same sale
    xlim_min = min([ax[0].get_xlim()[0], ax[1].get_xlim()[0]])
    xlim_max = max([ax[0].get_xlim()[1], ax[1].get_xlim()[1]])
    ylim_min = min([ax[0].get_ylim()[0], ax[1].get_ylim()[0]])
    ylim_max = max([ax[0].get_ylim()[1], ax[1].get_ylim()[1]])
    lims = [
            *ax[0].get_xlim(),
            *ax[0].get_ylim(),
            *ax[1].get_xlim(),
            *ax[1].get_ylim()
            ]
    lim_min = [min(lims)]*2
    lim_max = [max(lims)]*2

    ax[0].set_title('Train')
    ax[1].set_title('Test')
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    for i in range(2):
        ax[i].set_xlim(xlim_min, xlim_max)
        ax[i].set_ylim(ylim_min, ylim_max)

        ax[i].set_xlabel(r'$\sigma_{cal}/\sigma_y$')
        ax[i].set_ylabel(r'$RMSE/\sigma_y$')

        ax[i].axline(lim_min, lim_max, linestyle=':', color='k')
        ax[i].set_aspect('equal')
        ax[i].axvline(stdc, linestyle=':', color='r', label=r'$\sigma_{c}$')

    save = 'figures/calibration/'
    os.makedirs(save, exist_ok=True)
    fig.savefig(save+'calibration.png')
    pl.close('all')
    with open(save+'calibration.json', 'w') as handle:
        json.dump(fig_data, handle)


def plot_gt_pred(data, stdc, ecut):
    '''
    Plot the error in erros versus the dissimilarity metric.

    inputs:
        data = A list of calibration data dictionaries.
        stdc = Calibrated STDEV point deviating from ideal.
        ecut = Max error in error cutoff from CV.
    '''

    for k in range(2):

        fig_data = []
        fig, ax = pl.subplots(1, 2)
        for i in data:
            if 'te' == i['set']:
                j = 1
            elif 'tr' == i['set']:
                j = 0
            else:
                continue

            y = np.array(i['stdcal'])
            if k == 0:
                x = np.array(i['errs'])
                indx = (y <= stdc) & (x <= ecut)
            elif k == 1:
                x = np.array(i['score'])
                indx = (y <= stdc)

            marker = i['marker']
            color = i['color']
            ax[j].scatter(
                          x[indx],
                          y[indx],
                          marker=marker,
                          color=color,
                          label=str(i['set'])+'_'+str(i['domain'])+'_'+str(i['run'])
                          )
            ax[j].scatter(
                          x[~indx],
                          y[~indx],
                          facecolors='none',
                          marker=marker,
                          edgecolors=color,
                          )

            # For plot data export
            fig_data.append({
                             'set': i['set'],
                             'run': i['run'],
                             'xopen': x[~indx].tolist(),
                             'yopen': y[~indx].tolist(),
                             'xclosed': x[indx].tolist(),
                             'yclosed': y[indx].tolist(),
                             'stdc': stdc,
                             'ecut': ecut
                             })

        ax[0].set_title('Train')
        ax[1].set_title('Test')
        ax[1].yaxis.tick_right()
        ax[1].yaxis.set_label_position("right")

        # Make same sale
        xlim_min = min([ax[0].get_xlim()[0], ax[1].get_xlim()[0]])
        xlim_max = max([ax[0].get_xlim()[1], ax[1].get_xlim()[1]])
        ylim_min = min([ax[0].get_ylim()[0], ax[1].get_ylim()[0]])
        ylim_max = max([ax[0].get_ylim()[1], ax[1].get_ylim()[1]])

        for q in range(2):
            ax[q].set_xlim(xlim_min, xlim_max)
            ax[q].set_ylim(ylim_min, ylim_max)

            ax[q].axhline(stdc, linestyle=':', color='g', label=r'$\sigma_{c}$')
            if k == 0:
                ax[q].set_xlabel(r'$|RMSE-\sigma_{cal}|/\sigma_{y}$')
                ax[q].axvline(ecut, linestyle=':', color='r', label=r'$E_{c}$')
            elif k == 1:
                ax[q].set_xlabel(r'GPR $\sigma$')

            ax[q].set_ylabel(r'$\sigma_{cal}/\sigma_{y}$')

        if k == 0:
            name = 'ground_truth'
        elif k == 1:
            name = 'prediction_time'

        save = 'figures/'+name+'/'
        os.makedirs(save, exist_ok=True)
        fig.savefig(save+name+'.png')
        with open(save+name+'.json', 'w') as handle:
            json.dump(fig_data, handle)

    pl.close('all')


def plot_score(data, stdc, ecut):
    '''
    Plot the error in erros versus the dissimilarity metric.

    inputs:
        data = A list of calibration data dictionaries.
        stdc = Calibrated STDEV point deviating from ideal.
        ecut = Max error in error cutoff from CV.
    '''

    fig_data = []

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

        marker = i['marker']
        color = i['color']
        ax[j].scatter(
                      x[indx],
                      y[indx],
                      marker=marker,
                      color=color,
                      label=str(i['set'])+'_'+str(i['domain'])+'_'+str(i['run'])
                      )
        ax[j].scatter(
                      x[~indx],
                      y[~indx],
                      facecolors='none',
                      marker=marker,
                      edgecolors=color,
                      )

        # For plot data export
        fig_data.append({
                         'set': i['set'],
                         'run': i['run'],
                         'xopen': x[~indx].tolist(),
                         'yopen': y[~indx].tolist(),
                         'xclosed': x[indx].tolist(),
                         'yclosed': y[indx].tolist(),
                         'stdc': stdc,
                         'ecut': ecut
                         })

    ax[0].set_title('Train')
    ax[1].set_title('Test')
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")

    # Make same sale
    xlim_min = min([ax[0].get_xlim()[0], ax[1].get_xlim()[0]])
    xlim_max = max([ax[0].get_xlim()[1], ax[1].get_xlim()[1]])
    ylim_min = min([ax[0].get_ylim()[0], ax[1].get_ylim()[0]])
    ylim_max = max([ax[0].get_ylim()[1], ax[1].get_ylim()[1]])

    for i in range(2):
        ax[i].set_xlim(xlim_min, xlim_max)
        ax[i].set_ylim(ylim_min, ylim_max)
        ax[i].set_ylabel(r'$|RMSE-\sigma_{cal}|/\sigma_{y}$')
        ax[i].set_xlabel(r'GPR $\sigma$')
        ax[i].axhline(ecut, linestyle=':', color='r', label=r'$E_{c}$')

    save = 'figures/error_in_error/'
    os.makedirs(save, exist_ok=True)
    fig.savefig(save+'error_in_error.png')
    pl.close('all')
    with open(save+'error_in_error.json', 'w') as handle:
        json.dump(fig_data, handle)


def plot_confusion(y_true, score, counts, stdcal, stdc, thresh, name):
    '''
    Confusion matrix based on threshold choice.

    inputs:
        data = A list of calibration data dictionaries.
        stdc = Calibrated STDEV point deviating from ideal.
        ecut = Max error in error cutoff from CV.
    '''

    fig, ax = pl.subplots()

    y_pred = []
    for i, j in zip(score, stdcal):
        if stdc is None:
            if (i < thresh):
                y_pred.append(1)
            else:
                y_pred.append(0)
        else:
            if (i < thresh) and (j < stdc):
                y_pred.append(1)
            else:
                y_pred.append(0)

    # Correct for number of cases per quantile
    y_true_mod = []
    y_pred_mod = []
    for i, j, k in zip(y_true, y_pred, counts):
        y_true_mod += [i]*k
        y_pred_mod += [j]*k

    conf = metrics.confusion_matrix(y_true_mod, y_pred_mod)
    disp = metrics.ConfusionMatrixDisplay(conf, display_labels=['OD', 'ID'])
    disp.plot()
    fig_data = conf.tolist()

    disp.figure_.savefig(name+'_confusion.png')
    with open(name+'_confusion'+'.json', 'w') as handle:
        json.dump(fig_data, handle)


def pr_sub(
           y_true,
           recall,
           precision,
           thresholds,
           stdcal,
           scores,
           counts,
           title,
           stdc
           ):
    '''
    Precision recall cruve.
    '''

    baseline = sum(y_true)/len(y_true)
    auc_score = metrics.auc(recall, precision)
    auc_baseline = auc_score-baseline

    num = 2*recall*precision
    den = recall+precision
    f1_scores = np.divide(num, den, out=np.zeros_like(den), where=(den != 0))
    max_f1 = np.max(f1_scores)
    max_f1_thresh = thresholds[np.argmax(f1_scores)]
    max_auc_thresh = thresholds[np.argmax(precision)]

    max_f1_baseline = max_f1-baseline

    label = 'AUC={:.2f}\n'.format(auc_score)
    label += r'$\Delta$AUC='+'{:.2f}\n'.format(auc_baseline)
    label += 'Max F1={:.2f}\n'.format(max_f1)
    label += 'Max F1 Threshold={:.2f}\n'.format(max_f1_thresh)
    label += r'Max $\Delta$ F1={:.2f}'.format(max_f1_baseline)

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

    fig_data = {
                'auc': auc_score,
                'auc_vs_baseline': auc_baseline,
                'baseline': baseline,
                'max_f1': max_f1,
                'max_f1_thresh': max_f1_thresh,
                'max_f1_vs_baseline': max_f1_baseline,
                'recall': recall.tolist(),
                'precision': precision.tolist(),
                'thresholds': thresholds.tolist(),
                }

    save = 'figures/pr/'
    os.makedirs(save, exist_ok=True)

    name = save+title
    if stdc is None:
        name += '_STDc_None'
    else:
        name += '_STDc_on'

    plot_confusion(
                   y_true,
                   scores,
                   counts,
                   stdcal,
                   stdc,
                   max_f1_thresh,
                   name+'_max_F1'
                   )
    plot_confusion(
                   y_true,
                   scores,
                   counts,
                   stdcal,
                   stdc,
                   max_auc_thresh,
                   name+'_max_auc'
                   )

    fig.savefig(name+'.png')
    pl.close('all')
    with open(name+'.json', 'w') as handle:
        json.dump(fig_data, handle)


def precision_recall(y_true, y_pred):
    '''
    Canculate precision recall.

    inputs:
        y_true = The true labels with 1 being positive.
        y_pred = The predicted labels with 1 being positive.

    return:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
    '''

    tp = 0
    fp = 0
    fn = 0

    for i in range(len(y_true)):
        if (y_true[i] == y_pred[i] == 1):
            tp += 1
        elif (y_true[i] != y_pred[i]) and (y_pred[i] == 1):
            fp += 1
        elif (y_true[i] != y_pred[i]) and (y_pred[i] == 0):
            fn += 1

    try:
        precision = tp/(tp+fp)
    except Exception:
        precision = 1

    try:
        recall = tp/(tp+fn)
    except Exception:
        recall = 1

    return precision, recall


def precision_recall_curve(
                           y_true,
                           scores,
                           stdcal,
                           stdc=None,
                           ):
    '''
    Build precision recall curve.

    inputs:
        y_true = The true labels with 1 being positive.
        scores = The scores the varying threshold.
        stdcal = The calibrated uncertainties.
        errs = The error in error.
        stdc = Calibrated STDEV point deviating from ideal.

    outputs:
        precision = List of TP/(TP+FP)
        recall = List of TP/(TP+FN)
        thresholds = List of scores for the precision recall result.
    '''

    thres = sorted(set(scores))
    indx = list(range(len(scores)))

    n = len(stdcal)

    def calc(i):
        y_pred = np.zeros(n)

        # Need to modif yfor the out of doamin also, now only worsk for ID
        for j in indx:
            if (stdc is not None):
                if (scores[j] > i) and (stdcal[j] < stdc):
                    y_pred[j] = 1
                else:
                    y_pred[j] = 0
            else:
                if scores[j] > i:
                    y_pred[j] = 1
                else:
                    y_pred[j] = 0

        pr, rec = precision_recall(y_true, y_pred)

        return pr, rec

    data = parallel(calc, thres)
    precision = [i[0] for i in data]
    recall = [i[1] for i in data]

    precision = np.array(precision)
    recall = np.array(recall)
    thresholds = np.array(thres)

    return precision, recall, thresholds


def plot_pr(data, stdc):
    '''
    Plot precision recall curve.

    inputs:
        data = A list of calibration data dictionaries.
        stdc = Calibrated STDEV point deviating from ideal.

    outputs:
        fig = The figure object.
        ax = The axis object.
    '''

    # Hold all data needed for PR
    pr = {
          'te_id_domain': [],
          'te_od_score': [],
          'te_stdcal': [],
          'te_errs': [],
          'te_counts': [],
          }

    for i in data:
        if i['set'] == 'te':
            pr['te_id_domain'] += i['in_domain']
            pr['te_od_score'] += i['score']
            pr['te_stdcal'] += i['stdcal']
            pr['te_errs'] += i['errs']
            pr['te_counts'] += i['count']

        else:
            continue

    # Score for ID score (just need small values to be large and vice versa)
    pr['te_id_score'] = [np.exp(-i) for i in pr['te_od_score']]

    # Compute precision recall
    te_id_pr = precision_recall_curve(
                                      pr['te_id_domain'],
                                      pr['te_id_score'],
                                      pr['te_stdcal'],
                                      stdc,
                                      )

    # Convert thresholds back to GPR STDEV
    te_id_pr = list(te_id_pr)
    te_id_pr[2] = np.log(1/te_id_pr[2])
    pr['te_id_score'] = np.log(1/np.array(pr['te_id_score']))

    pr_sub(
           pr['te_id_domain'],
           te_id_pr[1],
           te_id_pr[0],
           te_id_pr[2],
           pr['te_stdcal'],
           pr['te_id_score'],
           pr['te_counts'],
           'Test ID as +',
           stdc,
           )


def main():
    '''
    The general workflow.
    '''

    bins = 15
    root = '.'
    control = 'quantiles'
    dist = 'gpr_std'  # The dissimilarity metric to use
    perc_stdc = 70
    perc_ecut = 95
    name = 'model_errors_leaveout_calibrated.csv'

    df = pd.concat(parallel(load, find(root, name)))
    df = df.sort_values(by=['stdcal', 'y_pred', dist])

    # Assign colors and markers based on groups
    df['color'] = 'r'
    df['marker'] = '.'
    for group, c in zip(df.groupby(['run', 'domain']), cc):
        group, values = group
        indx = (df['run'] == group[0]) & (df['domain'] == group[1])
        df['color'].loc[indx] = c['color']
        df['marker'].loc[indx] = c['marker']

    # Get values from training CV
    if 'random' in df['run'].values:
        df_tr = df[df['in_domain'] == 'tr']
        std_y = df_tr['y'].std()
        df.loc[df['run'] == 'random', 'domain'] = 'random'
    else:
        raise 'Need RepeatedCV run for ID'

    df['score'] = df[dist]

    # Grab quantile data for each set of runs
    group = df.groupby([
                        'run',
                        'in_domain',
                        'domain',
                        'color',
                        'marker'
                        ], sort=False)
    data = parallel(quantiles, group, bins=bins, std_y=std_y, control=control)

    # Grab ground truth from validation sets and assign class
    ecut, stdc = ground_truth(data, perc_stdc, perc_ecut)
    domain_pred(data, ecut, stdc)

    # Make plots
    plot_qq(df)
    plot_calibration(data, stdc, ecut)
    plot_score(data, stdc, ecut)
    plot_gt_pred(data, stdc, ecut)
    plot_pr(data, stdc=None)
    plot_pr(data, stdc)


if __name__ == '__main__':
    main()
