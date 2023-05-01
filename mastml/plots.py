"""
This module contains classes used for generating different types of analysis plots

Scatter:
    This class contains a variety of scatter plot types, e.g. parity (predicted vs. true) plots

Error:
    This class contains plotting methods used to better quantify the model errors and uncertainty quantification.

Histogram:
    This class contains methods for constructing histograms of data distributions and visualization of model residuals.

Line:
    This class contains methods for making line plots, e.g. for constructing learning curves of model performance vs.
    amount of data or number of features.

"""

import warnings
import math
import os
import pandas as pd
import numpy as np
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from math import log, ceil
import scipy
from scipy.stats import gaussian_kde, norm
import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.exceptions import NotFittedError

from mastml.metrics import Metrics
from mastml.error_analysis import ErrorUtils

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure, figaspect
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import statsmodels.api as sm
except:
    print('statsmodels is an optional dependency. If you want to create QQ plots for error analysis, do pip install statsmodels')

matplotlib.rc('font', size=18, family='sans-serif')  # set all font to bigger
matplotlib.rc('figure', autolayout=True)  # turn on autolayout

warnings.filterwarnings(action="ignore")

class Scatter():
    """
    Class to generate scatter plots, such as parity plots showing true vs. predicted data values

    Args:
        None

    Methods:

        plot_predicted_vs_true: method to plot a parity plot
            Args:
                y_true: (pd.Series), series of true y data

                y_pred: (pd.Series), series of predicted y data

                savepath: (str), string denoting the save path for the figure image

                data_type: (str), string denoting the data type (e.g. train, test, leaveout)

                x_label: (str), string denoting the true and predicted property name

                metrics_list: (list), list of strings of metric names to evaluate and include on the figure

                show_figure: (bool), whether or not to show the figure output (e.g. when using Jupyter notebook)

            Returns:
                None

        plot_best_worst_split: method to find the best and worst split in an evaluation set and plot them together
            Args:
                savepath: (str), string denoting the save path for the figure image

                data_type: (str), string denoting the data type (e.g. train, test, leaveout)

                x_label: (str), string denoting the true and predicted property name

                metrics_list: (list), list of strings of metric names to evaluate and include on the figure

                show_figure: (bool), whether or not to show the figure output (e.g. when using Jupyter notebook)

            Returns:
                None

        plot_best_worst_per_point: method to find all of the best and worst data points from an evaluation set and plot them together
            Args:
                savepath: (str), string denoting the save path for the figure image

                data_type: (str), string denoting the data type (e.g. train, test, leaveout)

                x_label: (str), string denoting the true and predicted property name

                metrics_list: (list), list of strings of metric names to evaluate and include on the figure

                show_figure: (bool), whether or not to show the figure output (e.g. when using Jupyter notebook)

            Returns:
                None

        plot_predicted_vs_true_bars: method to plot the average predicted value of each data point from an evaluation set with error bars denoting the standard deviation in predicted values
            Args:
                savepath: (str), string denoting the save path for the figure image

                data_type: (str), string denoting the data type (e.g. train, test, leaveout)

                x_label: (str), string denoting the true and predicted property name

                metrics_list: (list), list of strings of metric names to evaluate and include on the figure

                show_figure: (bool), whether or not to show the figure output (e.g. when using Jupyter notebook)

            Returns:
                None

        plot_metric_vs_group: method to plot the metric value for each group during e.g. a LeaveOneGroupOut data split
            Args:
                savepath: (str), string denoting the save path for the figure image

                data_type: (str), string denoting the data type (e.g. train, test, leaveout)

                show_figure: (bool), whether or not to show the figure output (e.g. when using Jupyter notebook)

            Returns:
                 None

    """
    @classmethod
    def plot_predicted_vs_true(cls, y_true, y_pred, savepath, data_type, x_label, metrics_list=None, show_figure=False,
                               ebars=None, file_extension='.csv', image_dpi=250, groups=None):

        # Make the dataframe/array 1D if it isn't
        y_true = check_dimensions(y_true)
        y_pred = check_dimensions(y_pred)

        # Set image aspect ratio:
        fig, ax = make_fig_ax()

        # gather max and min
        maxx = max(np.nanmax(y_true), np.nanmax(y_pred))
        minn = min(np.nanmin(y_true), np.nanmin(y_pred))

        _set_tick_labels(ax, maxx, minn)

        if groups is not None:
            groups_unique = np.unique(groups)
            colors = ['blue', 'green', 'red', 'purple', 'orange', 'black', 'grey']
            shapes = ['o', 's', '^', 'x', '<', '>', 'h']
            count = 0
            lap = 0
            for group in groups_unique:
                ax.scatter(np.array(y_true)[np.where(groups==group)], np.array(y_pred)[np.where(groups==group)], c=colors[count],
                           marker=shapes[lap], zorder=2, s=100, alpha=0.7, label=group)
                if count < len(colors)-1:
                    count += 1
                else:
                    lap += 1
                    count = 0
            ax.legend(loc='best', fontsize=8)

        else:
            ax.scatter(y_true, y_pred, c='b', edgecolor='darkblue', zorder=2, s=100, alpha=0.7)

        if ebars is not None:
            if groups is not None:
                groups_unique = np.unique(groups)
                colors = ['blue', 'green', 'red', 'purple', 'orange', 'black', 'grey']
                shapes = ['o', 's', '^', 'x', '<', '>', 'h']
                count = 0
                lap = 0
                for group in groups_unique:
                    ax.errorbar(np.array(y_true)[np.where(groups == group)], np.array(y_pred)[np.where(groups == group)],
                                yerr=np.array(ebars)[np.where(groups == group)], fmt=shapes[lap], markerfacecolor=colors[count],
                                markeredgecolor=colors[count], ecolor=colors[count], markersize=10, alpha=0.7, capsize=3)
                    if count < len(colors) - 1:
                        count += 1
                    else:
                        lap += 1
                        count = 0
                ax.legend(loc='best', fontsize=8)
            else:
                ax.errorbar(y_true, y_pred, yerr=ebars, fmt='o',
                        markerfacecolor='blue', markeredgecolor='black', ecolor='blue',
                        markersize=10, alpha=0.7, capsize=3)

        # draw dashed horizontal line
        ax.plot([minn, maxx], [minn, maxx], 'k--', lw=2, zorder=1)

        ax.set_xlabel('True ' + x_label, fontsize=14)
        ax.set_ylabel('Predicted ' + x_label, fontsize=14)

        if metrics_list is None:
            # Use some default metric set
            metrics_list = ['r2_score', 'mean_absolute_error', 'root_mean_squared_error', 'rmse_over_stdev']
        stats_dict = Metrics(metrics_list=metrics_list).evaluate(y_true=y_true, y_pred=y_pred)

        if groups is not None:
            stats_dict_group = dict()
            for group in groups_unique:
                stats_dict_group[group] = Metrics(metrics_list=metrics_list).evaluate(y_true=np.array(y_true)[np.where(groups==group)],
                                                                                      y_pred=np.array(y_pred)[np.where(groups==group)])
            stats_dict_group['Overall'] = Metrics(metrics_list=metrics_list).evaluate(y_true=y_true, y_pred=y_pred)
            stats_group_df = pd.DataFrame().from_dict(stats_dict_group, orient='index', columns=metrics_list)
            if file_extension == '.xlsx':
                stats_group_df.to_excel(os.path.join(savepath, str(data_type)+ '_stats_pergroup_summary' +'.xlsx'))
            elif file_extension == '.csv':
                stats_group_df.to_csv(os.path.join(savepath, str(data_type)+'_stats_pergroup_summary' +'.csv'))

            cls.plot_metric_vs_group(groups_unique, stats_group_df, metrics_list, savepath, data_type, show_figure, file_extension, image_dpi)

        plot_stats(fig, stats_dict, x_align=0.65, y_align=0.90, fontsize=12)
        if ebars is not None:
            if groups is not None:
                fig.savefig(os.path.join(savepath, 'parity_plot_withcalibratederrorbars_grouplabels_' + str(data_type) + '.png'), dpi=image_dpi, bbox_inches='tight')
            else:
                fig.savefig(os.path.join(savepath, 'parity_plot_withcalibratederrorbars_' + str(data_type) + '.png'), dpi=image_dpi, bbox_inches='tight')
            df = pd.DataFrame({'y true': y_true,
                               'y pred': y_pred,
                               'y err': ebars})
            if file_extension == '.xlsx':
                df.to_excel(os.path.join(savepath, 'parity_plot_withcalibratederrorbars_' + str(data_type) + '.xlsx'), index=False)
            elif file_extension == '.csv':
                df.to_csv(os.path.join(savepath, 'parity_plot_withcalibratederrorbars_' + str(data_type) + '.csv'), index=False)
        else:
            if groups is not None:
                fig.savefig(os.path.join(savepath, 'parity_plot_grouplabels_' + str(data_type) + '.png'), dpi=image_dpi, bbox_inches='tight')
            else:
                fig.savefig(os.path.join(savepath, 'parity_plot_'+str(data_type) + '.png'), dpi=image_dpi, bbox_inches='tight')
            df = pd.DataFrame({'y true': y_true,
                               'y pred': y_pred})
            if file_extension == '.xlsx':
                df.to_excel(os.path.join(savepath, 'parity_plot_' + str(data_type) + '.xlsx'), index=False)
            elif file_extension == '.csv':
                df.to_csv(os.path.join(savepath, 'parity_plot_' + str(data_type) + '.csv'), index=False)

        if show_figure == True:
            plt.show()
        else:
            plt.close()
        return

    @classmethod
    def plot_best_worst_split(cls, savepath, data_type, x_label, metrics_list, show_figure=False, file_extension='.csv', image_dpi=250):

        dirs = os.listdir(savepath)
        splitdirs = [d for d in dirs if 'split_' in d and '.png' not in d]

        stats_files_dict = dict()
        for splitdir in splitdirs:
            if file_extension == '.xlsx':
                stats_files_dict[splitdir] = pd.read_excel(os.path.join(os.path.join(savepath, splitdir), data_type + '_stats_summary.xlsx'), engine='openpyxl').to_dict('records')[0]
            elif file_extension == '.csv':
                stats_files_dict[splitdir] = \
                pd.read_csv(os.path.join(os.path.join(savepath, splitdir), data_type + '_stats_summary.csv')).to_dict('records')[0]

        # Find best/worst splits based on RMSE value
        rmse_best = 10**20
        rmse_worst = 0
        for split, stats_dict in stats_files_dict.items():
            if stats_dict['root_mean_squared_error'] < rmse_best:
                best_split = split
                rmse_best = stats_dict['root_mean_squared_error']
            if stats_dict['root_mean_squared_error'] > rmse_worst:
                worst_split = split
                rmse_worst = stats_dict['root_mean_squared_error']
        if file_extension == '.xlsx':
            if data_type == 'test':
                y_true_best = pd.read_excel(os.path.join(os.path.join(savepath, best_split), 'y_test.xlsx'), engine='openpyxl')
                y_pred_best = pd.read_excel(os.path.join(os.path.join(savepath, best_split), 'y_pred.xlsx'), engine='openpyxl')
                y_true_worst = pd.read_excel(os.path.join(os.path.join(savepath, worst_split), 'y_test.xlsx'), engine='openpyxl')
                y_pred_worst = pd.read_excel(os.path.join(os.path.join(savepath, worst_split), 'y_pred.xlsx'), engine='openpyxl')
            elif data_type == 'train':
                y_true_best = pd.read_excel(os.path.join(os.path.join(savepath, best_split), 'y_train.xlsx'), engine='openpyxl')
                y_pred_best = pd.read_excel(os.path.join(os.path.join(savepath, best_split), 'y_pred_train.xlsx'), engine='openpyxl')
                y_true_worst = pd.read_excel(os.path.join(os.path.join(savepath, worst_split), 'y_train.xlsx'), engine='openpyxl')
                y_pred_worst = pd.read_excel(os.path.join(os.path.join(savepath, worst_split), 'y_pred_train.xlsx'), engine='openpyxl')
        elif file_extension == '.csv':
            if data_type == 'test':
                y_true_best = pd.read_csv(os.path.join(os.path.join(savepath, best_split), 'y_test.csv'))
                y_pred_best = pd.read_csv(os.path.join(os.path.join(savepath, best_split), 'y_pred.csv'))
                y_true_worst = pd.read_csv(os.path.join(os.path.join(savepath, worst_split), 'y_test.csv'))
                y_pred_worst = pd.read_csv(os.path.join(os.path.join(savepath, worst_split), 'y_pred.csv'))
            elif data_type == 'train':
                y_true_best = pd.read_csv(os.path.join(os.path.join(savepath, best_split), 'y_train.csv'))
                y_pred_best = pd.read_csv(os.path.join(os.path.join(savepath, best_split), 'y_pred_train.csv'))
                y_true_worst = pd.read_csv(os.path.join(os.path.join(savepath, worst_split), 'y_train.csv'))
                y_pred_worst = pd.read_csv(os.path.join(os.path.join(savepath, worst_split), 'y_pred_train.csv'))


        # Make the dataframe/array 1D if it isn't
        y_true_best = check_dimensions(y_true_best)
        y_pred_best = check_dimensions(y_pred_best)
        y_true_worst = check_dimensions(y_true_worst)
        y_pred_worst = check_dimensions(y_pred_worst)

        # Set image aspect ratio:
        fig, ax = make_fig_ax()

        # gather max and min
        maxx = max(np.nanmax(y_true_best), np.nanmax(y_pred_best), np.nanmax(y_true_worst), np.nanmax(y_pred_worst))
        minn = min(np.nanmin(y_true_best), np.nanmin(y_pred_best), np.nanmin(y_true_worst), np.nanmin(y_pred_worst))

        _set_tick_labels(ax, maxx, minn)

        ax.scatter(y_true_best, y_pred_best, c='b', edgecolor='darkblue', zorder=2, s=100, alpha=0.7, label='Best split')
        ax.scatter(y_true_worst, y_pred_worst, c='r', edgecolor='darkred', zorder=2, s=100, alpha=0.7, label='Worst split')

        ax.legend(loc='best')

        # draw dashed horizontal line
        ax.plot([minn, maxx], [minn, maxx], 'k--', lw=2, zorder=1)

        ax.set_xlabel('True ' + x_label, fontsize=14)
        ax.set_ylabel('Predicted ' + x_label, fontsize=14)

        stats_dict_best = Metrics(metrics_list=metrics_list).evaluate(y_true=y_true_best, y_pred=y_pred_best)
        stats_dict_worst = Metrics(metrics_list=metrics_list).evaluate(y_true=y_true_worst, y_pred=y_pred_worst)

        plot_stats(fig, stats_dict_best, x_align=0.65, y_align=0.90, font_dict={'fontsize': 12, 'color': 'blue'})
        plot_stats(fig, stats_dict_worst, x_align=0.65, y_align=0.50, font_dict={'fontsize': 12, 'color': 'red'})

        # Save data to excel file and image
        fig.savefig(os.path.join(savepath, 'parity_plot_best_worst_split_'+str(data_type)+'.png'), dpi=image_dpi, bbox_inches='tight')
        if show_figure == True:
            plt.show()
        else:
            plt.close()
        return

    @classmethod
    def plot_best_worst_per_point(cls, savepath, data_type, x_label, metrics_list, show_figure=False, file_extension='.csv', image_dpi=250):

        # Get lists of all ytrue and ypred for each split
        dirs = os.listdir(savepath)
        splitdirs = [d for d in dirs if 'split_' in d and '.png' not in d]

        y_true_list = list()
        y_pred_list = list()
        index_list = list()
        if file_extension == '.xlsx':
            for splitdir in splitdirs:
                y_true_list.append(pd.read_excel(os.path.join(os.path.join(savepath, splitdir), 'y_'+str(data_type)+'.xlsx'), engine='openpyxl'))
                if data_type == 'test':
                    y_pred_list.append(pd.read_excel(os.path.join(os.path.join(savepath, splitdir), 'y_pred.xlsx'), engine='openpyxl'))
                    index_list.append(pd.read_excel(os.path.join(os.path.join(savepath, splitdir), 'test_inds.xlsx'), engine='openpyxl'))
                elif data_type == 'train':
                    y_pred_list.append(pd.read_excel(os.path.join(os.path.join(savepath, splitdir), 'y_pred_train.xlsx'), engine='openpyxl'))
                    index_list.append(pd.read_excel(os.path.join(os.path.join(savepath, splitdir), 'train_inds.xlsx'), engine='openpyxl'))
        elif file_extension == '.csv':
            for splitdir in splitdirs:
                y_true_list.append(pd.read_csv(os.path.join(os.path.join(savepath, splitdir), 'y_'+str(data_type)+'.csv')))
                if data_type == 'test':
                    y_pred_list.append(pd.read_csv(os.path.join(os.path.join(savepath, splitdir), 'y_pred.csv')))
                    index_list.append(pd.read_csv(os.path.join(os.path.join(savepath, splitdir), 'test_inds.csv')))
                elif data_type == 'train':
                    y_pred_list.append(pd.read_csv(os.path.join(os.path.join(savepath, splitdir), 'y_pred_train.csv')))
                    index_list.append(pd.read_csv(os.path.join(os.path.join(savepath, splitdir), 'train_inds.csv')))

        all_y_true = list()
        all_y_pred = list()
        all_abs_residuals = list()
        all_indices = list()
        for yt, y_pred, indices in zip(y_true_list, y_pred_list, index_list):
            yt = np.array(check_dimensions(yt))
            y_pred = np.array(check_dimensions(y_pred))
            indices = np.array(check_dimensions(indices))
            abs_residuals = abs(yt-y_pred)
            all_y_true.append(yt)
            all_y_pred.append(y_pred)
            all_abs_residuals.append(abs_residuals)
            all_indices.append(indices)
        all_y_true_flat = np.array([item for sublist in all_y_true for item in sublist])
        all_y_pred_flat = np.array([item for sublist in all_y_pred for item in sublist])
        all_indices_flat = np.array([item for sublist in all_indices for item in sublist])
        all_residuals_flat = np.array([item for sublist in all_abs_residuals for item in sublist])

        # Loop over indices
        unique_inds = np.unique(all_indices_flat)
        bests = list()
        worsts = list()
        y_true_unique = list()
        for ind in unique_inds:
            y_true_unique.append(np.mean(all_y_true_flat[np.where(all_indices_flat==ind)[0]]))
            best = min(abs(all_y_pred_flat[np.where(all_indices_flat==ind)] - all_y_true_flat[np.where(all_indices_flat==ind)]))
            worst = max(abs(all_y_pred_flat[np.where(all_indices_flat==ind)] - all_y_true_flat[np.where(all_indices_flat==ind)]))
            bests.append(all_y_pred_flat[np.where(all_residuals_flat == best)])
            worsts.append(all_y_pred_flat[np.where(all_residuals_flat == worst)])

        y_true_unique = np.array(y_true_unique).ravel()
        bests = np.array(bests).ravel()
        worsts = np.array(worsts).ravel()

        stats_dict_best = Metrics(metrics_list=metrics_list).evaluate(y_true=y_true_unique, y_pred=bests)
        stats_dict_worst = Metrics(metrics_list=metrics_list).evaluate(y_true=y_true_unique, y_pred=worsts)

        fig, ax = make_fig_ax(x_align=0.65)

        # gather max and min
        maxx = float(max([max(y_true_unique), max(bests), max(worsts)]))
        minn = float(min([min(y_true_unique), min(bests), min(worsts)]))

        # draw dashed horizontal line
        ax.plot([minn, maxx], [minn, maxx], 'k--', lw=2, zorder=1)

        # set axis labels
        ax.set_xlabel('True '+x_label, fontsize=16)
        ax.set_ylabel('Predicted '+x_label, fontsize=16)

        # set tick labels
        #maxx = round(float(max1), rounder(max1-min1))
        #minn = round(float(min1), rounder(max1-min1))
        _set_tick_labels(ax, maxx, minn)

        ax.scatter(y_true_unique, bests,  c='b',  alpha=0.7, label='best all points', edgecolor='darkblue', zorder=2, s=100)
        ax.scatter(y_true_unique, worsts, c='r', alpha=0.7, label='worst all points', edgecolor='darkred', zorder=2, s=70)
        ax.legend(loc='best', fontsize=12)

        #plot_stats(fig, avg_stats, x_align=x_align, y_align=0.51, fontsize=10)
        plot_stats(fig, stats_dict_best, x_align=0.65, y_align=0.90, font_dict={'fontsize': 10, 'color': 'b'})
        plot_stats(fig, stats_dict_worst, x_align=0.65, y_align=0.50, font_dict={'fontsize': 10, 'color': 'r'})

        # Save data to excel file and image
        fig.savefig(os.path.join(savepath, 'parity_plot_best_worst_eachpoint_'+str(data_type)+'.png'), dpi=image_dpi, bbox_inches='tight')
        if show_figure == True:
            plt.show()
        else:
            plt.close()
        return

    @classmethod
    def plot_predicted_vs_true_bars(cls, savepath, x_label, data_type, metrics_list, show_figure=False, ebars=None,
                                    file_extension='.csv', image_dpi=250, groups=None):

        # Get lists of all ytrue and ypred for each split
        dirs = os.listdir(savepath)
        splitdirs = [d for d in dirs if 'split_' in d and '.png' not in d]

        y_true_list = list()
        y_pred_list = list()
        data_ind_list = list()
        groups_list = list()
        if file_extension == '.xlsx':
            for splitdir in splitdirs:
                y_true_list.append(pd.read_excel(os.path.join(os.path.join(savepath, splitdir), 'y_'+str(data_type)+'.xlsx'), engine='openpyxl'))
                if data_type == 'test':
                    y_pred_list.append(pd.read_excel(os.path.join(os.path.join(savepath, splitdir), 'y_pred.xlsx'), engine='openpyxl'))
                    data_ind_list.append(pd.read_excel(os.path.join(os.path.join(savepath, splitdir), 'test_inds.xlsx'), engine='openpyxl'))
                    if groups is not None:
                        groups_list.append(pd.read_excel(os.path.join(os.path.join(savepath, splitdir), 'test_groups.xlsx'), engine='openpyxl'))
                elif data_type == 'train':
                    y_pred_list.append(pd.read_excel(os.path.join(os.path.join(savepath, splitdir), 'y_pred_train.xlsx'), engine='openpyxl'))
                    data_ind_list.append(pd.read_excel(os.path.join(os.path.join(savepath, splitdir), 'train_inds.xlsx'), engine='openpyxl'))
                    if groups is not None:
                        groups_list.append(pd.read_excel(os.path.join(os.path.join(savepath, splitdir), 'train_groups.xlsx'), engine='openpyxl'))
                elif data_type == 'leaveout':
                    y_pred_list.append(pd.read_excel(os.path.join(os.path.join(savepath, splitdir), 'y_pred_leaveout.xlsx'), engine='openpyxl'))
                    data_ind_list.append(pd.read_excel(os.path.join(os.path.join(savepath, splitdir), 'leaveout_inds.xlsx'), engine='openpyxl'))
                    if groups is not None:
                        groups_list.append(pd.read_excel(os.path.join(os.path.join(savepath, splitdir), 'leaveout_groups.xlsx'), engine='openpyxl'))
        elif file_extension == '.csv':
            for splitdir in splitdirs:
                y_true_list.append(pd.read_csv(os.path.join(os.path.join(savepath, splitdir), 'y_' + str(data_type) + '.csv')))
                if data_type == 'test':
                    y_pred_list.append(pd.read_csv(os.path.join(os.path.join(savepath, splitdir), 'y_pred.csv')))
                    data_ind_list.append(pd.read_csv(os.path.join(os.path.join(savepath, splitdir), 'test_inds.csv')))
                    if groups is not None:
                        groups_list.append(pd.read_csv(os.path.join(os.path.join(savepath, splitdir), 'test_groups.csv')))
                elif data_type == 'train':
                    y_pred_list.append(pd.read_csv(os.path.join(os.path.join(savepath, splitdir), 'y_pred_train.csv')))
                    data_ind_list.append(pd.read_csv(os.path.join(os.path.join(savepath, splitdir), 'train_inds.csv')))
                    if groups is not None:
                        groups_list.append(pd.read_csv(os.path.join(os.path.join(savepath, splitdir), 'train_groups.csv')))
                elif data_type == 'leaveout':
                    y_pred_list.append(pd.read_csv(os.path.join(os.path.join(savepath, splitdir), 'y_pred_leaveout.csv')))
                    data_ind_list.append(pd.read_csv(os.path.join(os.path.join(savepath, splitdir), 'leaveout_inds.csv')))
                    if groups is not None:
                        groups_list.append(pd.read_csv(os.path.join(os.path.join(savepath, splitdir), 'leaveout_groups.csv')))

        all_y_true = list()
        all_y_pred = list()
        all_data_inds = list()
        all_groups = list()
        if groups is not None:
            for groups in groups_list:
                all_groups.append(np.array(groups))
        for yt, y_pred, data_inds in zip(y_true_list, y_pred_list, data_ind_list):
            yt = np.array(check_dimensions(yt))
            y_pred = np.array(check_dimensions(y_pred))
            data_inds = np.array(check_dimensions(data_inds))
            all_y_true.append(yt)
            all_y_pred.append(y_pred)
            all_data_inds.append(data_inds)

        df_all = pd.DataFrame({'all_y_true': np.array([item for sublist in all_y_true for item in sublist]),
                               'all_y_pred': np.array([item for sublist in all_y_pred for item in sublist]),
                               'all_data_inds': np.array([item for sublist in all_data_inds for item in sublist])})
        if groups is not None:
            df_all['all_groups']= np.array([item for sublist in all_groups for item in sublist])

        if ebars is not None:
            df_all['ebars'] = np.array(ebars)

        df_all_grouped = df_all.groupby('all_data_inds', as_index=False, sort=False)
        df_avg = df_all_grouped.mean()
        df_std = df_all_grouped.std()

        # make fig and ax, use x_align when placing text so things don't overlap
        x_align = 0.64
        fig, ax = make_fig_ax(x_align=x_align)

        trues = df_avg['all_y_true']
        preds = df_avg['all_y_pred']
        if groups is not None:
            # Need to use the indices of df.groupby to get the original list of groups
            grps = df_all_grouped.groups
            inds = list()
            for k, v, in grps.items():
                inds.append(v[0])
            groups = np.array(df_all['all_groups'])[inds]

        # gather max and min
        maxx = max(np.nanmax(trues), np.nanmax(preds))
        minn = min(np.nanmin(trues), np.nanmin(preds))

        # draw dashed horizontal line
        ax.plot([minn, maxx], [minn, maxx], 'k--', lw=2, zorder=1)

        # set axis labels
        ax.set_xlabel('True ' + x_label, fontsize=16)
        ax.set_ylabel('Predicted ' + x_label, fontsize=16)

        # set tick labels
        _set_tick_labels(ax, maxx, minn)

        if groups is not None:
            groups_unique = np.unique(groups)
            colors = ['blue', 'green', 'red', 'purple', 'orange', 'black', 'grey']
            shapes = ['o', 's', '^', 'x', '<', '>', 'h']
            count = 0
            lap = 0
            for group in groups_unique:
                ax.scatter(np.array(trues)[np.where(groups==group)], np.array(preds)[np.where(groups==group)], c=colors[count],
                           marker=shapes[lap], zorder=2, s=100, alpha=0.7, label=group)
                if count < len(colors)-1:
                    count += 1
                else:
                    lap += 1
                    count = 0
            ax.legend(loc='best', fontsize=8)
        else:
            ax.scatter(trues, preds, c='b', edgecolor='darkblue', zorder=2, s=100, alpha=0.7)

        if ebars is not None:
            if groups is not None:
                groups_unique = np.unique(groups)
                colors = ['blue', 'green', 'red', 'purple', 'orange', 'black', 'grey']
                shapes = ['o', 's', '^', 'x', '<', '>', 'h']
                count = 0
                lap = 0
                for group in groups_unique:
                    ax.errorbar(np.array(trues)[np.where(groups == group)], np.array(preds)[np.where(groups == group)],
                                yerr=np.array(df_avg['ebars'])[np.where(groups == group)], fmt=shapes[lap], markerfacecolor=colors[count],
                                markeredgecolor=colors[count], ecolor=colors[count], markersize=10, alpha=0.7, capsize=3)
                    if count < len(colors) - 1:
                        count += 1
                    else:
                        lap += 1
                        count = 0
                ax.legend(loc='best', fontsize=8)
            else:
                ax.errorbar(trues, preds, yerr=df_avg['ebars'], fmt='o',
                        markerfacecolor='blue', markeredgecolor='black', ecolor='blue',
                        markersize=10, alpha=0.7, capsize=3)
        else:
            if groups is not None:
                groups_unique = np.unique(groups)
                colors = ['blue', 'green', 'red', 'purple', 'orange', 'black', 'grey']
                shapes = ['o', 's', '^', 'x', '<', '>', 'h']
                count = 0
                lap = 0
                for group in groups_unique:
                    ax.errorbar(np.array(trues)[np.where(groups == group)], np.array(preds)[np.where(groups == group)],
                                yerr=np.array(df_std['all_y_pred'])[np.where(groups == group)], fmt=shapes[lap], markerfacecolor=colors[count],
                                markeredgecolor=colors[count], ecolor=colors[count], markersize=10, alpha=0.7, capsize=3)
                    if count < len(colors) - 1:
                        count += 1
                    else:
                        lap += 1
                        count = 0
                ax.legend(loc='best', fontsize=8)
            else:
                ax.errorbar(trues, preds, yerr=df_std['all_y_pred'], fmt='o',
                        markerfacecolor='blue', markeredgecolor='black', ecolor='blue',
                        markersize=10, alpha=0.7, capsize=3)

        stats_files_dict = dict()
        for splitdir in splitdirs:
            if file_extension == '.xlsx':
                stats_files_dict[splitdir] = pd.read_excel(os.path.join(os.path.join(savepath, splitdir), data_type + '_stats_summary.xlsx'), engine='openpyxl').to_dict('records')[0]
            elif file_extension == '.csv':
                stats_files_dict[splitdir] = pd.read_csv(os.path.join(os.path.join(savepath, splitdir), data_type + '_stats_summary.csv')).to_dict('records')[0]
            metrics_list = list(stats_files_dict[splitdir].keys())

        avg_stats = dict()
        for metric in metrics_list:
            stats = list()
            for splitdir in splitdirs:
                stats.append(stats_files_dict[splitdir][metric])

            avg_stats[metric] = (np.mean(stats), np.std(stats))

        plot_stats(fig, avg_stats, x_align=x_align, y_align=0.90)

        if ebars is not None:
            if groups is not None:
                fig.savefig(os.path.join(savepath, 'parity_plot_allsplits_average_withcalibratederrorbars_grouplabels_' + str(data_type) + '.png'), dpi=image_dpi, bbox_inches='tight')
            else:
                fig.savefig(os.path.join(savepath, 'parity_plot_allsplits_average_withcalibratederrorbars_' + str(data_type) + '.png'), dpi=image_dpi, bbox_inches='tight')

            df = pd.DataFrame({'y true': trues,
                               'average predicted values': preds,
                               'error bar values': df_avg['ebars']})
            if file_extension == '.xlsx':
                df.to_excel(os.path.join(savepath, 'parity_plot_allsplits_average_withcalibratederrorbars_' + str(data_type) + '.xlsx'), index=False)
            elif file_extension == '.csv':
                df.to_csv(os.path.join(savepath, 'parity_plot_allsplits_average_withcalibratederrorbars_' + str(data_type) + '.csv'), index=False)
        else:
            if groups is not None:
                fig.savefig(os.path.join(savepath, 'parity_plot_allsplits_average_grouplabels_' + str(data_type) + '.png'), dpi=image_dpi, bbox_inches='tight')
            else:
                fig.savefig(os.path.join(savepath, 'parity_plot_allsplits_average_'+str(data_type)+'.png'), dpi=image_dpi, bbox_inches='tight')

            df = pd.DataFrame({'y true': trues,
                               'average predicted values': preds,
                               'error bar values': df_std['all_y_pred']})
            if file_extension == '.xlsx':
                df.to_excel(os.path.join(savepath, 'parity_plot_allsplits_average_'+str(data_type)+'.xlsx'), index=False)
            elif file_extension == '.csv':
                df.to_csv(os.path.join(savepath, 'parity_plot_allsplits_average_' + str(data_type) + '.csv'), index=False)

        df_stats = pd.DataFrame().from_dict(avg_stats)
        if file_extension == '.xlsx':
            df_stats.to_excel(os.path.join(savepath, str(data_type)+'_average_stdev_stats_summary.xlsx'), index=False)
        elif file_extension == '.csv':
            df_stats.to_csv(os.path.join(savepath, str(data_type) + '_average_stdev_stats_summary.csv'), index=False)

        if show_figure == True:
            plt.show()
        else:
            plt.close()
        return

    @classmethod
    def plot_metric_vs_group(cls, groups, stats_group_df, metrics_list, savepath, data_type, show_figure, file_extension='.csv', image_dpi=250):

        for metric in metrics_list:
            stats = list()
            for group in groups:
                stats.append(stats_group_df[metric][group])

            avg_stats = {metric: (np.mean(stats), np.std(stats))}

            # make fig and ax, use x_align when placing text so things don't overlap
            x_align = 0.64
            fig, ax = make_fig_ax(x_align=x_align)

            # do the actual plotting
            ax.scatter(groups, stats, c='blue', alpha=0.7, edgecolor='darkblue', zorder=2, s=100)

            # set axis labels
            ax.set_xlabel('Group', fontsize=14)
            ax.set_ylabel(metric, fontsize=14)
            ax.set_xticklabels(labels=groups, fontsize=14)
            plot_stats(fig, avg_stats, x_align=x_align, y_align=0.90)

            fig.savefig(os.path.join(savepath, str(metric)+'_value_per_group_'+str(data_type)+'.png'), dpi=image_dpi, bbox_inches='tight')
            if show_figure == True:
                plt.show()
            else:
                plt.close()
        return


class Error():
    """
    Class to make plots related to model error assessment and uncertainty quantification

    Args:
        None

    Methods:
        plot_normalized_error: Method to plot the normalized residual errors of a model prediction
            Args:
                residuals: (pd.Series), series containing the true errors (model residuals)

                savepath: (str), string denoting the save path to save the figure to

                data_type: (str), string denoting the data type, e.g. train, test, leftout

                model_errors: (pd.Series), series containing the predicted model errors (optional, default None)

                show_figure: (bool), whether or not the generated figure is output to the notebook screen (default False)

            Returns:
                None

        plot_cumulative_normalized_error: Method to plot the cumulative normalized residual errors of a model prediction
            Args:
                residuals: (pd.Series), series containing the true errors (model residuals)

                savepath: (str), string denoting the save path to save the figure to

                data_type: (str), string denoting the data type, e.g. train, test, leftout

                model_errors: (pd.Series), series containing the predicted model errors (optional, default None)

                show_figure: (bool), whether or not the generated figure is output to the notebook screen (default False)


            Returns:
                None

        plot_rstat: Method for plotting the r-statistic distribution (true divided by predicted error)
            Args:
                savepath: (str), string denoting the save path to save the figure to

                data_type: (str), string denoting the data type, e.g. train, test, leftout

                residuals: (pd.Series), series containing the true errors (model residuals)

                model_errors: (pd.Series), series containing the predicted model errors

                show_figure: (bool), whether or not the generated figure is output to the notebook screen (default False)

                is_calibrated: (bool), whether or not the model errors have been recalibrated (default False)

            Returns:
                None

        plot_rstat_uncal_cal_overlay: Method for plotting the r-statistic distribution for two cases together: the as-obtained uncalibrated model errors and calibrated errors
            Args:
                savepath: (str), string denoting the save path to save the figure to

                data_type: (str), string denoting the data type, e.g. train, test, leftout

                residuals: (pd.Series), series containing the true errors (model residuals)

                model_errors: (pd.Series), series containing the predicted model errors

                model_errors_cal: (pd.Series), series containing the calibrated predicted model errors

                show_figure: (bool), whether or not the generated figure is output to the notebook screen (default False)

            Returns:
                None

        plot_real_vs_predicted_error: Sometimes called the RvE plot, or residual vs. error plot, this method plots the binned RMS residuals as a function of the binned model errors
            Args:
                savepath: (str), string denoting the save path to save the figure to

                model: (mastml.models object), a MAST-ML model object, e.g. SklearnModel or EnsembleModel

                data_type: (str), string denoting the data type, e.g. train, test, leftout

                model_errors: (pd.Series), series containing the predicted model errors

                residuals: (pd.Series), series containing the true errors (model residuals)

                dataset_stdev: (float), the standard deviation of the training dataset

                show_figure: (bool), whether or not the generated figure is output to the notebook screen (default False)

                is_calibrated: (bool), whether or not the model errors have been recalibrated (default False)

                well_sampled_fraction: (float), number denoting whether a bin qualifies as well-sampled or not. Default to 0.025 (2.5% of total samples). Only affects visuals, not fitting

            Returns:
                None

        plot_real_vs_predicted_error_uncal_cal_overlay: Method for making the residual vs. error plot for two cases together: using the as-obtained uncalibrated model errors and calibrated errors
            Args:
                savepath: (str), string denoting the save path to save the figure to

                model: (mastml.models object), a MAST-ML model object, e.g. SklearnModel or EnsembleModel

                data_type: (str), string denoting the data type, e.g. train, test, leftout

                model_errors: (pd.Series), series containing the predicted model errors

                model_errors_cal: (pd.Series), series containing the calibrated predicted model errors

                residuals: (pd.Series), series containing the true errors (model residuals)

                dataset_stdev: (float), the standard deviation of the training dataset

                show_figure: (bool), whether or not the generated figure is output to the notebook screen (default False)

                well_sampled_fraction: (float), number denoting whether a bin qualifies as well-sampled or not. Default to 0.025 (2.5% of total samples). Only affects visuals, not fitting

            Returns:
                None

    """

    @classmethod
    def plot_qq(cls, residuals, savepath, data_type, show_figure, image_dpi=250):
        x_align = 0.64
        fig, ax = make_fig_ax(x_align=x_align)
        fig = sm.qqplot(data=residuals, dist=scipy.stats.distributions.norm, line='45', fit=True,
                            ax=ax, markerfacecolor='blue', markeredgecolor='darkblue',
                            zorder=2, markersize=10, alpha=0.7)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('Normal distribution quantiles', fontsize=14)
        ax.set_ylabel('Model residual quantiles', fontsize=14)
        ax.get_lines()[1].set_color("black")
        ax.get_lines()[1].set_linewidth("1.5")
        ax.get_lines()[1].set_linestyle("--")
        ax.xaxis.get_label().set_fontsize(12)
        ax.yaxis.get_label().set_fontsize(12)
        fig.savefig(os.path.join(savepath, 'qq_plot_'+str(data_type)+'.png'), dpi=image_dpi, bbox_inches='tight')
        if show_figure is True:
            plt.show()
        else:
            plt.close()
        return

    @classmethod
    def plot_normalized_error(cls, residuals, savepath, data_type, model_errors=None, show_figure=False, file_extension='.csv', image_dpi=250):

        x_align = 0.64
        fig, ax = make_fig_ax(x_align=x_align)
        mu = 0
        sigma = 1
        residuals[residuals == 0.0] = 10**-6
        normalized_residuals = residuals / np.std(residuals)
        density_residuals = gaussian_kde(normalized_residuals)
        x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, residuals.shape[0])
        ax.plot(x, norm.pdf(x, mu, sigma), linewidth=4, color='blue', label="Analytical Gaussian")
        ax.plot(x, density_residuals(x), linewidth=4, color='green', label="Model Residuals")
        maxx = 5
        minn = -5

        if model_errors is not None:
            model_errors[model_errors == 0.0] = 0.0001
            rstat = residuals / model_errors
            density_errors = gaussian_kde(rstat)
            maxy = max(max(density_residuals(x)), max(norm.pdf(x, mu, sigma)), max(density_errors(x)))
            miny = min(min(density_residuals(x)), min(norm.pdf(x, mu, sigma)), max(density_errors(x)))
            ax.plot(x, density_errors(x), linewidth=4, color='purple', label="Model Errors")
            # Save data to csv file
            data_dict = {"Plotted x values": x, "model_errors": model_errors,
                         # "analytical gaussian (plotted y blue values)": norm.pdf(x, mu, sigma),
                         "residuals": residuals,
                         "model normalized residuals (plotted y green values)": density_residuals(x),
                         "model errors (plotted y purple values)": density_errors(x)}
        else:
            # Save data to csv file
            data_dict = {"x values": x,
                         # "analytical gaussian": norm.pdf(x, mu, sigma),
                         "model normalized residuals (plotted y green values)": density_residuals(x)}
            maxy = max(max(density_residuals(x)), max(norm.pdf(x, mu, sigma)))
            miny = min(min(density_residuals(x)), min(norm.pdf(x, mu, sigma)))

        if file_extension == '.xlsx':
            pd.DataFrame(data_dict).to_excel(os.path.join(savepath, 'normalized_error_data_'+str(data_type)+'.xlsx'))
        elif file_extension == '.csv':
            pd.DataFrame(data_dict).to_csv(os.path.join(savepath, 'normalized_error_data_' + str(data_type) + '.csv'))
        ax.legend(loc=0, fontsize=12, frameon=False)
        ax.set_xlabel(r"$\mathrm{x}/\mathit{\sigma}$", fontsize=18)
        ax.set_ylabel("Probability density", fontsize=18)
        _set_tick_labels_different(ax, maxx, minn, maxy, miny)
        fig.savefig(os.path.join(savepath, 'normalized_errors_'+str(data_type)+'.png'), dpi=image_dpi, bbox_inches='tight')
        if show_figure is True:
            plt.show()
        else:
            plt.close()
        return

    @classmethod
    def plot_cumulative_normalized_error(cls, residuals, savepath, data_type, model_errors=None, show_figure=False, file_extension='.csv', image_dpi=250):

        x_align = 0.64
        fig, ax = make_fig_ax(x_align=x_align)

        analytic_gau = np.random.normal(0, 1, 10000)
        analytic_gau = abs(analytic_gau)
        n_analytic = np.arange(1, len(analytic_gau) + 1) / np.float(len(analytic_gau))
        X_analytic = np.sort(analytic_gau)
        residuals[residuals == 0.0] = 10 ** -6
        normalized_residuals = abs((residuals) / np.std(residuals))
        n_residuals = np.arange(1, len(normalized_residuals) + 1) / np.float(len(normalized_residuals))
        X_residuals = np.sort(normalized_residuals)  # r"$\mathrm{Predicted \/ Value}, \mathit{eV}$"
        ax.set_xlabel(r"$\mathrm{x}/\mathit{\sigma}$", fontsize=18)
        ax.set_ylabel("Fraction", fontsize=18)
        ax.step(X_residuals, n_residuals, linewidth=3, color='green', label="Model Residuals")
        ax.step(X_analytic, n_analytic, linewidth=3, color='blue', label="Analytical Gaussian")
        ax.set_xlim([0, 5])

        if model_errors is not None:
            model_errors[model_errors == 0.0] = 0.0001
            rstat = abs((residuals) / model_errors)
            n_errors = np.arange(1, len(rstat) + 1) / np.float(len(rstat))
            X_errors = np.sort(rstat)
            ax.step(X_errors, n_errors, linewidth=3, color='purple', label="Model Errors")
            # Save data to csv file
            data_dict = {  # "Analytical Gaussian values": analytic_gau,
                # "Analytical Gaussian (sorted, blue data)": X_analytic,
                "residuals": residuals,
                "normalized residuals": normalized_residuals,
                "Model Residuals (sorted, green data)": X_residuals,
                "Model error values (r value: (ytrue-ypred)/(model error avg))": rstat,
                "Model errors (sorted, purple values)": X_errors}
        else:
            # Save data to csv file
            data_dict = {  # "x analytical": X_analytic,
                # "analytical gaussian": n_analytic,
                "Model Residuals (sorted, green data)": X_residuals,
                "model residuals": n_residuals}
        # Save this way to avoid issue with different array sizes in data_dict
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_dict.items()]))
        if file_extension == '.xlsx':
            df.to_excel(os.path.join(savepath, 'cumulative_normalized_errors_'+str(data_type)+'.xlsx'), index=False)
        elif file_extension == '.csv':
            df.to_csv(os.path.join(savepath, 'cumulative_normalized_errors_'+str(data_type)+'.csv'), index=False)

        ax.legend(loc=0, fontsize=14, frameon=False)
        xlabels = np.linspace(2, 3, 3)
        ylabels = np.linspace(0.9, 1, 2)
        axin = zoomed_inset_axes(ax, 2.5, loc=7)
        axin.step(X_residuals, n_residuals, linewidth=3, color='green', label="Model Residuals")
        axin.step(X_analytic, n_analytic, linewidth=3, color='blue', label="Analytical Gaussian")
        if model_errors is not None:
            axin.step(X_errors, n_errors, linewidth=3, color='purple', label="Model Errors")
        axin.set_xticklabels(xlabels, fontsize=8, rotation=90)
        axin.set_yticklabels(ylabels, fontsize=8)
        axin.set_xlim([2, 3])
        axin.set_ylim([0.9, 1])

        maxx = 5
        minn = 0
        maxy = 1.1
        miny = 0
        _set_tick_labels_different(ax, maxx, minn, maxy, miny)

        mark_inset(ax, axin, loc1=1, loc2=2)
        fig.savefig(os.path.join(savepath, 'cumulative_normalized_errors_'+str(data_type)+'.png'), dpi=image_dpi, bbox_inches='tight')
        if show_figure is True:
            plt.show()
        else:
            plt.close()
        return

    @classmethod
    def plot_rstat(cls, savepath, data_type, residuals, model_errors, show_figure=False, is_calibrated=False, image_dpi=250):

        # Eliminate model errors with value 0, so that the ratios can be calculated
        zero_indices = []
        for i in range(0, len(model_errors)):
            if model_errors[i] == 0:
                zero_indices.append(i)
        residuals = np.delete(residuals, zero_indices)
        model_errors = np.delete(model_errors, zero_indices)
        # make data for gaussian plot
        gaussian_x = np.linspace(-5, 5, 1000)
        # create plot
        x_align = 0.64
        fig, ax = make_fig_ax(x_align=x_align)
        ax.set_xlabel('residuals / model error estimates')
        ax.set_ylabel('relative counts')
        ax.hist(residuals/model_errors, bins=30, color='blue', edgecolor='black', density=True)
        ax.plot(gaussian_x, stats.norm.pdf(gaussian_x, 0, 1), label='Gaussian mu: 0 std: 1', color='black', linestyle='--', linewidth=1.5)
        ax.text(0.05, 0.9, 'mean = %.3f' % (np.mean(residuals / model_errors)), transform=ax.transAxes)
        ax.text(0.05, 0.85, 'std = %.3f' % (np.std(residuals / model_errors)), transform=ax.transAxes)

        if is_calibrated == False:
            calibrate = 'uncalibrated'
        if is_calibrated == True:
            calibrate = 'calibrated'

        fig.savefig(os.path.join(savepath, 'rstat_histogram_'+str(data_type)+'_'+calibrate+'.png'), dpi=image_dpi, bbox_inches='tight')

        if show_figure is True:
            plt.show()
        else:
            plt.close()
        return

    @classmethod
    def plot_rstat_uncal_cal_overlay(cls, savepath, data_type, residuals, model_errors, model_errors_cal,
                                     show_figure=False, image_dpi=250):

        # Eliminate model errors with value 0, so that the ratios can be calculated
        zero_indices = []
        for i in range(0, len(model_errors)):
            if model_errors[i] == 0:
                zero_indices.append(i)
        residuals = np.delete(residuals, zero_indices)
        model_errors = np.delete(model_errors, zero_indices)
        model_errors_cal = np.delete(model_errors_cal, zero_indices)

        # make data for gaussian plot
        gaussian_x = np.linspace(-5, 5, 1000)
        # create plot
        x_align = 0.64
        fig, ax = make_fig_ax(x_align=x_align)
        ax.set_xlabel('residuals / model error estimates')
        ax.set_ylabel('relative counts')
        ax.hist(residuals/model_errors, bins=30, color='gray', edgecolor='black', density=True, alpha=0.4)
        ax.hist(residuals/model_errors_cal, bins=30, color='blue', edgecolor='black', density=True, alpha=0.4)
        ax.plot(gaussian_x, stats.norm.pdf(gaussian_x, 0, 1), label='Gaussian mu: 0 std: 1', color='black', linestyle='--', linewidth=1.5)
        ax.text(0.05, 0.9, 'mean = %.3f' % (np.mean(residuals / model_errors)), transform=ax.transAxes, fontdict={'fontsize': 10, 'color': 'gray'})
        ax.text(0.05, 0.85, 'std = %.3f' % (np.std(residuals / model_errors)), transform=ax.transAxes, fontdict={'fontsize': 10, 'color': 'gray'})
        ax.text(0.05, 0.8, 'mean = %.3f' % (np.mean(residuals / model_errors_cal)), transform=ax.transAxes, fontdict={'fontsize': 10, 'color': 'blue'})
        ax.text(0.05, 0.75, 'std = %.3f' % (np.std(residuals / model_errors_cal)), transform=ax.transAxes, fontdict={'fontsize': 10, 'color': 'blue'})
        fig.savefig(os.path.join(savepath, 'rstat_histogram_'+str(data_type)+'_uncal_cal_overlay.png'), dpi=image_dpi, bbox_inches='tight')

        if show_figure is True:
            plt.show()
        else:
            plt.close()
        return

    @classmethod
    def plot_real_vs_predicted_error(cls, savepath, model, data_type, model_errors, residuals, dataset_stdev,
                                     show_figure=False, is_calibrated=False, well_sampled_number=30, image_dpi=250):

        bin_values, rms_residual_values, num_values_per_bin, number_of_bins, ms_residual_values, var_sq_residual_values = ErrorUtils()._parse_error_data(model_errors=model_errors,
                                                                                                             residuals=residuals,
                                                                                                             dataset_stdev=dataset_stdev)

        model_name = model.model.__class__.__name__
        if model_name == 'RandomForestRegressor':
            model_type = 'RF'
        elif model_name == 'GradientBoostingRegressor':
            model_type = 'GBR'
        elif model_name == 'ExtraTreesRegressor':
            model_type = 'ET'
        elif model_name == 'GaussianProcessRegressor':
            model_type = 'GPR'
        elif model_name == 'BaggingRegressor':
            model_type = 'BR'
        elif model_name == 'AdaBoostRegressor':
            model_type = 'ABR'

        if data_type not in ['train', 'test', 'leaveout']:
            print('Error: data_test_type must be one of "train", "test" or "leaveout"')
            exit()

        # Make RF error plot
        fig, ax = make_fig_ax(aspect_ratio=0.5, x_align=0.65)

        linear = LinearRegression(fit_intercept=True)
        # Fit just blue circle data
        # Find nan entries
        nans = np.argwhere(np.isnan(rms_residual_values)).tolist()

        # use nans (which are indices) to delete relevant parts of bin_values and
        # rms_residual_values as they can't be used to fit anyway
        bin_values_copy = np.empty_like(bin_values)
        bin_values_copy[:] = bin_values
        rms_residual_values_copy = np.empty_like(rms_residual_values)
        rms_residual_values_copy[:] = rms_residual_values
        bin_values_copy = np.delete(bin_values_copy, nans)
        rms_residual_values_copy = np.delete(rms_residual_values_copy, nans)

        num_values_per_bin_copy = np.array(num_values_per_bin)[np.array(num_values_per_bin) != 0]

        # Only examine the bins that are well-sampled, i.e. have number of data points in them above a given threshold
        #well_sampled_number = round(well_sampled_fraction*np.sum(num_values_per_bin_copy))
        rms_residual_values_wellsampled = rms_residual_values_copy[np.where(num_values_per_bin_copy > well_sampled_number)]
        bin_values_wellsampled = bin_values_copy[np.where(num_values_per_bin_copy > well_sampled_number)]
        num_values_per_bin_wellsampled = num_values_per_bin_copy[np.where(num_values_per_bin_copy > well_sampled_number)]
        rms_residual_values_poorlysampled = rms_residual_values_copy[np.where(num_values_per_bin_copy <= well_sampled_number)]
        bin_values_poorlysampled = bin_values_copy[np.where(num_values_per_bin_copy <= well_sampled_number)]
        num_values_per_bin_poorlysampled = num_values_per_bin_copy[np.where(num_values_per_bin_copy <= well_sampled_number)]

        yerr = list()
        for i, j, k in zip(var_sq_residual_values, num_values_per_bin, rms_residual_values):
            if j > 1:
                yerr.append(np.sqrt(i) / (2 * np.sqrt(j) * k))
            else:
                yerr.append(1)
        yerr = np.array(yerr)

        yerr_wellsampled = yerr[np.where(num_values_per_bin > well_sampled_number)[0]]
        yerr_poorlysampled = yerr[np.where(num_values_per_bin <= well_sampled_number)[0]]

        ax.scatter(bin_values_wellsampled, rms_residual_values_wellsampled, s=80, color='blue', alpha=0.7)
        ax.scatter(bin_values_poorlysampled, rms_residual_values_poorlysampled, s=40, color='blue', alpha=0.3)
        ax.errorbar(bin_values_wellsampled, rms_residual_values_wellsampled, yerr=yerr_wellsampled, ecolor='blue', capsize=2, linewidth=0, elinewidth=1)
        ax.errorbar(bin_values_poorlysampled, rms_residual_values_poorlysampled, yerr=yerr_poorlysampled, ecolor='blue', capsize=2, linewidth=0, elinewidth=1, alpha=0.4)

        ax.set_xlabel(str(model_type) + ' model errors / dataset stdev', fontsize=12)
        ax.set_ylabel('RMS Absolute residuals\n / dataset stdev', fontsize=12)
        ax.tick_params(labelsize=10)

        if not rms_residual_values_copy.size:
            print("---WARNING: ALL ERRORS TOO LARGE FOR PLOTTING---")
            exit()
        else:
            # Fit the line to all data, including the poorly sampled data, and weight data points by number of samples per bin
            linear.fit(np.array(bin_values_copy).reshape(-1, 1), rms_residual_values_copy,
                       sample_weight=num_values_per_bin_copy)
            yfit = linear.predict(np.array(bin_values_copy).reshape(-1, 1))
            ax.plot(bin_values_copy, yfit, 'k--', linewidth=2)
            r2 = r2_score(rms_residual_values_copy, yfit, sample_weight=num_values_per_bin_copy)

            slope = linear.coef_
            intercept = linear.intercept_

        divider = make_axes_locatable(ax)
        axbarx = divider.append_axes("top", 1.2, pad=0.12, sharex=ax)

        axbarx.bar(x=bin_values, height=num_values_per_bin, width=bin_values[1]-bin_values[0], color='blue', edgecolor='black', alpha=0.7)
        axbarx.tick_params(labelsize=10, axis='y')
        axbarx.tick_params(labelsize=0, axis='x')
        axbarx.set_ylabel('Counts', fontsize=12)

        total_samples = sum(num_values_per_bin)

        #xmax = max(max(bin_values_copy) + 0.05, 1.6)
        #ymax = max(1.3, max(rms_residual_values))

        xmax = round(max(bin_values_copy) * 1.10, 2)
        ymax = round(max(rms_residual_values) * 1.10, 2)

        axbarx.text(0.6 * xmax, round(0.9 * max(num_values_per_bin)), 'Total counts = ' + str(total_samples),
                    fontsize=10)

        ax.set_ylim(bottom=0, top=ymax)
        axbarx.set_ylim(bottom=0, top=round(max(num_values_per_bin) + 0.1*max(num_values_per_bin)))
        ax.set_xlim(left=0, right=xmax)

        ax.text(0.02, 0.9*ymax, 'R$^2$ = %3.2f ' % r2, fontdict={'fontsize': 10, 'color': 'k'})
        ax.text(0.02, 0.8*ymax, 'slope = %3.2f ' % slope, fontdict={'fontsize': 10, 'color': 'k'})
        ax.text(0.02, 0.7*ymax, 'intercept = %3.2f ' % intercept, fontdict={'fontsize': 10, 'color': 'k'})

        # Plot y = x line as reference point
        maxx = max(xmax, ymax)
        ax.plot([0, maxx], [0, maxx], 'k--', lw=2, zorder=1, color='gray', alpha=0.5)

        if is_calibrated == False:
            calibrate = 'uncalibrated'
        if is_calibrated == True:
            calibrate = 'calibrated'

        fig.savefig(os.path.join(savepath, str(model_type) + '_residuals_vs_modelerror_' + str(data_type) + '_' + calibrate + '.png'),
                    dpi=image_dpi, bbox_inches='tight')

        if show_figure is True:
            plt.show()
        else:
            plt.close()

        return

    @classmethod
    def plot_real_vs_predicted_error_uncal_cal_overlay(cls, savepath, model, data_type, model_errors, model_errors_cal,
                                                       residuals, dataset_stdev, show_figure=False,
                                                       well_sampled_number=30, image_dpi=250):

        bin_values_uncal, rms_residual_values_uncal, num_values_per_bin_uncal, number_of_bins_uncal, ms_residual_values_uncal, var_sq_residual_values_uncal = ErrorUtils()._parse_error_data(model_errors=model_errors,
                                                                                                                                     residuals=residuals,
                                                                                                                                     dataset_stdev=dataset_stdev)

        bin_values_cal, rms_residual_values_cal, num_values_per_bin_cal, number_of_bins_cal, ms_residual_values_cal, var_sq_residual_values_cal = ErrorUtils()._parse_error_data(model_errors=model_errors_cal,
                                                                                                                             residuals=residuals,
                                                                                                                             dataset_stdev=dataset_stdev)

        model_name = model.model.__class__.__name__
        if model_name == 'RandomForestRegressor':
            model_type = 'RF'
        elif model_name == 'GradientBoostingRegressor':
            model_type = 'GBR'
        elif model_name == 'ExtraTreesRegressor':
            model_type = 'ET'
        elif model_name == 'GaussianProcessRegressor':
            model_type = 'GPR'
        elif model_name == 'BaggingRegressor':
            model_type = 'BR'
        elif model_name == 'AdaBoostRegressor':
            model_type = 'ABR'

        if data_type not in ['train', 'test', 'leaveout']:
            print('Error: data_test_type must be one of "train", "test" or "leaveout"')
            exit()

        # Make RF error plot
        fig, ax = make_fig_ax(aspect_ratio=0.5, x_align=0.65)

        linear_uncal = LinearRegression(fit_intercept=True)
        linear_cal = LinearRegression(fit_intercept=True)

        # Only examine the bins that are well-sampled, i.e. have number of data points in them above a given threshold
        #well_sampled_number_uncal = round(well_sampled_fraction*np.sum(num_values_per_bin_uncal))
        rms_residual_values_wellsampled_uncal = rms_residual_values_uncal[np.where(num_values_per_bin_uncal > well_sampled_number)[0]]
        bin_values_wellsampled_uncal = bin_values_uncal[np.where(num_values_per_bin_uncal > well_sampled_number)[0]]
        num_values_per_bin_wellsampled_uncal = num_values_per_bin_uncal[np.where(num_values_per_bin_uncal > well_sampled_number)[0]]
        rms_residual_values_poorlysampled_uncal = rms_residual_values_uncal[np.where(num_values_per_bin_uncal <= well_sampled_number)[0]]
        bin_values_poorlysampled_uncal = bin_values_uncal[np.where(num_values_per_bin_uncal <= well_sampled_number)[0]]
        num_values_per_bin_poorlysampled_uncal = num_values_per_bin_uncal[np.where(num_values_per_bin_uncal <= well_sampled_number)[0]]

        yerr_uncal = list()
        for i, j, k in zip(var_sq_residual_values_uncal, num_values_per_bin_uncal, rms_residual_values_uncal):
            if j > 1:
                yerr_uncal.append(np.sqrt(i) / (2 * np.sqrt(j) * k))
            else:
                yerr_uncal.append(1)
        yerr_uncal = np.array(yerr_uncal)

        yerr_cal = list()
        for i, j, k in zip(var_sq_residual_values_cal, num_values_per_bin_cal, rms_residual_values_cal):
            if j > 1:
                yerr_cal.append(np.sqrt(i) / (2 * np.sqrt(j) * k))
            else:
                yerr_cal.append(1)
        yerr_cal = np.array(yerr_cal)

        yerr_wellsampled_uncal = yerr_uncal[np.where(num_values_per_bin_uncal > well_sampled_number)[0]]
        yerr_poorlysampled_uncal = yerr_uncal[np.where(num_values_per_bin_uncal <= well_sampled_number)[0]]

        #well_sampled_number_cal = round(well_sampled_fraction * np.sum(num_values_per_bin_cal))
        rms_residual_values_wellsampled_cal = rms_residual_values_cal[np.where(num_values_per_bin_cal > well_sampled_number)[0]]
        bin_values_wellsampled_cal = bin_values_cal[np.where(num_values_per_bin_cal > well_sampled_number)]
        num_values_per_bin_wellsampled_cal = num_values_per_bin_cal[np.where(num_values_per_bin_cal > well_sampled_number)[0]]
        rms_residual_values_poorlysampled_cal = rms_residual_values_cal[np.where(num_values_per_bin_cal <= well_sampled_number)[0]]
        bin_values_poorlysampled_cal = bin_values_cal[np.where(num_values_per_bin_cal <= well_sampled_number)[0]]
        num_values_per_bin_poorlysampled_cal = num_values_per_bin_cal[np.where(num_values_per_bin_cal <= well_sampled_number)[0]]

        yerr_wellsampled_cal = yerr_cal[np.where(num_values_per_bin_cal > well_sampled_number)[0]]
        yerr_poorlysampled_cal = yerr_cal[np.where(num_values_per_bin_cal <= well_sampled_number)[0]]

        ax.scatter(bin_values_wellsampled_uncal, rms_residual_values_wellsampled_uncal, s=80, color='gray', edgecolor='gray', alpha=0.7, label='uncalibrated')
        ax.scatter(bin_values_poorlysampled_uncal, rms_residual_values_poorlysampled_uncal, s=40, color='gray', edgecolor='gray', alpha=0.4)
        ax.errorbar(bin_values_wellsampled_uncal, rms_residual_values_wellsampled_uncal, yerr=yerr_wellsampled_uncal, ecolor='gray', capsize=2, linewidth=0, elinewidth=1)
        ax.errorbar(bin_values_poorlysampled_uncal, rms_residual_values_poorlysampled_uncal, yerr=yerr_poorlysampled_uncal, ecolor='gray', capsize=2, linewidth=0, elinewidth=1, alpha=0.4)

        ax.scatter(bin_values_wellsampled_cal, rms_residual_values_wellsampled_cal, s=80, color='blue', edgecolor='blue', alpha=0.7, label='calibrated')
        ax.scatter(bin_values_poorlysampled_cal, rms_residual_values_poorlysampled_cal, s=40, color='blue', edgecolor='blue', alpha=0.4)
        ax.errorbar(bin_values_wellsampled_cal, rms_residual_values_wellsampled_cal, yerr=yerr_wellsampled_cal, ecolor='blue', capsize=2, linewidth=0, elinewidth=1)
        ax.errorbar(bin_values_poorlysampled_cal, rms_residual_values_poorlysampled_cal, yerr=yerr_poorlysampled_cal, ecolor='blue', capsize=2, linewidth=0, elinewidth=1, alpha=0.4)

        ax.set_xlabel(str(model_type) + ' model errors / dataset stdev', fontsize=12)
        ax.set_ylabel('RMS Absolute residuals\n / dataset stdev', fontsize=12)
        ax.tick_params(labelsize=10)

        # Fit the line to all data, including the poorly sampled data, and weight data points by number of samples per bin
        linear_uncal.fit(np.array(bin_values_uncal).reshape(-1, 1), rms_residual_values_uncal,
                         sample_weight=num_values_per_bin_uncal)
        yfit_uncal = linear_uncal.predict(np.array(bin_values_uncal).reshape(-1, 1))
        ax.plot(bin_values_uncal, yfit_uncal, 'gray', linewidth=2)
        r2_uncal = r2_score(rms_residual_values_uncal, yfit_uncal, sample_weight=num_values_per_bin_uncal)

        slope_uncal = linear_uncal.coef_
        intercept_uncal = linear_uncal.intercept_

        # Fit the line to all data, including the poorly sampled data, and weight data points by number of samples per bin
        linear_cal.fit(np.array(bin_values_cal).reshape(-1, 1), rms_residual_values_cal,
                       sample_weight=num_values_per_bin_cal)
        yfit_cal = linear_cal.predict(np.array(bin_values_cal).reshape(-1, 1))
        ax.plot(bin_values_cal, yfit_cal, 'blue', linewidth=2)
        r2_cal = r2_score(rms_residual_values_cal, yfit_cal, sample_weight=num_values_per_bin_cal)

        slope_cal = linear_cal.coef_
        intercept_cal = linear_cal.intercept_

        divider = make_axes_locatable(ax)
        axbarx = divider.append_axes("top", 1.2, pad=0.12, sharex=ax)

        axbarx.bar(x=bin_values_uncal, height=num_values_per_bin_uncal, width=bin_values_uncal[1]-bin_values_uncal[0],
                   color='gray', edgecolor='gray', alpha=0.3)
        axbarx.bar(x=bin_values_cal, height=num_values_per_bin_cal, width=bin_values_cal[1] - bin_values_cal[0],
                   color='blue', edgecolor='blue', alpha=0.3)
        axbarx.tick_params(labelsize=10, axis='y')
        axbarx.tick_params(labelsize=0, axis='x')
        axbarx.set_ylabel('Counts', fontsize=12)

        #xmax = max(max(bin_values_uncal) + 0.05, 1.6)
        #ymax = max(1.3, max(rms_residual_values_uncal))

        xmax = round(max([max(bin_values_uncal), max(bin_values_cal)]) * 1.10, 2)
        ymax = round(max([max(rms_residual_values_uncal), max(rms_residual_values_cal)]) * 1.10, 2)

        axbarx.text(0.6 * xmax, round(0.9 * max(num_values_per_bin_uncal)), 'Total counts = ' + str(sum(num_values_per_bin_uncal)),
                    fontsize=10)

        ax.set_ylim(bottom=0, top=ymax)
        axbarx.set_ylim(bottom=0, top=round(max(num_values_per_bin_uncal) + 0.1*max(num_values_per_bin_uncal)))
        ax.set_xlim(left=0, right=xmax)

        ax.text(0.02, 0.9*ymax, 'R$^2$ = %3.2f ' % r2_uncal, fontdict={'fontsize': 8, 'color': 'gray'})
        ax.text(0.02, 0.8*ymax, 'slope = %3.2f ' % slope_uncal, fontdict={'fontsize': 8, 'color': 'gray'})
        ax.text(0.02, 0.7*ymax, 'intercept = %3.2f ' % intercept_uncal, fontdict={'fontsize': 8, 'color': 'gray'})
        ax.text(0.02, 0.6*ymax, 'R$^2$ = %3.2f ' % r2_cal, fontdict={'fontsize': 8, 'color': 'blue'})
        ax.text(0.02, 0.5*ymax, 'slope = %3.2f ' % slope_cal, fontdict={'fontsize': 8, 'color': 'blue'})
        ax.text(0.02, 0.4*ymax, 'intercept = %3.2f ' % intercept_cal, fontdict={'fontsize': 8, 'color': 'blue'})

        # Plot y = x line as reference point
        maxx = max(xmax, ymax)
        ax.plot([0, maxx], [0, maxx], 'k--', lw=2, color='red', alpha=0.5)

        ax.legend(loc='lower right', fontsize=8)

        fig.savefig(os.path.join(savepath, str(model_type) + '_residuals_vs_modelerror_' + str(data_type) + '_uncal_cal_overlay.png'),
                    dpi=image_dpi, bbox_inches='tight')

        if show_figure is True:
            plt.show()
        else:
            plt.close()

        return


class Histogram():
    """
    Class to generate histogram plots, such as histograms of residual values

    Args:
        None

    Methods:
        plot_histogram: method to plot a basic histogram of supplied data
            Args:
                df: (pd.DataFrame), dataframe or series of data to plot as a histogram

                savepath: (str), string denoting the save path for the figure image

                file_name: (str), string denoting the character of the file name, e.g. train vs. test

                x_label: (str), string denoting the  property name

                show_figure: (bool), whether or not to show the figure output (e.g. when using Jupyter notebook)

            Returns:
                None

        plot_residuals_histogram: method to plot a histogram of residual values
            Args:
                y_true: (pd.Series), series of true y data

                y_pred: (pd.Series), series of predicted y data

                savepath: (str), string denoting the save path for the figure image

                file_name: (str), string denoting the character of the file name, e.g. train vs. test

                show_figure: (bool), whether or not to show the figure output (e.g. when using Jupyter notebook)

            Returns:
                None

        _get_histogram_bins: Method to obtain the number of bins to use when plotting a histogram
            Args:
                df: (pandas Series or numpy array), array of y data used to construct histogram

            Returns:
                num_bins: (int), the number of bins to use when plotting a histogram

    """
    @classmethod
    def plot_histogram(cls, df, savepath, file_name, x_label, show_figure=False, file_extension='.csv', image_dpi=250):
        # Make the dataframe 1D if it isn't
        df = check_dimensions(df)

        # make fig and ax, use x_align when placing text so things don't overlap
        x_align = 0.70
        fig, ax = make_fig_ax(aspect_ratio=0.5, x_align=x_align)

        # Get num_bins using smarter method
        num_bins = cls._get_histogram_bins(df=df)

        # do the actual plotting
        ax.hist(df, bins=num_bins, color='b', edgecolor='k')

        # normal text stuff
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel('Number of occurrences', fontsize=14)

        plot_stats(fig, dict(df.describe()), x_align=x_align, y_align=0.90, fontsize=12)

        # Save data to excel file and image
        if file_extension == '.xlsx':
            df.to_excel(os.path.join(savepath, file_name + '.xlsx'))
            df.describe().to_excel(os.path.join(savepath, file_name + '_statistics.xlsx'))
        elif file_extension == '.csv':
            df.to_csv(os.path.join(savepath, file_name + '.csv'))
            df.describe().to_csv(os.path.join(savepath, file_name + '_statistics.csv'))
        fig.savefig(os.path.join(savepath, file_name + '.png'), dpi=image_dpi, bbox_inches='tight')
        if show_figure == True:
            plt.show()
        else:
            plt.close()
        return

    @classmethod
    def plot_residuals_histogram(cls, y_true, y_pred, savepath, show_figure=False, file_extension='.csv', image_dpi=250, file_name='residual_histogram'):
        y_true = check_dimensions(y_true)
        y_pred = check_dimensions(y_pred)
        residuals = y_pred-y_true
        cls.plot_histogram(df=residuals,
                           savepath=savepath,
                           file_name=file_name,
                           x_label='Residuals',
                           show_figure=show_figure,
                           file_extension=file_extension,
                           image_dpi=image_dpi)
        return

    @classmethod
    def _get_histogram_bins(cls, df):

        bin_dividers = np.linspace(df.shape[0], 0.05*df.shape[0], df.shape[0])
        bin_list = list()
        try:
            for divider in bin_dividers:
                if divider == 0:
                    continue
                bins = int((df.shape[0])/divider)
                if bins < df.shape[0]/2:
                    bin_list.append(bins)
        except:
            num_bins = 10
        if len(bin_list) > 0:
            num_bins = max(bin_list)
        else:
            num_bins = 10
        return num_bins


class Line():
    '''
    Class containing methods for constructing line plots

    Args:
        None

    Methods:
        plot_learning_curve: Method used to plot both data and feature learning curves

        Args:
            train_sizes: (numpy array), array of x-axis values, such as fraction of data used or number of features

            train_mean: (numpy array), array of training data mean values, averaged over some type/number of CV splits

            test_mean: (numpy array), array of test data mean values, averaged over some type/number of CV splits

            train_stdev: (numpy array), array of training data standard deviation values, from some type/number of CV splits

            test_stdev: (numpy array), array of test data standard deviation values, from some type/number of CV splits

            score_name: (str), type of score metric for learning curve plotting; used in y-axis label

            learning_curve_type: (str), type of learning curve employed: 'sample_learning_curve' or 'feature_learning_curve'

            savepath: (str), path to save the plotted learning curve to

        Returns:
            None

    '''
    @classmethod
    def plot_learning_curve(cls, train_sizes, train_mean, test_mean, train_stdev, test_stdev, score_name,
                            learning_curve_type, savepath, image_dpi=250):

        # Set image aspect ratio (do custom for learning curve):
        w, h = figaspect(0.75)
        fig = Figure(figsize=(w, h))
        FigureCanvas(fig)
        gs = plt.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0:, 0:])

        max_x = max(train_sizes)
        min_x = min(train_sizes)

        max_y, min_y = recursive_max_and_min([
            train_mean,
            train_mean + train_stdev,
            train_mean - train_stdev,
            test_mean,
            test_mean + test_stdev,
            test_mean - test_stdev,
        ])

        max_x = round(float(max_x), rounder(max_x - min_x))
        min_x = round(float(min_x), rounder(max_x - min_x))
        max_y = round(float(max_y), rounder(max_y - min_y))
        min_y = round(float(min_y), rounder(max_y - min_y))
        _set_tick_labels_different(ax, max_x, min_x, max_y, min_y)

        # plot and collect handles h1 and h2 for making legend
        h1 = ax.plot(train_sizes, train_mean, '-o', color='blue', markersize=10, alpha=0.7)[0]
        ax.fill_between(train_sizes, train_mean - train_stdev, train_mean + train_stdev,
                        alpha=0.1, color='blue')
        h2 = ax.plot(train_sizes, test_mean, '-o', color='red', markersize=10, alpha=0.7)[0]
        ax.fill_between(train_sizes, test_mean - test_stdev, test_mean + test_stdev,
                        alpha=0.1, color='red')
        ax.legend([h1, h2], ['train score', 'validation score'], loc='center right', fontsize=12)
        if learning_curve_type == 'data_learning_curve':
            ax.set_xlabel('Number of training data points', fontsize=16)
        elif learning_curve_type == 'feature_learning_curve':
            ax.set_xlabel('Number of features selected', fontsize=16)
        else:
            raise ValueError(
                'The param "learning_curve_type" must be either "data_learning_curve" or "feature_learning_curve"')
        ax.set_ylabel(score_name, fontsize=16)
        fig.savefig(os.path.join(savepath, learning_curve_type + '.png'), dpi=image_dpi, bbox_inches='tight')
        return


class Classification():
    '''
    Classification plots

    Args:
        None

    Methods:
        plot_classification_report: Method used to plot the classification report

        Args:
            savepath: (str), path to save the plotted learning curve to

            data_type: (str), string denoting the data type, e.g. train, test, leftout

            y_true: (pd.Series), series of true y data

            y_pred: (pd.Series), series of predicted y data

            show_figure: (bool), whether or not to show the figure output (e.g. when using Jupyter notebook)

        Returns:
            None

    '''

    def createClassificationReport(savepath, report_dict, show_figure=False, data_type=''):
        # Parse report_dict data for export
        class_metrics = list(report_dict.items())[:-3]
        all_metric_names = list(class_metrics[0][1].keys())
        class_names = [x[0] for x in class_metrics]
        all_class_values = [list(x[1].values()) for x in class_metrics]
        class_values_no_support = np.array([list(x[1].values())[:-1] for x in class_metrics])
        metrics = list(report_dict.items())[-3:]
        meta_names = [m[0] for m in metrics]
        meta_values = np.array([([m[1]] if isinstance(m[1], float) else list(m[1].values())) for m in metrics], dtype=object)
        report_as_array = []

        report_as_array.append([""]+all_metric_names)
        for i in range(len(class_names)):
            report_as_array.append([class_names[i]] + all_class_values[i])
        for i in range(len(meta_names)):
            report_as_array.append([meta_names[i]] + meta_values[i])

        # EXCEL
        pd.DataFrame(report_as_array).to_excel(os.path.join(savepath, 'classification_report_'+str(data_type)+'.xlsx'), index=False, header=None)

        # PLOT
        plot_metric_names = all_metric_names[:-1]

        dimensions = max(5, len(class_names))
        fig, ax = plt.subplots(figsize=(dimensions, dimensions))
        im = ax.imshow(class_values_no_support)
        ax.set_xticks(np.arange(len(plot_metric_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(plot_metric_names)
        ax.set_yticklabels(class_names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        font = {'family': 'serif',
                'weight': 'normal',
                'size': 10,
                }

        for i in range(len(class_names)):
            for j in range(len(plot_metric_names)):
                text = ax.text(j, i, round(class_values_no_support[i, j], 3),
                               ha="center",
                               va="center",
                               color="w",
                               fontdict=font)

        ax.set_title("Classification Report")
        fig.tight_layout()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="15%", pad=0.05)
        plt.colorbar(im, cax=cax)
        fig.savefig(os.path.join(savepath, 'classification_report_'+str(data_type)+'.png'), bbox_inches='tight')
        plt.show() if show_figure else plt.close()

    @classmethod
    def plot_classification_report(cls, savepath, data_type, y_true, y_pred, show_figure):
        report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
        Classification.createClassificationReport(savepath, report, show_figure, data_type)

    @classmethod
    def doProba(cls, model, X_test):
        print("="*4 + "doProba" + "="*4)
        print("model:")
        print(model)
        # print("X_test:")
        # print(X_test)
        print(f"hasattr(model, 'predict_proba') {hasattr(model, 'predict_proba')}")

        if hasattr(model, 'predict_proba'):
            try:
                print("predict_proba")
                foo = model.predict_proba(X_test)
                print(foo)
            except NotFittedError as e:
                print(e)
        print("="*18)

def plot_feature_occurrence(savepath, feature, occurrence):
    """
    Function to plot the occurrence of each feature in all of the splits

    Args:
        savepath: (str), string denoting the path to save output to

        feature: (list) the list of features

        occurrence: (list) the list of occurrence of each feature

    Returns:
        None.
    """
    plt.tick_params(axis='y', labelsize=10)
    plt.barh(feature, occurrence)
    plt.xlabel("Occurrence")
    plt.ylabel("Feature")
    plt.title("Occurrence of Features")
    plt.savefig(os.path.join(savepath, 'Feature_Occurrence.png'))
    plt.clf()
    return

def plot_avg_score_vs_occurrence(savepath, occurrence, score, std_score):
    """
    Function to plot the average score of each feature against their occurrence in all of the splits

    Args:
        savepath: (str), string denoting the path to save output to

        occurrence: (list), the list of occurrence of each feature

        score: (list), the list of the feature ranking score of each feature

        std_score(list), the list of standard deviation of the feature ranking score

    Returns:
        None.
    """
    plt.plot(occurrence, score, marker='o')
    plt.fill_between(occurrence, score - std_score, score + std_score, alpha=0.1, color='blue')
    plt.title('Avg Score vs Occurrence')
    plt.xlabel('Occurrence')
    plt.ylabel('Avg Score')
    plt.savefig(os.path.join(savepath + '/AvgScoreVsOccurrence.png'))
    plt.clf()
    return

def make_plots(plots, y_true, y_pred, groups, dataset_stdev, metrics, model, residuals, model_errors, has_model_errors,
               savepath, data_type, X_test=None, show_figure=False, recalibrate_errors=False, model_errors_cal=None, splits_summary=False,
               file_extension='.csv', image_dpi=250):
    """
    Helper function to make collections of different types of plots after a single or multiple data splits are evaluated.

    Args:
        plots: (list of str), list denoting which types of plots to make. Viable entries are "Scatter", "Histogram", "Error"

        y_true: (pd.Series), series containing the true y data

        y_pred: (pd.Series), series containing the predicted y data

        groups: (list), list denoting the group label for each data point

        dataset_stdev: (float), the dataset standard deviation

        metrics: (list of str), list denoting the metric names to evaluate. See mastml.metrics.Metrics.metrics_zoo_ for full list

        model: (mastml.models object), a MAST-ML model object, e.g. SklearnModel or EnsembleModel

        residuals: (pd.Series), series containing the residuals (true model errors)

        model_errors: (pd.Series), series containing the as-obtained uncalibrated model errors

        has_model_errors: (bool), whether the model type used can be subject to UQ and thus have model errors calculated

        savepath: (str), string denoting the path to save output to

        data_type: (str), string denoting the data type analyzed, e.g. train, test, leftout

        show_figure: (bool), whether or not the generated figure is output to the notebook screen (default False)

        recalibrate_errors: (bool), whether or not the model errors have been recalibrated (default False)

        model_errors_cal: (pd.Series), series containing the calibrated predicted model errors

        splits_summary: (bool), whether or not the data used in the plots comes from a collection of many splits (default False), False denotes a single split folder

    Returns:
        None.

    """

    if 'Histogram' in plots:
        try:
            Histogram.plot_residuals_histogram(y_true=y_true,
                                               y_pred=y_pred,
                                               savepath=savepath,
                                               file_name='residual_histogram_'+str(data_type),
                                               show_figure=show_figure,
                                               file_extension=file_extension,
                                               image_dpi=image_dpi)
        except:
            print('Warning: unable to make Histogram.plot_residuals_histogram. Skipping...')
    if 'Scatter' in plots:
        try:
            Scatter.plot_predicted_vs_true(y_true=y_true,
                                           y_pred=y_pred,
                                           savepath=savepath,
                                           x_label='values',
                                           data_type=data_type,
                                           metrics_list=metrics,
                                           show_figure=show_figure,
                                           ebars=None,
                                           file_extension=file_extension,
                                           image_dpi=image_dpi,
                                           groups=None)
            if groups is not None:
                Scatter.plot_predicted_vs_true(y_true=y_true,
                                               y_pred=y_pred,
                                               savepath=savepath,
                                               x_label='values',
                                               data_type=data_type,
                                               metrics_list=metrics,
                                               show_figure=show_figure,
                                               ebars=None,
                                               file_extension=file_extension,
                                               image_dpi=image_dpi,
                                               groups=groups)
            if recalibrate_errors == True:
                Scatter.plot_predicted_vs_true(y_true=y_true,
                                               y_pred=y_pred,
                                               savepath=savepath,
                                               x_label='values',
                                               data_type=data_type,
                                               metrics_list=metrics,
                                               show_figure=show_figure,
                                               ebars=model_errors_cal,
                                               file_extension=file_extension,
                                               image_dpi=image_dpi,
                                               groups=None)
                if groups is not None:
                    Scatter.plot_predicted_vs_true(y_true=y_true,
                                                   y_pred=y_pred,
                                                   savepath=savepath,
                                                   x_label='values',
                                                   data_type=data_type,
                                                   metrics_list=metrics,
                                                   show_figure=show_figure,
                                                   ebars=model_errors_cal,
                                                   file_extension=file_extension,
                                                   image_dpi=image_dpi,
                                                   groups=groups)
        except:
            print('Warning: unable to make Scatter.plot_predicted_vs_true plot. Skipping...')
        if splits_summary is True:
            if data_type != 'leaveout':
                try:
                    Scatter.plot_best_worst_split(savepath=savepath,
                                                  data_type=data_type,
                                                  x_label='values',
                                                  metrics_list=metrics,
                                                  show_figure=show_figure,
                                                  file_extension=file_extension,
                                                  image_dpi=image_dpi)
                except:
                    print('Warning: unable to make Scatter.plot_best_worst_split plot. Skipping...')
                try:
                    Scatter.plot_best_worst_per_point(savepath=savepath,
                                                      data_type=data_type,
                                                      x_label='values',
                                                      metrics_list=metrics,
                                                      show_figure=show_figure,
                                                      file_extension=file_extension,
                                                      image_dpi=image_dpi)
                except:
                    print('Warning: unable to make Scatter.plot_best_worst_per_point plot. Skipping...')
            try:
                Scatter.plot_predicted_vs_true_bars(savepath=savepath,
                                                    data_type=data_type,
                                                    x_label='values',
                                                    metrics_list=metrics,
                                                    show_figure=show_figure,
                                                    file_extension=file_extension,
                                                    image_dpi=image_dpi,
                                                    groups=None)
                if groups is not None:
                    Scatter.plot_predicted_vs_true_bars(savepath=savepath,
                                                        data_type=data_type,
                                                        x_label='values',
                                                        metrics_list=metrics,
                                                        show_figure=show_figure,
                                                        file_extension=file_extension,
                                                        image_dpi=image_dpi,
                                                        groups=groups)
                if recalibrate_errors == True:
                    Scatter.plot_predicted_vs_true_bars(savepath=savepath,
                                                        data_type=data_type,
                                                        x_label='values',
                                                        metrics_list=metrics,
                                                        show_figure=show_figure,
                                                        ebars=model_errors_cal,
                                                        file_extension=file_extension,
                                                        image_dpi=image_dpi,
                                                        groups=None)
                    if groups is not None:
                        Scatter.plot_predicted_vs_true_bars(savepath=savepath,
                                                            data_type=data_type,
                                                            x_label='values',
                                                            metrics_list=metrics,
                                                            show_figure=show_figure,
                                                            ebars=model_errors_cal,
                                                            file_extension=file_extension,
                                                            image_dpi=image_dpi,
                                                            groups=groups)
            except:
                print('Warning: unable to make Scatter.plot_predicted_vs_true_bars plot. Skipping...')

    if 'Error' in plots:
        try:
            Error.plot_qq(residuals=residuals,
                          savepath=savepath,
                          data_type=data_type,
                          show_figure=show_figure,
                          image_dpi=image_dpi)
        except:
            print('Warning: unable to make Error.plot_qq plot. Skipping...')
        try:
            Error.plot_normalized_error(residuals=residuals,
                                        savepath=savepath,
                                        data_type=data_type,
                                        model_errors=model_errors,
                                        show_figure=show_figure,
                                        file_extension=file_extension,
                                        image_dpi=image_dpi)
        except:
            print('Warning: unable to make Error.plot_normalized_error plot. Skipping...')
        try:
            Error.plot_cumulative_normalized_error(residuals=residuals,
                                                   savepath=savepath,
                                                   data_type=data_type,
                                                   model_errors=model_errors,
                                                   show_figure=show_figure,
                                                    file_extension=file_extension,
                                                    image_dpi=image_dpi)
        except:
            print('Warning: unable to make Error.plot_cumulative_normalized_error plot. Skipping...')
        if has_model_errors is True:
            try:
                Error.plot_rstat(savepath=savepath,
                                 data_type=data_type,
                                 model_errors=model_errors,
                                 residuals=residuals,
                                 show_figure=show_figure,
                                 is_calibrated=False,
                                 image_dpi=image_dpi)
            except:
                print('Warning: unable to make Error.plot_rstat plot. Skipping...')
            try:
                Error.plot_real_vs_predicted_error(savepath=savepath,
                                                   model=model,
                                                   data_type=data_type,
                                                   model_errors=model_errors,
                                                   residuals=residuals,
                                                   dataset_stdev=dataset_stdev,
                                                   show_figure=show_figure,
                                                   is_calibrated=False,
                                                    image_dpi=image_dpi)
            except:
                print('Warning: unable to make Error.plot_real_vs_predicted_error plot. Skipping...')
            if recalibrate_errors is True:
                try:
                    Error.plot_rstat(savepath=savepath,
                                     data_type=data_type,
                                     residuals=residuals,
                                     model_errors=model_errors_cal,
                                     show_figure=show_figure,
                                     is_calibrated=True,
                                     image_dpi=image_dpi)
                except:
                    print('Warning: unable to make Error.plot_rstat plot. Skipping...')
                try:
                    Error.plot_rstat_uncal_cal_overlay(savepath=savepath,
                                                       data_type=data_type,
                                                       residuals=residuals,
                                                       model_errors=model_errors,
                                                       model_errors_cal=model_errors_cal,
                                                       show_figure=False,
                                                       image_dpi=image_dpi)
                except:
                    print('Warning: unable to make Error.plot_rstat_uncal_cal_overlay plot. Skipping...')
                try:
                    Error.plot_real_vs_predicted_error(savepath=savepath,
                                                       model=model,
                                                       data_type=data_type,
                                                       residuals=residuals,
                                                       model_errors=model_errors_cal,
                                                       dataset_stdev=dataset_stdev,
                                                       show_figure=show_figure,
                                                       is_calibrated=True,
                                                        image_dpi = image_dpi)
                except:
                    print('Warning: unable to make Error.plot_real_vs_predicted_error plot. Skipping...')
                try:
                    Error.plot_real_vs_predicted_error_uncal_cal_overlay(savepath=savepath,
                                                                         model=model,
                                                                         data_type=data_type,
                                                                         model_errors=model_errors,
                                                                         model_errors_cal=model_errors_cal,
                                                                         residuals=residuals,
                                                                         dataset_stdev=dataset_stdev,
                                                                         show_figure=False,
                                                                         image_dpi=image_dpi)
                except:
                    print('Warning: unable to make Error.plot_real_vs_predicted_error_uncal_cal_overlay plot. Skipping...')
    if 'Classification' in plots:
        Classification.plot_classification_report(savepath=savepath, data_type=data_type, y_true=y_true, y_pred=y_pred, show_figure=show_figure)
        Classification.doProba(model=model, X_test=X_test)
    return


def check_dimensions(y):
    """
    Method to check the dimensions of supplied data. Plotters need data to be 1D and often data is passed in as 2D

    Args:

        y: (numpy array or pd.DataFrame), array or dataframe of data used for plotting

    Returns:

        y: (pd.Series), series that is now 1D

    """
    if len(y.shape) > 1:
        if type(y) == pd.core.frame.DataFrame:
            y = pd.DataFrame.squeeze(y)
        elif type(y) == np.ndarray:
            y = pd.DataFrame(y.ravel()).squeeze()
            #y = y.ravel()
    else:
        if type(y) == np.ndarray:
            y = pd.DataFrame(y).squeeze()
    return y


def reset_index(y):
    return pd.DataFrame(np.array(y))


def trim_array(arr_list):
    """
    Method used to trim a set of arrays to make all arrays the same shape

    Args:

        arr_list: (list), list of numpy arrays, where arrays are different sizes

    Returns:

        arr_list: (), list of trimmed numpy arrays, where arrays are same size

    """

    # TODO: a better way to handle arrays with very different shapes? Otherwise average only uses # of points of smallest array
    # Need to make arrays all same shapes if they aren't
    sizes = [arr.shape[0] for arr in arr_list]
    size_min = min(sizes)
    arr_list_ = list()
    for i, arr in enumerate(arr_list):
        if arr.shape[0] > size_min:
            while arr.shape[0] > size_min:
                arr = np.delete(arr, -1)
        arr_list_.append(arr)
    arr_list = arr_list_
    return arr_list


def rounder(delta):
    """
    Method to obtain number of decimal places to report on plots

    Args:

        delta: (float), a float representing the change in two y values on a plot, used to obtain the plot axis spacing size

    Return:

        (int), an integer denoting the number of decimal places to use

    """
    if 0.001 <= delta < 0.01:
        return 3
    elif 0.01 <= delta < 0.1:
        return 2
    elif 0.1 <= delta < 1:
        return 1
    elif 1 <= delta < 100000:
        return 0
    else:
        return 0


def stat_to_string(name, value, nice_names):
    """
    Method that converts a metric object into a string for displaying on a plot

    Args:

        name: (str), long name of a stat metric or quantity

        value: (float), value of the metric or quantity

    Return:

        (str), a string of the metric name, adjusted to look nicer for inclusion on a plot

    """

    " Stringifies the name value pair for display within a plot "
    if name in nice_names:
        name = nice_names[name]
    else:
        name = name.replace('_', ' ')

    # has a name only
    if not value:
        return name
    # has a mean and std
    if isinstance(value, tuple):
        mean, std = value
        return f'{name}:' + '\n\t' + f'{mean:.3f}' + r'$\pm$' + f'{std:.3f}'
    # has a name and value only
    if isinstance(value, int) or (isinstance(value, float) and value % 1 == 0):
        return f'{name}: {int(value)}'
    if isinstance(value, float):
        return f'{name}: {value:.3f}'
    return f'{name}: {value}'  # probably a string


def plot_stats(fig, stats, x_align=0.65, y_align=0.90, font_dict=dict(), fontsize=14):
    """
    Method that prints stats onto the plot. Goes off screen if they are too long or too many in number.

    Args:

        fig: (matplotlib figure object), a matplotlib figure object

        stats: (dict), dict of statistics to be included with a plot

        x_align: (float), float denoting x position of where to align display of stats on a plot

        y_align: (float), float denoting y position of where to align display of stats on a plot

        font_dict: (dict), dict of matplotlib font options to alter display of stats on plot

        fontsize: (int), the fontsize of stats to display on plot

    Returns:

        None

    """

    stat_str = '\n'.join(stat_to_string(name, value, nice_names=nice_names())
                         for name, value in stats.items())

    fig.text(x_align, y_align, stat_str,
             verticalalignment='top', wrap=True, fontdict=font_dict, fontproperties=FontProperties(size=fontsize))


def make_fig_ax(aspect_ratio=0.5, x_align=0.65, left=0.10):
    """
    Method to make matplotlib figure and axes objects. Using Object Oriented interface from https://matplotlib.org/gallery/api/agg_oo_sgskip.html

    Args:

        aspect_ratio: (float), aspect ratio for figure and axes creation

        x_align: (float), x position to draw edge of figure. Needed so can display stats alongside plot

        left: (float), the leftmost position to draw edge of figure

    Returns:

        fig: (matplotlib fig object), a matplotlib figure object with the specified aspect ratio

        ax: (matplotlib ax object), a matplotlib axes object with the specified aspect ratio

    """
    # Set image aspect ratio:
    w, h = figaspect(aspect_ratio)
    fig = plt.figure(figsize=(w, h))
    #fig = Figure(figsize=(w, h))
    FigureCanvas(fig)

    # Set custom positioning, see this guide for more details:
    # https://python4astronomers.github.io/plotting/advanced.html
    #left   = 0.10
    bottom = 0.15
    right = 0.01
    top = 0.05
    width = x_align - left - right
    height = 1 - bottom - top
    ax = fig.add_axes((left, bottom, width, height), frameon=True)
    fig.set_tight_layout(False)

    return fig, ax


def make_fig_ax_square(aspect='equal', aspect_ratio=1):
    """
    Method to make square shaped matplotlib figure and axes objects. Using Object Oriented interface from

    https://matplotlib.org/gallery/api/agg_oo_sgskip.html

    Args:

        aspect: (str), 'equal' denotes x and y aspect will be equal (i.e. square)

        aspect_ratio: (float), aspect ratio for figure and axes creation

    Returns:

        fig: (matplotlib fig object), a matplotlib figure object with the specified aspect ratio

        ax: (matplotlib ax object), a matplotlib axes object with the specified aspect ratio

    """
    # Set image aspect ratio:
    w, h = figaspect(aspect_ratio)
    fig = Figure(figsize=(w, h))
    FigureCanvas(fig)
    ax = fig.add_subplot(111, aspect=aspect)

    return fig, ax


def make_axis_same(ax, max1, min1):
    """
    Method to make the x and y ticks for each axis the same. Useful for parity plots

    Args:

        ax: (matplotlib axis object), a matplotlib axes object

        max1: (float), the maximum value of a particular axis

        min1: (float), the minimum value of a particular axis

    Returns:

        None

    """
    if max1 - min1 > 5:
        step = (int(max1) - int(min1)) // 3
        ticks = range(int(min1), int(max1)+step, step)
    else:
        ticks = np.linspace(min1, max1, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


def nice_mean(ls):
    """
    Method to return mean of a list or equivalent array with NaN values

    Args:

        ls: (list), list of values

    Returns:

        (numpy array), array containing mean of list of values or NaN if list has no values

    """

    if len(ls) > 0:
        return np.mean(ls)
    return np.nan


def nice_std(ls):
    """
    Method to return standard deviation of a list or equivalent array with NaN values

    Args:

        ls: (list), list of values

    Returns:

        (numpy array), array containing standard deviation of list of values or NaN if list has no values

    """
    if len(ls) > 0:
        return np.std(ls)
    return np.nan


def round_down(num, divisor):
    """
    Method to return a rounded down number

    Args:

        num: (float), a number to round down

        divisor: (int), divisor to denote how to round down

    Returns:

        (float), the rounded-down number

    """

    return num - (num % divisor)


def round_up(num, divisor):
    """
    Method to return a rounded up number

    Args:

        num: (float), a number to round up

        divisor: (int), divisor to denote how to round up

    Returns:

        (float), the rounded-up number

    """
    return float(math.ceil(num / divisor)) * divisor


def get_divisor(high, low):
    """
    Method to obtain a sensible divisor based on range of two values

    Args:

        high: (float), a max data value

        low: (float), a min data value

    Returns:

        divisor: (float), a number used to make sensible axis ticks

    """

    delta = high-low
    divisor = 10
    if delta > 1000:
        divisor = 100
    if delta < 1000:
        if delta > 100:
            divisor = 10
        if delta < 100:
            if delta > 10:
                divisor = 1
            if delta < 10:
                if delta > 1:
                    divisor = 0.1
                if delta < 1:
                    if delta > 0.01:
                        divisor = 0.001
                else:
                    divisor = 0.001
    return divisor


def recursive_max(arr):
    """
    Method to recursively find the max value of an array of iterables.

    Credit: https://www.linkedin.com/pulse/ask-recursion-during-coding-interviews-identify-good-talent-veteanu/

    Args:

        arr: (numpy array), an array of values or iterables

    Returns:

        (float), max value in arr

    """
    return max(
        recursive_max(e) if isinstance(e, Iterable) else e
        for e in arr
    )


def recursive_min(arr):
    """
    Method to recursively find the min value of an array of iterables.

    Credit: https://www.linkedin.com/pulse/ask-recursion-during-coding-interviews-identify-good-talent-veteanu/

    Args:

        arr: (numpy array), an array of values or iterables

    Returns:

        (float), min value in arr

    """

    return min(
        recursive_min(e) if isinstance(e, Iterable) else e
        for e in arr
    )


def recursive_max_and_min(arr):
    """
    Method to recursively return max and min of values or iterables in array

    Args:

        arr: (numpy array), an array of values or iterables

    Returns:

        (tuple), tuple containing max and min of arr

    """
    return recursive_max(arr), recursive_min(arr)


def _set_tick_labels(ax, maxx, minn):
    """
    Method that sets the x and y ticks to be in the same range

    Args:

        ax: (matplotlib axes object), a matplotlib axes object

        maxx: (float), a maximum value

        minn: (float), a minimum value

    Returns:

        None

    """
    _set_tick_labels_different(ax, maxx, minn, maxx, minn)  # I love it when this happens


def _set_tick_labels_different(ax, max_tick_x, min_tick_x, max_tick_y, min_tick_y):
    """
    Method that sets the x and y ticks, when the axes have different ranges

    Args:

        ax: (matplotlib axes object), a matplotlib axes object

        max_tick_x: (float), the maximum tick value for the x axis

        min_tick_x: (float), the minimum tick value for the x axis

        max_tick_y: (float), the maximum tick value for the y axis

        min_tick_y: (float), the minimum tick value for the y axis

    Returns:

        None

    """

    tickvals_x = nice_range(min_tick_x, max_tick_x)
    tickvals_y = nice_range(min_tick_y, max_tick_y)

    if tickvals_x[-1]-tickvals_x[len(tickvals_x)-2] < tickvals_x[len(tickvals_x)-3]-tickvals_x[len(tickvals_x)-4]:
        tickvals_x = tickvals_x[:-1]
    if tickvals_y[-1]-tickvals_y[len(tickvals_y)-2] < tickvals_y[len(tickvals_y)-3]-tickvals_y[len(tickvals_y)-4]:
        tickvals_y = tickvals_y[:-1]
    #tickvals_x = _clean_tick_labels(tickvals=tickvals_x, delta=max_tick_x-min_tick_x)
    #tickvals_y = _clean_tick_labels(tickvals=tickvals_y, delta=max_tick_y - min_tick_y)

    ax.set_xticks(ticks=tickvals_x)
    ax.set_yticks(ticks=tickvals_y)

    ticklabels_x = [str(tick) for tick in tickvals_x]
    ticklabels_y = [str(tick) for tick in tickvals_y]

    rotation = 0
    # Look at length of x tick labels to see if may be possibly crowded. If so, rotate labels
    tick_length = len(str(tickvals_x[1]))
    if tick_length >= 4:
        rotation = 45
    ax.set_xticklabels(labels=ticklabels_x, fontsize=14, rotation=rotation)
    ax.set_yticklabels(labels=ticklabels_y, fontsize=14)


def _clean_tick_labels(tickvals, delta):
    """
    Method to attempt to clean up axis tick values so they don't overlap from being too dense

    Args:

        tickvals: (list), a list containing the initial axis tick values

        delta: (float), number representing the numerical difference of two ticks

    Returns:

        tickvals_clean: (list), a list containing the updated axis tick values

    """
    tickvals_clean = list()
    if delta >= 100:
        for i, val in enumerate(tickvals):
            if i <= len(tickvals)-1:
                if tickvals[i]-tickvals[i-1] >= 100:
                    tickvals_clean.append(val)
    else:
        tickvals_clean = tickvals
    return tickvals_clean

# Math utilities to aid plot_helper to make ranges


def nice_range(lower, upper):
    """
    Method to create a range of values, including the specified start and end points, with nicely spaced intervals

    Args:

        lower: (float or int), lower bound of range to create

        upper: (float or int), upper bound of range to create

    Returns:

        (list), list of numerical values in established range

    """

    flipped = 1  # set to -1 for inverted

    # Case for validation where nan is passed in
    if np.isnan(lower):
        lower = 0
    if np.isnan(upper):
        upper = 0.1

    if upper < lower:
        upper, lower = lower, upper
        flipped = -1
    return [_int_if_int(x) for x in _nice_range_helper(lower, upper)][::flipped]


def _nice_range_helper(lower, upper):
    """
    Method to help make a better range of axis ticks

    Args:

        lower: (float), lower value of axis ticks

        upper: (float), upper value of axis ticks

    Returns:

        upper: (float), modified upper tick value fixed based on set of axis ticks

    """
    steps = 8
    diff = abs(lower - upper)

    # special case where lower and upper are the same
    if diff == 0:
        return [lower, ]

    # the exact step needed
    step = diff / steps

    # a rough estimate of best step
    step = _nearest_pow_ten(step)  # whole decimal increments

    # tune in one the best step size
    factors = [0.1, 0.2, 0.5, 1, 2, 5, 10]

    # use this to minimize how far we are from ideal step size
    def best_one(steps_factor):
        steps_count, factor = steps_factor
        return abs(steps_count - steps)
    n_steps, best_factor = min([(diff / (step * f), f) for f in factors], key=best_one)

    #print('should see n steps', ceil(n_steps + 2))
    # multiply in the optimal factor for getting as close to ten steps as we can
    step = step * best_factor

    # make the bounds look nice
    lower = _three_sigfigs(lower)
    upper = _three_sigfigs(upper)

    start = _round_up(lower, step)

    # prepare for iteration
    x = start  # pointless init
    i = 0

    # itereate until we reach upper
    while x < upper - step:
        x = start + i * step
        yield _three_sigfigs(x)  # using sigfigs because of floating point error
        i += 1

    # finish off with ending bound
    yield upper


def _three_sigfigs(x):
    """
    Method invoking special case of _n_sigfigs to return 3 sig figs

    Args:

        x: (float), an axis tick number

    Returns:

        (float), number of sig figs (always 3)

    """
    return _n_sigfigs(x, 3)


def _n_sigfigs(x, n):
    """
    Method to return number of sig figs to use for axis ticks

    Args:

        x: (float), an axis tick number

    Returns:

        (float), number of sig figs

    """
    sign = 1
    if x == 0:
        return 0
    if x < 0:  # case for negatives
        x = -x
        sign = -1
    if x < 1:
        base = n - round(log(x, 10))
    else:
        base = (n-1) - round(log(x, 10))
    return sign * round(x, base)


def _nearest_pow_ten(x):
    """
    Method to return the nearest power of ten for an axis tick value

    Args:

        x: (float), an axis tick number

    Returns:

        (float), nearest power of ten of x

    """
    sign = 1
    if x == 0:
        return 0
    if x < 0:  # case for negatives
        x = -x
        sign = -1
    return sign*10**ceil(log(x, 10))


def _int_if_int(x):
    """
    Method to return integer mapped value of x

    Args:

        x: (float or int), a number

    Returns:

        x: (float), value of x mapped as integer

    """
    if int(x) == x:
        return int(x)
    return x


def _round_up(x, inc):
    """
    Method to round up the value of x

    Args:

        x: (float or int), a number

        inc: (float), an increment for axis ticks

    Returns:

        (float), value of x rounded up

    """
    sign = 1
    if x < 0:  # case for negative
        x = -x
        sign = -1

    return sign * inc * ceil(x / inc)


def nice_names():
    nice_names = {
        # classification:
        'accuracy': 'Accuracy',
        'f1_binary': '$F_1$',
        'f1_macro': 'f1_macro',
        'f1_micro': 'f1_micro',
        'f1_samples': 'f1_samples',
        'f1_weighted': 'f1_weighted',
        'log_loss': 'log_loss',
        'precision_binary': 'Precision',
        'precision_macro': 'prec_macro',
        'precision_micro': 'prec_micro',
        'precision_samples': 'prec_samples',
        'precision_weighted': 'prec_weighted',
        'recall_binary': 'Recall',
        'recall_macro': 'rcl_macro',
        'recall_micro': 'rcl_micro',
        'recall_samples': 'rcl_samples',
        'recall_weighted': 'rcl_weighted',
        'roc_auc': 'ROC_AUC',
        # regression:
        'explained_variance': 'expl_var',
        'mean_absolute_error': 'MAE',
        'mean_squared_error': 'MSE',
        'mean_squared_log_error': 'MSLE',
        'median_absolute_error': 'MedAE',
        'root_mean_squared_error': 'RMSE',
        'rmse_over_stdev': r'RMSE/$\sigma_y$',
        'r2_score': '$R^2$',
        'r2_score_noint': '$R^2_{noint}$',
        'r2_score_adjusted': '$R^2_{adjusted}$',
        'r2_score_fitted': '$R^2_{fitted}$'
    }
    return nice_names
