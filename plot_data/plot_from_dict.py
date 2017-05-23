import numpy as np
from sklearn.metrics import mean_squared_error
import plot_data.plot_xy as plotxy

def plot_group_splits_with_outliers(group_dict=None, outlying_groups=list(), label="group_splits", group_notelist=list(), addl_kwargs=dict(),
    *args, **kwargs):
    """Plot from a data dictionary.
        Converts into plot_xy.multiple_overlay
    """
    xdatalist=list()
    ydatalist=list()
    labellist=list()
    xerrlist=list()
    yerrlist=list()
    otherxdata=list()
    otherxerrdata=list()
    otherydata=list()
    groups = list(group_dict.keys())
    groups.sort()
    show_rmse = 0
    for group in groups:
        if group in outlying_groups:
            xdatalist.append(group_dict[group]['xdata'])
            xerrlist.append(group_dict[group]['xerrdata'])
            ydatalist.append(group_dict[group]['ydata'])
            yerrlist.append(None)
            labellist.append(group)
            if 'rmse' in group_dict[group].keys():
                show_rmse = 1 # if any RMSE shown, do RMSE for remaining
                rmse = group_dict[group]['rmse']
                group_notelist.append('{:<1}: {:.2f}'.format(group, rmse))
        else:
            otherxdata.extend(group_dict[group]['xdata'])
            otherxerrdata.extend(group_dict[group]['xerrdata'])
            otherydata.extend(group_dict[group]['ydata'])
    if len(otherxdata) > 0:
        xdatalist.insert(0,otherxdata) #prepend
        xerrlist.insert(0,otherxerrdata)
        ydatalist.insert(0,otherydata)
        yerrlist.insert(0,None)
        labellist.insert(0,"All others")
        if show_rmse == 1:
            all_other_rmse = np.sqrt(mean_squared_error(otherydata, otherxdata))
            group_notelist.append('{:<1}: {:.2f}'.format("All others", all_other_rmse))
    kwargs=dict()
    kwargs['xlabel'] = "X"
    kwargs['ylabel'] = "Y"
    kwargs['save_path'] = label
    kwargs['xdatalist'] = xdatalist
    kwargs['ydatalist'] = ydatalist
    kwargs['stepsize'] = 1
    kwargs['xerrlist'] = xerrlist
    kwargs['yerrlist'] = yerrlist
    kwargs['labellist'] = labellist
    kwargs['notelist'] = group_notelist
    kwargs['plotlabel'] = label
    kwargs['guideline'] = 1
    for addl_kwarg in addl_kwargs.keys():
        kwargs[addl_kwarg] = addl_kwargs[addl_kwarg]
    plotxy.multiple_overlay(**kwargs) 
    return
