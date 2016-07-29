import matplotlib.pyplot as plt
import math
import matplotlib
import numpy as np

__author__ = 'haotian'


def calculate_distance(data, feature, threshold, target):
    x = data.get_data(feature)
    distances = []
    if target<threshold:
        for i in x:
            distances.append(threshold - i[0])
    else:
        for i in x:
            distances.append(i[0]-threshold)
    return distances


def feature_extrapolation(data, model, feature, threshold, target, savepath):
    data.remove_all_filters()
    if target<threshold:
        data.add_inclusive_filter(feature, '>=', threshold)
    else:
        data.add_inclusive_filter(feature, '<=', threshold)
    model.fit(data.get_x_data(),data.get_y_data())
    data.remove_all_filters()
    y_predict = model.predict(data.get_x_data())
    if not isinstance(y_predict[0], float):
        y_predict = [x[0] for x in y_predict]
    y_actual = data.get_y_data()
    if not isinstance(y_actual[0], float):
        y_actual = [x[0] for x in y_actual]
    error = [abs(y_predict[i] - y_actual[i]) for i in range(len(y_predict))]
    #distances = calculate_distance(data, feature, threshold, target)
    #distances = [max(0, i) for i in distances]
    distances = data.get_data(feature)
    distances = [i[0] for i in distances]

    '''
    calculate average of each x
    '''
    x = list(set(distances))
    x.sort()
    y = []
    for i in range(len(x)):
        y.append([])
    for i in range(len(error)):
        y[x.index(distances[i])].append(error[i])
    avg = []
    for i in range(len(y)):
        avg.append(sum(y[i])/len(y[i]))

    '''
    calculate folder average
    '''
    folder = calculate_threshold(data, feature, cut = 10)
    folder_x = []
    folder_y = []
    folder_x.append(min(distances))
    for i in folder:
        folder_x.append(i)
        folder_x.append(i)
    folder_x.append(max(distances))
    folded_data = []
    for i in range(len(folder)+1):
        size = 0
        total = 0
        for j in range(len(distances)):
            if ((i == len(folder) and distances[j] >= folder[i-1]) or
                        (i < len(folder) and distances[j] < folder[i] and (i == 0 or distances[j] >= folder[i-1]))):
                total += error[j]
                size += 1
        folder_y.append(total/size)
        folder_y.append(total/size)

    '''
    calculate linear fit
    '''
    if target>threshold:
        linear_x_all = [max(threshold, i) for i in distances]
    else:
        linear_x_all = [min(threshold, i) for i in distances]
    linear_x = []
    linear_err = []
    for i in range(len(linear_x_all)):
        if linear_x_all[i] != threshold:
            linear_x.append(linear_x_all[i])
            linear_err.append(error[i])
    z = np.polyfit(linear_x, linear_err, 1)
    p = np.poly1d(z)

    plt.clf()
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rcParams.update({'font.weight': 'bold'})
    plt.plot([target, target], [0, max(error)], '--r',label='LWR')
    plt.plot([threshold, threshold], [0, max(error)], '--', label='threshold')
    plt.plot(distances, error, '.')
    plt.plot(x, avg, label = 'average')
    plt.plot([min(linear_x),max(linear_x)],[p(min(linear_x)),p(max(linear_x))], linewidth=2.0, label = 'linear trend')
    plt.plot(folder_x, folder_y,linewidth=2.5, label = 'folder avg')
    plt.title('[{}] extrapolation toward lwr \nw/ thres={}'.format(feature, round(threshold,2)))
    plt.xlabel('distance')
    plt.ylabel('abs(err)')
    plt.gcf().gca().set_xlim([min(target,min(distances))-0.1,max(target, max(distances))+0.1])
    if target>threshold:
        plt.legend(bbox_to_anchor=(0.12, 0.9), loc=2, bbox_transform=plt.gcf().transFigure,fontsize=10)
    else:
        plt.legend(bbox_to_anchor=(1, 0.9),bbox_transform=plt.gcf().transFigure, fontsize=10)

    plt.savefig(savepath.format("{}_{}_extrapolation_thres_{}".format(type(model).__name__, feature,round(threshold,2))))
    plt.clf()


def calculate_radius(data, features, center, name):
    x_data = data.get_data(features)
    distances = []
    for i in range(len(x_data)):
        radius = 0
        for j in range(len(x_data[i])):
            radius += (x_data[i][j]-center[j])**2
        radius **= 0.5
        distances.append(radius)
    data.add_feature(name, distances)


def calculate_threshold(data,feature, cut=5):
    x_data = data.get_data(feature)
    x_data = [i[0] for i in x_data]
    x_data.sort()
    thresholds = []
    for i in range(cut-1):
        thresholds.append(x_data[int(len(x_data)*(i+1)/cut)])
    return thresholds


def execute(model, data, savepath, cd=False, *args):
    year_to_sec = 365*24*3600
    if not cd:
        calculate_radius(data, ['log(flux)','log(fluence)'], [math.log10(3e10),math.log10(3e10*80*year_to_sec)],
                        'flux_flu_radius')
        calculate_radius(data, ['log(flux)', 'log(time)'], [math.log10(3e10), math.log10(80*year_to_sec)],
                        'flux_time_radius')
    thresholds = calculate_threshold(data, 'flux_flu_radius')
    for i in thresholds:
        feature_extrapolation(data, model, 'flux_flu_radius', i, 0, savepath)

    thresholds = calculate_threshold(data, 'flux_time_radius')
    for i in thresholds:
        feature_extrapolation(data, model, 'flux_time_radius', i, 0, savepath)

    thresholds = calculate_threshold(data, 'log(flux)')
    for i in thresholds:
        feature_extrapolation(data, model, 'log(flux)', i, math.log10(3e10), savepath)

    thresholds = calculate_threshold(data, 'log(fluence)')
    for i in thresholds:
        feature_extrapolation(data, model, 'log(fluence)', i, math.log10(3e10*80*year_to_sec), savepath)

    '''
    for i in fluence_divisions:
        feature_extrapolation(data, model, 'log(fluence)', math.log10(i), math.log10(3e10*80*year_to_sec), savepath)
    '''
    thresholds = calculate_threshold(data, 'log(time)')
    for i in thresholds:
        feature_extrapolation(data, model, 'log(time)', i, math.log10(80*year_to_sec), savepath)

    if not cd:
        data.set_y_feature('CD delta sigma')
        execute(model, data, savepath.format('CD_{}'),cd=True)



