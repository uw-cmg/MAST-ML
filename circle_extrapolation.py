__author__ = 'haotian'

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import data_parser as dp
from sklearn.metrics import mean_squared_error
import modelio


def execute(model, data, savepath, recursive=False, lwr_data_path = '../DBTT/CD_LWR_clean6.csv'):
    if not recursive:
        savepath = savepath.format(type(model).__name__+'_{}')
    ending = ''
    is_cd = False
    if 'CD' in data.y_feature:
        ending = ' on CD'
        is_cd = True
        lwr_data = dp.parse(lwr_data_path)
        lwr_data.set_x_features(data.x_features)
        lwr_data.set_y_feature('CD predicted delta sigma (Mpa)')

    groups, threshold = get_extrapolation_group(data, 'time')
    result = circle_test(model, data, savepath, groups)
    if is_cd:
        make_plot(threshold, result, savepath, 'log(time){}'.format(ending),
                  actual_rms=get_lwr_rmse(model, data, lwr_data))
    else:
        make_plot(threshold, result, savepath, 'log(time){}'.format(ending))

    groups, threshold = get_extrapolation_group(data, 'fluence')
    result = circle_test(model, data, savepath, groups)
    if is_cd:
        make_plot(threshold, result, savepath, 'log(fluence){}'.format(ending),
                  actual_rms=get_lwr_rmse(model, data, lwr_data))
    else:
        make_plot(threshold, result, savepath, 'log(fluence){}'.format(ending))
    if 'CD' not in data.y_feature:
        have_cd = data.set_y_feature('CD delta sigma')
        if have_cd:
            execute(model, data, savepath, recursive=True)


def get_lwr_rmse(model, train_data, lwrdata):
    day_to_sec = 24*60*60
    x0 = math.log10(80*365*day_to_sec)

    x = list(train_data.get_x_data())
    y = list(train_data.get_y_data())

    model.fit(x, y)
    all_rms = []
    time = lwrdata.get_data('Time(s)')
    time = list(set([x[0] for x in time]))
    time.sort()
    j = 1
    for i in time:
        lwrdata.remove_all_filters()
        lwrdata.add_inclusive_filter('Time(s)', '=', i)
        #print (len(lwrdata.get_y_data()))
        y_predict = model.predict(lwrdata.get_x_data())
        plt.clf()
        plt.plot(lwrdata.get_y_data(), y_predict, 'ro')
        plt.title('log(time) = {}'.format(math.log10(i/day_to_sec)))
        plt.plot([0,400],[0,400])
        plt.xlabel('actual')
        plt.ylabel('predict')
        if j < 10:
            plt.savefig("../DBTT/model/rf_time_0{}.png".format(j))
        else:
            plt.savefig("../DBTT/model/rf_time_{}.png".format(j))
        j+=1
        rms = np.sqrt(mean_squared_error(lwrdata.get_y_data(), y_predict))
        lwrdata.remove_all_filters()
        #print(rms)
        all_rms.append(rms)
    distance = [x0-math.log10(i) for i in time]
    #print (all_rms)
    #print (distance)
    return distance, all_rms


def get_extrapolation_group(data, type):
    day_to_sec = 24*60*60
    x_features = data.get_data(['log(fluence)', 'log(flux)', 'Temp (C)'])
    flux = [i[1] for i in x_features]
    fluence = [i[0] for i in x_features]
    temperature = [i[2] for i in x_features]
    time = [10**fluence[i]/10**flux[i] for i in range(len(flux))]
    logtime = [math.log10(i) for i in time]
    x = math.log10(3e10)
    y = math.log10(80*365*day_to_sec)

    if type == 'time':
        distance = [((logtime[i]-y)**2+(flux[i]-x)**2)**0.5 for i in range(len(flux))]
    else:
        y = math.log10((10**x)*(10**y))
        distance = [((fluence[i]-y)**2+(flux[i]-x)**2)**0.5 for i in range(len(flux))]
    sorted_distance = list(distance)
    sorted_distance.sort()
    indexes = []
    for i in range(4):
        indexes.append(sorted_distance[int(len(sorted_distance)/5*(i+1))])
    groups = [[],[],[],[],[]]
    #print (groups)
    for i in range(len(distance)):
        if distance[i] < indexes[0]: groups[4].append(i)
        elif distance[i] < indexes[1]: groups[3].append(i)
        elif distance[i] < indexes[2]: groups[2].append(i)
        elif distance[i] < indexes[3]:groups[1].append(i)
        else: groups[0].append(i)
    return groups, indexes


def actual_figure(model, data, savepath):
    day_to_sec = 24*60*60
    data.set_x_features(['log(fluence)', 'log(flux)', 'Temp (C)'])
    x_features = data.get_x_data()
    flux = [i[1] for i in x_features]
    fluence = [i[0] for i in x_features]
    temperature = [i[2] for i in x_features]
    time = [10**fluence[i]/10**flux[i] for i in range(len(flux))]
    logtime = [math.log10(i) for i in time]
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.plot(x,y,z, '.')
    #ax.plot([10.477], [19.879], [270], 'or')
    x = math.log10(3e10)
    y = math.log10(80*365*day_to_sec)

    distance = [((logtime[i]-y)**2+(flux[i]-x)**2)**0.5 for i in range(len(flux))]
    sorted_distance = list(distance)
    sorted_distance.sort()
    indexes = []
    for i in range(4):
        indexes.append(sorted_distance[int(len(sorted_distance)/5*(i+1))])
    print (indexes)
    #indexes = [y-7.5,y-7,y-6.5,y-6]
    groups = [[],[],[],[],[]]
    for i in range(len(distance)):
        if distance[i] < indexes[0]: groups[4].append(i)
        elif distance[i] < indexes[1]: groups[3].append(i)
        elif distance[i] < indexes[2]: groups[2].append(i)
        elif distance[i] < indexes[3]: groups[1].append(i)
        else: groups[0].append(i)

    plt.plot(flux, logtime, '.')
    plt.plot(x, y, 'ro')
    circles = [plt.Circle((x, y),indexes[i],color='r', fill=False) for i in range(4)]
    #circle=plt.Circle((math.log10(3e10),math.log10(80*365)),indexes[0],color='r', fill=False)
    for circle in circles:
        plt.gcf().gca().add_artist(circle)
    plt.xlabel('log(flux(n/cm2s)')
    plt.ylabel('log(time(second))')
    print (y)
    plt.show()
    # plt.savefig(savepath.format("circle/act".format(i+1)), dpi=200, bbox_inches='tight')
    plt.clf()

    return groups


def circle_test(model, data, savepath, groups):

    x_lists = [[], [], [], [], []]
    y_lists = [[], [], [], [], []]

    data.set_x_features(['N(Cu)','N(Ni)','N(Mn)','N(P)','N(Si)','N( C )','N(log(fluence)','N(log(flux)','N(Temp)'])
    x_features = data.get_x_data()
    y_features = data.get_y_data()

    for i in range(5):
        for j in groups[i]:
            x_lists[i].append(x_features[j])
            y_lists[i].append(y_features[j])

    result = [[], [], [], [], []]
    for i in range(5):
        x = []
        y = []
        for j in range(i+1):
            x = x+x_lists[j]
            y = y+y_lists[j]
        model.fit(x, y)

        for j in range(5):
            #print ('predict {}, test {}'.format(i+1,j+1))
            '''
            if j < i:
                continue
            '''
            y_predict = model.predict(x_lists[j])
            rms = np.sqrt(mean_squared_error(y_lists[j], y_predict))
            result[i].append(rms)

    return result


def make_plot(threshold, result, savepath, title, actual_rms=None):
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rcParams.update({'weight': 'bold'})

    x = []
    interval = (threshold[-1]-threshold[0])/(len(threshold)-1)
    for i in range(len(threshold)-1):
        x.append((threshold[i]+threshold[i+1])/2)
    x.append(x[-1]+interval)
    x.insert(0, x[0]-interval)
    x.reverse()
    plt.clf()
    label = 1
    for y in result:
        plt.plot(x, y, marker='o', label='{}'.format(label*20))
        label += 1
    if actual_rms:
        plt.plot(actual_rms[0],actual_rms[1], marker = 'o', label='lwr Δy')
    plt.title('circle test on log flux vs. {}'.format(title))
    plt.xlabel('predicted radius')
    plt.ylabel('rmse')
    #if not actual_rms:
    plt.gcf().gca().set_ylim([0,160])
    plt.gcf().gca().set_xlim([-0.10,x[0]*1.2])
    plt.legend(bbox_to_anchor=(1, 0.9),bbox_transform=plt.gcf().transFigure, title = 'training data %')
    #plt.show()
    plt.savefig(savepath.format("circle_log(flux)_vs_{}".format(title)))
    plt.clf()

