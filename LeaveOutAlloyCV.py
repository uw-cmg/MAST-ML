import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from mean_error import mean_error
import csv
import matplotlib

def execute(model, data, savepath, *args, **kwargs):

    rms_list = []
    alloy_list = []
    me_list = []
   

    for alloy in range(1, max(data.get_data("alloy_number"))[0] + 1):
        print(alloy)
        # fit model to all alloys except the one to be removed
        data.remove_all_filters()
        data.add_exclusive_filter("alloy_number", '=', alloy)
        model.fit(data.get_x_data(), np.array(data.get_y_data()).ravel())

        # predict removed alloy
        data.remove_all_filters()
        data.add_inclusive_filter("alloy_number", '=', alloy)
        if len(data.get_x_data()) == 0: continue  # if alloy doesn't exist(x data is empty), then continue
        Ypredict = model.predict(data.get_x_data())

        rms = np.sqrt(mean_squared_error(Ypredict, np.array(data.get_y_data()).ravel()))
        me = mean_error(Ypredict, np.array(data.get_y_data()).ravel())
        rms_list.append(rms)
        alloy_list.append(alloy)
        me_list.append(me)
    print('Mean RMSE: ', np.mean(rms_list))
    print('Mean Mean Error: ', np.mean(me_list))

    # graph rmse vs alloy 
    matplotlib.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.xticks(np.arange(0, max(alloy_list) + 1, 5))
    ax.scatter(alloy_list, rms_list, color='black', s=10)
    ax.plot((0, max(data.get_data("alloy_number"))[0]), (0, 0), ls="--", c=".3")
    ax.set_xlabel('Alloy Number')
    ax.set_ylabel('RMSE (MPa)')
    #ax.set_title('Leave out Alloy')
    ax.text(.5, .90, 'Mean RMSE: {:.2f} MPa'.format(np.mean(rms_list)), transform=ax.transAxes, fontsize=0.85*matplotlib.rcParams['font.size'])
    for x in np.argsort(rms_list)[-5:]:
        ax.annotate(s = alloy_list[x],xy = (alloy_list[x], rms_list[x]))
    fig.savefig("Leave_Out_Alloy", dpi=200, bbox_inches='tight')
    fig.clf()
    plt.close()

    with open("Leave_Out_Alloy.csv", 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        x = ["Alloy Number", "RMSE"]
        writer.writerow(x)
        for i in zip(alloy_list,rms_list):
            writer.writerow(i)
