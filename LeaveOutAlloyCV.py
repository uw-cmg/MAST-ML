import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from mean_error import mean_error
import csv


def execute(model, data, savepath, *args, **kwargs):

    rms_list = []
    alloy_list = []
    me_list = []
   
    #TTM modify alloys since they are no longer integers
    alloys_nested = data.get_data("Alloy")
    import itertools
    chain = itertools.chain.from_iterable(alloys_nested)
    alloys = list(chain)
    alloys = list(set(alloys))
    alloys.sort()

    print(alloys)
    #for alloy in range(1, max(data.get_data("Alloy"))[0] + 1):
    for alloy in alloys:
        print(alloy)
        # fit model to all alloys except the one to be removed
        data.remove_all_filters()
        data.add_exclusive_filter("Alloy", '=', alloy)
        model.fit(data.get_x_data(), np.array(data.get_y_data()).ravel())

        # predict removed alloy
        data.remove_all_filters()
        data.add_inclusive_filter("Alloy", '=', alloy)
        if len(data.get_x_data()) == 0: continue  # if alloy doesn't exist(x data is empty), then continue
        Ypredict = model.predict(data.get_x_data())

        rms = np.sqrt(mean_squared_error(Ypredict, np.array(data.get_y_data()).ravel()))
        me = mean_error(Ypredict, np.array(data.get_y_data()).ravel())
        rms_list.append(rms)
        alloy_list.append(alloy)
        me_list.append(me)
    print('Mean RMSE: ', np.mean(rms_list))

    # graph rmse vs alloy 
    fig, ax = plt.subplots(figsize=(10, 4))
    #TTM+4 alloy len
    #plt.xticks(np.arange(0, max(alloy_list) + 1, 5))
    plt.xticks(np.arange(0, len(alloys) + 1, 5))
    ax.scatter(np.arange(0, len(alloy_list)), rms_list, color='black', s=10)
    ax.set_xticklabels(alloy_list)
    #ax.plot((0, max(data.get_data("Alloy"))[0]), (0, 0), ls="--", c=".3")
    #TTM+1
    ax.plot((0, len(alloys)), (0, 0), ls="--", c=".3")
    ax.set_xlabel('Alloy Number')
    ax.set_ylabel('RMSE (Mpa)')
    ax.set_title('Leave out Alloy')
    ax.text(.05, .88, 'Mean RMSE: {:.2f} MPa'.format(np.mean(rms_list)), fontsize=14, transform=ax.transAxes)
    ax.text(.05, .82, 'Mean Mean Error: {:.2f} MPa'.format(np.mean(me_list)), fontsize=14, transform=ax.transAxes)
    for x in np.argsort(rms_list)[-5:]:
        #ax.annotate(s = alloy_list[x],xy = (alloy_list[x], rms_list[x]))
        ax.annotate(s = alloy_list[x],xy = (x, rms_list[x]))
    fig.savefig(savepath.format(ax.get_title()), dpi=200, bbox_inches='tight')
    fig.clf()
    plt.close()

    with open(savepath.replace(".png", "").format("Leave Out Alloy.csv"), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        x = ["Alloy Number", "RMSE"]
        writer.writerow(x)
        for i in zip(alloy_list,rms_list):
            writer.writerow(i)
