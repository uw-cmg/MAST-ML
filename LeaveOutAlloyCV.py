import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


def execute(model, data, savepath, *args):

    rms_list = []
    alloy_list = []

    for alloy in range(1, max(data.get_data("Alloy"))[0] + 1):

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
        rms_list.append(rms)
        alloy_list.append(alloy)

    print('Mean RMSE: ', np.mean(rms_list))

    # graph rmse vs alloy 
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.xticks(np.arange(0, max(alloy_list) + 1, 5))
    ax.scatter(alloy_list, rms_list, color='black', s=10)
    ax.plot((0, max(data.get_data("Alloy"))[0]), (0, 0), ls="--", c=".3")
    ax.set_xlabel('Alloy Number')
    ax.set_ylabel('RMSE (Mpa)')
    ax.set_title('Leave out Alloy')
    ax.text(.05, .88, 'Mean RMSE: {:.2f}'.format(np.mean(rms_list)), fontsize=14, transform=ax.transAxes)
    for x in np.argsort(rms_list)[-5:]:
        ax.annotate(s = alloy_list[x],xy = (alloy_list[x], rms_list[x]))
    fig.savefig(savepath.format(ax.get_title()), dpi=200, bbox_inches='tight')
    fig.clf()
    plt.close()
