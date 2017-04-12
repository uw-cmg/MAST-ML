import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import data_parser

#==========================================================================================
# Please download the index of 5 folders first: f1.csv, f2.csv, f3.csv, f4.csv, f5.csv
#   https://github.com/UWMad-Informatics/standardized/tree/master/TimeSlicesTest
#
#==========================================================================================


def get_folder_index(folderName):
    f = open(str(folderName))
    data = np.loadtxt(f, delimiter=',')
    dataTemp = data[:, np.newaxis]
    index = list(np.array(dataTemp).reshape(-1, ))
    index = [int(i - 1) for i in index]
    return index


def execute (model, data, savepath, *args, **kwargs):

    X = np.asarray(data.get_x_data())
    Y = np.asarray(data.get_y_data())

    #Get folder index.
    index1 = get_folder_index("f1.csv") #data index for folder 1
    index2 = get_folder_index("f2.csv") #data index for folder 2
    index3 = get_folder_index("f3.csv") #data index for folder 3
    index4 = get_folder_index("f4.csv") #data index for folder 4
    index5 = get_folder_index("f5.csv") #data index for folder 5
    #Combine the index
    index = [index1, index2, index3, index4, index5]
    index = np.asarray(index)

    #==========================================================================================
    #Key assessment summary numbers: 3 total numbers.
    #Train on Folders 1-3, Predict 4, Predict 5;
    #Train on Folders 1-4, Predict 5.
    #==========================================================================================

    rmse_list = []
    for i in [t+1 for t in list(range(4))]:
        if i == 1 or i == 2:
            continue;
        to_exclude = list(range(i))
        #Get index for training folders & test folder
        index_train = index[np.asarray(to_exclude).astype(int)];
        index_test = [element for k, element in enumerate(index) if k not in to_exclude]
        #train set starts with the first folder
        X_train = X[np.hstack(index_train)]
        Y_train = Y[np.hstack(index_train)]
        model.fit(X_train, Y_train)
        rmse_folder = []
        for item in index_test:
            folder_X = X[item]
            folder_Y = Y[item]
            # train on training sets
            Y_test_Pred = model.predict(folder_X)
            rmse = np.sqrt(mean_squared_error(folder_Y, Y_test_Pred))
            rmse_folder.append(rmse)
        rmse_list.append(rmse_folder)

    print('Train on Folders 1-3, Predict 4, Predict 5; rmse of folder 4 and 5 are %f and %f.' %(rmse_list[0][0], rmse_list[0][1]));
    print('Train on Folders 1-4, Predict 5; rmse of folder 5 is %f' %(rmse_list[1][0]));